import tkinter as tk
from tkinter import filedialog
import sys
from pathlib import Path
import scipy.io as sio
from scipy.io import wavfile
import mne
import os
import numpy as np
from scipy.signal import correlate

# pulls up the file explorer 
def select_file(title, filetype):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetype
    )
    root.destroy()
    if file_path is None:
        print(f"No file selected for {filetype[0][0]}")
        sys.exit(0)
    return Path(str(file_path))

# find the first stimulus as the experiment start and the last stimulus
def find_experiment_times(events, event_id):
    stimulus_codes = []
    for label in event_id:
        if label.startswith("Stimulus"):
            stimulus_codes.append(event_id[label])
    stimulus_rows = events[np.isin(events[:,2], stimulus_codes)]
    first_stim_sample = stimulus_rows[0,0]
    last_stim_sample = stimulus_rows[-1,0]

    return first_stim_sample, last_stim_sample

# parse the brain vision vhdr and vmrk file to get the pulses array and sampling frequency 
def parse_brainvision(vhdr_path_object):
    raw = mne.io.read_raw_brainvision(vhdr_path_object, preload=False)
    sfreq = raw.info['sfreq']
    events, event_id = mne.events_from_annotations(raw)
    # event_id -> {'Reponse name': number of responses }
    # events -> [sample_index, 0, event_code]
    # get the start and end time of experiment
    start_sample, end_sample= find_experiment_times(events, event_id)
    start_time = start_sample/sfreq
    end_time = end_sample/sfreq + 10   # pad the end time with 10 seconds after last stimulus

    response_labels = [
        label for label in event_id if label.startswith("Response")
    ]
    response_numbers = []
    for label in response_labels:
        number_str = label.split("R")[-1].strip()
        response_numbers.append(int(number_str))
    response_idx = response_numbers.index(max(response_numbers))
    response_label = response_labels[response_idx]   

    response_code = event_id[response_label]
    response_rows = events[events[:,2] == response_code]
    response_samples = response_rows[:,0]
    response_times = response_samples/sfreq
    return start_time, end_time, response_times

# parse the audio file to get the pulses array 
def parse_audio(audio_path_object):
    sfreq, voltages = wavfile.read(audio_path_object)
    voltages_abs = np.abs(voltages) 
    threshold = 0.5*np.max(voltages_abs)
    # count pulses as voltages over threshold
    pulses_boolean_array = voltages_abs > threshold
    pulses_binary_array = pulses_boolean_array.astype(int)
    pulses_binary_padded = np.concatenate(([0], pulses_binary_array))
    # only count pulses as voltages that go from below threshold (0) to over threshold (1)
    counted_pulses_boolean = np.diff(pulses_binary_padded)==1
    pulses_confirmed_samples = np.where(counted_pulses_boolean)[0]  # row indices, column indicess
    pulses_confirmed_times = pulses_confirmed_samples/sfreq

    return sfreq, pulses_confirmed_times

# perform the correlation to get the line of best fit b/w the two aligned time arrays for bv and audio
def find_time_alignment(bv_pulse_times, audio_pulse_times):
    # take the diff of each array
    bv_times_diff = np.diff(bv_pulse_times)
    audio_times_diff= np.diff(audio_pulse_times)
    c = correlate(audio_times_diff, bv_times_diff, mode="full")
    max_index= int(np.argmax(c))
    best_lag = max_index - (len(bv_times_diff) - 1)

    # convert the lag into start indices
    if best_lag>=0:
        bv_start_idx = 0
        audio_start_idx = best_lag
    else:
        bv_start_idx = -1*best_lag
        audio_start_idx=0
    
    overlap_length = min(len(bv_pulse_times) - bv_start_idx, len(audio_pulse_times)-audio_start_idx)
    audio_aligned_times = audio_pulse_times[audio_start_idx:audio_start_idx+overlap_length]
    bv_aligned_times = bv_pulse_times[bv_start_idx:bv_start_idx+overlap_length]

    rel_clock_rates, offset = np.polyfit(audio_aligned_times, bv_aligned_times, 1)
    
    return rel_clock_rates, offset

def apply_alignment(rel_clock_rates, offset, bv_time, audio_sfreq):
    audio_time = (bv_time - offset)/rel_clock_rates
    audio_sample = int(round(audio_time*audio_sfreq))
    return audio_sample



