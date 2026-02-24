import tkinter as tk
from tkinter import filedialog, messagebox
import sys
from pathlib import Path
from scipy.io import wavfile
import mne
import numpy as np
from scipy.signal import correlate
from datetime import datetime
import os
import logging

logger = logging.getLogger("audio_eeg_sync")

# pulls up the file explorer and selects a file
def select_file(title, filetype):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetype
    )
    root.destroy()
    if not file_path:
        print(f"No file selected for {filetype[0][0]}")
        sys.exit(0)
    return Path(str(file_path))

# pulls up the file explorer and selects a directory
def select_dir(title, initial_dir=None):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Select Directory", title)
    dir_path = filedialog.askdirectory(title=title, initialdir=initial_dir)
    root.destroy()

    if not dir_path:
        print("No directory selected")
        sys.exit(0)
    return Path(str(dir_path))

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
def parse_brainvision(vhdr_path_object, response_id="Response/R257"):
    raw = mne.io.read_raw_brainvision(vhdr_path_object, preload=False)
    sfreq = raw.info['sfreq']
    events, event_id = mne.events_from_annotations(raw)
    # event_id -> {'Reponse name': number of responses }
    # events -> [sample_index, 0, event_code]
    # get the start and end time of experiment
    start_sample, end_sample= find_experiment_times(events, event_id)
    start_time = start_sample/sfreq
    end_time = end_sample/sfreq + 10   # pad the end time with 10 seconds after last stimulus
    
    logger.debug(f"Brainvision task start time: {start_time}")
    logger.debug(f"Brainvision task end time: {end_time}")

    response_code = event_id[response_id]
    response_rows = events[events[:,2] == response_code]
    response_samples = response_rows[:,0]
    response_times = response_samples/sfreq

    logger.debug(f"Brainvision sync pulse array size {len(response_times)}")

    return start_time, end_time, response_times

# return the raw audio file list and audio pulse file list from the audio path object
def get_audio_files(audio_path_object):
    wav_files = list(audio_path_object.glob("*.wav"))
    numeric_prefix = [int(file_obj.name.split("-")[0]) for file_obj in wav_files]
    min_prefix = np.min(numeric_prefix)
    max_prefix = np.max(numeric_prefix)
    audio_pulse_files = [file_obj for file_obj in wav_files if max_prefix == int(file_obj.name.split("-")[0])]
    logger.info(f"Found {len(audio_pulse_files)} audio pulse files")
    raw_audio_files = [file_obj for file_obj in wav_files if min_prefix == int(file_obj.name.split("-")[0])]
    logger.info(f"Found {len(raw_audio_files)} raw audio files")
    if len(audio_pulse_files) != len(raw_audio_files):
        logger.warning("Number of audio pulse files doesn't match number of raw audio files")
    return raw_audio_files, audio_pulse_files

# get date time from filename
def get_datetime(file_obj, date_format="%y%m%d_%H%M"):
    file_date_time = file_obj.stem.split("-")[1]
    datetime_obj = datetime.strptime(file_date_time, date_format)
    return datetime_obj

# sort the files chronologically
def sort_files_chronologically(file_list):
    sorted_files = []
    file_datetimes = []
    for file in file_list:
        dt = get_datetime(file)
        file_datetimes.append((file,dt))
    
    sorted_file_datetimes = sorted(file_datetimes, key=lambda pair:pair[1])
    for file, dt in sorted_file_datetimes:
        sorted_files.append(file)
    return sorted_files   

# stitch files together
def stitch_files(sorted_file_list):
    voltage_chunks = []
    for file in sorted_file_list:
        sfreq, voltages= wavfile.read(file)
        voltage_chunks.append(voltages)
    
    stitched_voltages = np.concatenate(voltage_chunks)
    return sfreq, stitched_voltages

# parse the audio file to get the pulses array 
def parse_audio(audio_path_object):

    raw_audio_files, audio_pulse_files = get_audio_files(audio_path_object=audio_path_object)
    raw_audio_sorted = sort_files_chronologically(file_list=raw_audio_files)
    audio_pulse_sorted = sort_files_chronologically(file_list=audio_pulse_files)
    sfreq, stitched_voltages_pulses = stitch_files(sorted_file_list=audio_pulse_sorted)
    logger.info("Finished stitching audio pulse files")
    sfreq_raw, stitched_voltages_raw = stitch_files(sorted_file_list=raw_audio_sorted)
    logger.info("Finished stitching raw audio files")
    if sfreq != sfreq_raw:
        logger.warning("Sampling frequency of audio files do not match")
    voltages_abs = np.abs(stitched_voltages_pulses) 
    threshold = 0.5*np.max(voltages_abs)
    # count pulses as voltages over threshold
    pulses_boolean_array = voltages_abs > threshold
    pulses_binary_array = pulses_boolean_array.astype(int)
    pulses_binary_padded = np.concatenate(([0], pulses_binary_array))
    # only count pulses as voltages that go from below threshold (0) to over threshold (1)
    counted_pulses_boolean = np.diff(pulses_binary_padded)==1
    pulses_confirmed_samples = np.where(counted_pulses_boolean)[0]  # row indices, column indicess
    pulses_confirmed_times = pulses_confirmed_samples/sfreq
    logger.debug(f"Found {len(pulses_confirmed_times)} pulses in stitched audio pulse file")

    return sfreq, pulses_confirmed_times, stitched_voltages_pulses,stitched_voltages_raw

# perform the correlation to get the line of best fit b/w the two aligned time arrays for bv and audio
def find_time_alignment(bv_pulse_times, audio_pulse_times):
    # take the diff of each array
    bv_times_diff = np.diff(bv_pulse_times)
    audio_times_diff= np.diff(audio_pulse_times)
    logger.debug(f"Brainvision pulse diffs vector length: {len(bv_times_diff)} Range: {np.min(bv_times_diff)} to {np.max(bv_times_diff)}")
    logger.debug(f"Audio pulse diffs vector length: {len(audio_times_diff)} Range: {np.min(audio_times_diff)} to {np.max(audio_times_diff)}")
    logger.debug(f"Smallest 10 audio diffs: {np.sort(audio_times_diff)[:10]}")
    
    # z-score 
    bv_times_diff_norm = (bv_times_diff - np.mean(bv_times_diff)) / np.std(bv_times_diff)
    audio_times_diff_norm = (audio_times_diff - np.mean(audio_times_diff)) / np.std(audio_times_diff)

    c = correlate(audio_times_diff_norm, bv_times_diff_norm, mode="valid")
    audio_start_idx = int(np.argmax(c)) 
    logger.debug(f"Audio_start_idx (valid corr): {audio_start_idx}")
    logger.debug(f"Max corr: {float(np.max(c))}")

    num_bv_pulses = len(bv_pulse_times)
    audio_aligned_times = audio_pulse_times[audio_start_idx:audio_start_idx + num_bv_pulses]
    logger.info(f"First 5 Brainvision aligned times: {bv_times_diff[:5]}")
    logger.info(f"First 5 audio aligned times: {audio_times_diff[audio_start_idx:audio_start_idx+5]}")
    
    rel_clock_rates, offset = np.polyfit(audio_aligned_times, bv_pulse_times[:len(audio_aligned_times)], 1)
    logger.debug(f"Line of best fit metrics: Rel_clock_rates: {rel_clock_rates} Offset: {offset}")
    
    return rel_clock_rates, offset

def apply_alignment(rel_clock_rates, offset, bv_time, audio_sfreq):
    audio_time = (bv_time - offset)/rel_clock_rates
    audio_sample = int(round(audio_time*audio_sfreq))
    return audio_sample

# estimate the beep with the old method (waveform matching)
def estimate_beep(voltages_raw, audio_sfreq, beep_file, ds_factor=22):
    
    _, voltages_beep = wavfile.read(beep_file)

    # remove leading and trailing zeros from beep
    beep_nonzero_indices = np.nonzero(voltages_beep)[0]
    voltages_beep = voltages_beep[beep_nonzero_indices[0]:beep_nonzero_indices[-1]]

    # downsample voltages for audio and beep
    voltages_audio_ds = voltages_raw[::ds_factor]
    voltages_beep_ds = voltages_beep[::ds_factor]

    sfreq_ds = audio_sfreq/ds_factor

    # find the index where the beep occurs in the audio file
    corr = correlate(voltages_audio_ds, voltages_beep_ds, mode="valid")
    istart = int(np.argmax(corr))

    # convert that to time
    matched_start_time = istart/sfreq_ds
    
    return matched_start_time


    


