import tkinter as tk
from tkinter import filedialog, messagebox
import sys
from pathlib import Path
from scipy.io import wavfile
import mne
import numpy as np
from scipy.signal import correlate, find_peaks, resample_poly
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import os
import logging
import subprocess
import json
import tempfile
import re

logger = logging.getLogger("audio_eeg_sync")

# pulls up the file explorer and selects a file
def select_file(
    title: str, 
    filetype: list[tuple[str, str]]
) -> Path:
    
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
def select_dir(
    title: str, 
    initial_dir=None
) -> Path:
    
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Select Directory", title)
    dir_path = filedialog.askdirectory(title=title, initialdir=initial_dir)
    root.destroy()

    if not dir_path:
        print("No directory selected")
        sys.exit(0)
    return Path(str(dir_path))

# helper function for confirming directory
def confirm_or_select_dir(
    guessed_path: Path, 
    title: str, 
    dir_type: str
) -> Path: 
    
    correct_dir = messagebox.askyesno(
        "Confirm Directory",
        f"Is the following {dir_type} directory correct? \n {guessed_path}"
    )
    if correct_dir is True:
        return guessed_path
    else:
        correct_dir = select_dir(title=title, initial_dir=guessed_path)
        return correct_dir

# find the first stimulus as the experiment start and the last stimulus
def find_experiment_times(
    events: np.ndarray, 
    event_id: dict[str,int]
)-> tuple[int, int]:
    
    stimulus_codes = []
    for label in event_id:
        if label.startswith("Stimulus"):
            stimulus_codes.append(event_id[label])
    stimulus_rows = events[np.isin(events[:,2], stimulus_codes)]
    first_stim_sample = stimulus_rows[0,0]
    last_stim_sample = stimulus_rows[-1,0]
    return first_stim_sample, last_stim_sample

# make sure vhdr file calls out the same name vmrk file
def prep_vhdr(
    vhdr_path_object: Path
) -> Path:
    
    exp_name = vhdr_path_object.stem
    expected_vmrk = f"{exp_name}.vmrk"
    expected_eeg = f"{exp_name}.eeg"

    with open(vhdr_path_object, "r") as f:
        lines = f.readlines()
    
    correct = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("MarkerFile="):
            # check if the correct name is in there
            if exp_name in line:
                correct = correct +1
            else:
                full_vmrk_path = vhdr_path_object.parent / expected_vmrk
                lines[i] = f"MarkerFile={full_vmrk_path}\n"
        elif line.strip().startswith("DataFile="):
            if exp_name in line:
                correct = correct +1
            else:
                full_eeg_path = vhdr_path_object.parent / expected_eeg
                lines[i] = f"DataFile={full_eeg_path}\n"
   
    # if both paths are correct, just return the original path object
    if correct == 2:
        return vhdr_path_object
    
    temp_vhdr_file = tempfile.NamedTemporaryFile(delete=False, suffix=".vhdr")
    
    with open(temp_vhdr_file.name, "w") as f:
        f.writelines(lines)
    
    return Path(temp_vhdr_file.name)

# parse the brain vision vhdr and vmrk file to get the pulses array and sampling frequency 
def parse_brainvision(
    vhdr_path_object: Path, 
    use_pulse_flag: bool, 
    response_id="Response/R257"
) -> tuple[datetime, float, float, np.ndarray | None]:
    
    vhdr_path = prep_vhdr(vhdr_path_object=vhdr_path_object)
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False)
    sfreq = raw.info['sfreq']
    bv_file_start = raw.info['meas_date']   # UTC time
    logger.info(f"Found brainvision file start {bv_file_start}")
    events, event_id = mne.events_from_annotations(raw)
    # event_id -> {'Reponse name': number of responses }
    # events -> [sample_index, 0, event_code]
    # get the start and end time of experiment
    start_sample, end_sample= find_experiment_times(events, event_id)
    start_time = start_sample/sfreq
    end_time = end_sample/sfreq + 10   # pad the end time with 10 seconds after last stimulus
    
    logger.debug(f"Brainvision task start time: {start_time}")
    logger.debug(f"Brainvision task end time: {end_time}")
    
    if use_pulse_flag:
        response_code = event_id[response_id]
        response_rows = events[events[:,2] == response_code]
        response_samples = response_rows[:,0]
        response_times = response_samples/sfreq
        logger.debug(f"Brainvision sync pulse array size {len(response_times)}")
    else:
        response_times = None

    return bv_file_start, start_time, end_time, response_times

# return the raw audio file list from the audio path object
def get_audio_files(
    audio_path_object: Path, 
    use_pulse_flag: bool
) -> tuple[list[Path], list[Path] | None]:
    
    wav_files = list(audio_path_object.glob("*.wav"))
    file_numbers = []
    audio_pulse_files = None
    for file_obj in wav_files:
        stem = file_obj.stem
        match_start = re.match(r"^(\d+)-", stem)
        match_end = re.search(r"-(\d+)$",stem)
        
        if match_start:
            file_num = int(match_start.group(1))
            file_numbers.append((file_obj, file_num))
        elif match_end: 
            file_num = int(match_end.group(1))
            file_numbers.append((file_obj, file_num))
    if not file_numbers:
        logger.error(f"No wav files recognized with matching naming pattern in {audio_path_object}")
        raise ValueError(f"No wav files recognized in {audio_path_object}")
    # get the raw audio files
    min_prefix = min(num for obj, num in file_numbers)
    raw_audio_files = [obj for obj, num in file_numbers if num==min_prefix]
    logger.info(f"Found {len(raw_audio_files)} raw audio files")

    if use_pulse_flag:
        # get the pulse audio files
        max_prefix = max(num for obj, num in file_numbers)
        audio_pulse_files = [obj for obj, num in file_numbers if num==max_prefix]
        logger.info(f"Found {len(audio_pulse_files)} audio pulse files")

    return raw_audio_files, audio_pulse_files

# get date time from filename
def get_datetime(
    file_obj: Path, 
    date_format="%y%m%d_%H%M"
) -> datetime:
    
    stem = file_obj.stem
    match = re.search(r"\d{6}_\d{4}", stem)
    if not match:
        logger.error(f"Could not find a valid date time in the wav audio file name in {file_obj}")
        raise ValueError(f"No valid datetime in wav audio file names in {file_obj}")
    
    datetime_obj = datetime.strptime(match.group(), date_format)
    return datetime_obj

# sort the files chronologically
def sort_files_chronologically(
    file_list: list[Path]
) -> list[Path]:
    
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
def stitch_files(
    sorted_file_list: list[Path]
) -> tuple[int, np.ndarray]:
    
    voltage_chunks = []
    for file in sorted_file_list:
        sfreq, voltages= wavfile.read(str(file))
        voltage_chunks.append(voltages)
    
    stitched_voltages = np.concatenate(voltage_chunks)
    return sfreq, stitched_voltages

# parse the audio file to get the pulses array 
def parse_audio(
    audio_path_object: Path, 
    use_pulse_flag: None | bool
) -> tuple[datetime | None, int, np.ndarray] | tuple[datetime | None, int, np.ndarray, np.ndarray, np.ndarray]:
    
    raw_audio_files, audio_pulse_files= get_audio_files(audio_path_object=audio_path_object, use_pulse_flag=use_pulse_flag) 

    if len(raw_audio_files) == 1:
        audio_file_start = get_audio_dt(audio_path=raw_audio_files[0])
        logger.info(f"Found datetime object for audio file start: {audio_file_start}")
        sfreq_raw, stitched_voltages_raw = stitch_files(sorted_file_list=raw_audio_files)
 
    else:
        raw_audio_sorted = sort_files_chronologically(file_list=raw_audio_files)
        audio_file_start = get_audio_dt(audio_path=raw_audio_sorted[0])
        logger.info(f"Found datetime object for audio file start: {audio_file_start}")
        sfreq_raw, stitched_voltages_raw = stitch_files(sorted_file_list=raw_audio_sorted)
        logger.info("Finished stitching raw audio files")
        
    if use_pulse_flag:
        if len(audio_pulse_files) != len(raw_audio_files):
            logger.warning("Number of audio pulse files doesn't match number of raw audio files")
        if len(audio_pulse_files)==1:
            sfreq, stitched_voltages_pulses = stitch_files(sorted_file_list=audio_pulse_files)
        else:  
            audio_pulse_sorted = sort_files_chronologically(file_list=audio_pulse_files) 
            sfreq, stitched_voltages_pulses = stitch_files(sorted_file_list=audio_pulse_sorted)
        
        logger.info("Finished stitching audio pulse files")

        if sfreq != sfreq_raw:
            logger.error("Sampling frequency of audio files do not match")
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

        return audio_file_start, sfreq, pulses_confirmed_times, stitched_voltages_pulses, stitched_voltages_raw
    else:
        return audio_file_start, sfreq_raw, stitched_voltages_raw
# get the local datetime of reaper audio file creation from the metadata
def get_audio_dt(
    audio_path: Path
) -> datetime | None: 
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format_tags",
        "-of", "json",
        str(audio_path)
    ]
    out = subprocess.check_output(cmd, text=True)
    logger.debug(f"Looking for audio file datetime, found out: {out}")
    tags = json.loads(out)["format"].get("tags", {})
    logger.debug(f"Looking for audio file datetime, found tags: {tags}")
    date_str = tags.get("date")
    time_str = tags.get("creation_time")
    if date_str is None or time_str is None:
        logger.warning(f"Did not find any datetime in audio file path")
        return None
    dt = datetime.strptime(f"{date_str} {time_str}","%Y-%m-%d %H-%M-%S")
    return dt.replace(tzinfo=timezone.utc)

# perform the correlation to get the line of best fit b/w the two aligned time arrays for bv and audio
def find_time_alignment(
    bv_pulse_times: np.ndarray, 
    audio_pulse_times: np.ndarray
) -> tuple[float, float]:
    
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

def apply_alignment(
    rel_clock_rates: float, 
    offset: float, 
    bv_time: float, 
    audio_sfreq: int
) -> int:
    
    audio_time = (bv_time - offset)/rel_clock_rates
    audio_sample = int(round(audio_time*audio_sfreq))
    return audio_sample

# save wav file with predicted beep time and 2 second buffer
def save_predicted_beep(
    beep_time: float, 
    audio_voltages: np.ndarray, 
    sfreq: int, 
    output_path: Path, 
    buffer_sec=10
) -> None:
    
    start_sample = int(round((beep_time - buffer_sec)*sfreq))
    end_sample = int(round((beep_time+buffer_sec)*sfreq))
    if start_sample < 0 or end_sample>len(audio_voltages):
        message = (
            f"Predicted beep window out of bounds: "
            f"start={start_sample}, end={end_sample}, "
            f"signal_length={len(audio_voltages)}"
            )
        logger.error(message)
        raise ValueError(message)
    beep_segment = audio_voltages[start_sample:end_sample]
    wavfile.write(output_path, sfreq, beep_segment)

# find the best guess window for waveform matching beep estimation
def find_window(
    audio_file_start: datetime, 
    bv_file_start: datetime, 
    exp_start_time: float, 
    buffer_sec: float= 120.0
) -> tuple[float, float]:
    
    audio_file_start_utc = audio_file_start.astimezone(timezone.utc)
    exp_delta = timedelta(seconds=exp_start_time)
    logger.debug(f"Find window: exp delta: {exp_delta}")
    exp_absolute_time = bv_file_start + exp_delta
    logger.debug(f"Find window: exp absolute time: {exp_absolute_time}")
    audio_search_middle = (exp_absolute_time - audio_file_start_utc).total_seconds()
    logger.debug(f"Find window: audio search middle: {audio_search_middle}")
    audio_search_start = audio_search_middle - buffer_sec
    audio_search_end = audio_search_middle +buffer_sec
    logger.debug(f"Found the window for the audio beep matching, start: {audio_search_start} end: {audio_search_end}")
    return audio_search_start, audio_search_end

# estimate the beep with normalized cross correlation of beep template in audio file
def beep_matching_all(
    voltages_raw: np.ndarray, 
    audio_sfreq: int, 
    beep_file: Path,
    min_gap_sec: float=300.0,
    downsample_factor: int=10,
    local_window_sec: float=10.0

) -> np.ndarray:
    
    logger.info("Performing beep matching all")
    _, voltages_beep = wavfile.read(str(beep_file))

    if voltages_beep.ndim > 1:
        voltages_beep = voltages_beep[:,0]
    # remove leading and trailing zeros from beep
    beep_nonzero_indices = np.nonzero(voltages_beep)[0]
    voltages_beep = voltages_beep[beep_nonzero_indices[0]:beep_nonzero_indices[-1]+1]
    
    audio_ds = resample_poly(voltages_raw, up=1, down=downsample_factor)
    beep_ds = resample_poly(voltages_beep, up=1, down=downsample_factor)

    audio_ds = audio_ds - np.mean(audio_ds)
    beep_ds = beep_ds - np.mean(beep_ds)

    corr = np.abs(correlate(audio_ds, beep_ds, mode="valid", method="fft"))

    coarse_threshold = .7 * np.max(corr)
    min_gap_samples_ds = int(round((min_gap_sec * audio_sfreq) / downsample_factor))

    peak_samples_ds, _ = find_peaks(
        corr, 
        height=coarse_threshold,
        distance = min_gap_samples_ds
    )

    candidate_times_sec = (peak_samples_ds * downsample_factor) / audio_sfreq

    refined_times = []
    refined_scores = []

    buffer_samples = int(round(local_window_sec * audio_sfreq))

    for candidate_time in candidate_times_sec:
        center_sample = int(round(candidate_time * audio_sfreq))
        start_sample = max(0, center_sample - buffer_samples)
        end_sample = min(len(voltages_raw), center_sample + buffer_samples)

        audio_window = voltages_raw[start_sample:end_sample]

        if len(audio_window) < len(voltages_beep):
            continue

        best_index, scores = normalize_cross_correlation(
            audio_voltages=audio_window,
            template_voltages=voltages_beep
        )

        best_score = scores[best_index]

        if best_score >= 0.75:
            refined_time = (start_sample + best_index) / audio_sfreq
            refined_times.append(refined_time)
            refined_scores.append(best_score)

    if len(refined_times) == 0:
        return np.array([])

    refined_times = np.array(refined_times)
    refined_scores = np.array(refined_scores)

    sort_idx = np.argsort(refined_scores)[::-1]
    refined_times = refined_times[sort_idx]

    final_times = []
    for beep_time in refined_times:
        if len(final_times) == 0:
            final_times.append(beep_time)
        else:
            if np.min(np.abs(np.array(final_times) - beep_time)) > min_gap_sec:
                final_times.append(beep_time)

    return np.array(sorted(final_times))

# estimate the beep with normalized cross correlation of beep template in audio file
def beep_matching_window(
    voltages_raw: np.ndarray, 
    audio_sfreq: int, 
    beep_file: Path, 
    audio_search_start: float, 
    audio_search_end: float
) -> float:
    
    _, voltages_beep = wavfile.read(str(beep_file))

    if voltages_beep.ndim > 1:
        voltages_beep = voltages_beep[:,0]

    # remove leading and trailing zeros from beep
    beep_nonzero_indices = np.nonzero(voltages_beep)[0]
    voltages_beep = voltages_beep[beep_nonzero_indices[0]:beep_nonzero_indices[-1]+1]

    # convert search start and end to samples
    start_sample = int(round(audio_search_start*audio_sfreq))
    end_sample = int(round(audio_search_end*audio_sfreq))

    # safety check for cropping window
    if start_sample < 0 or end_sample > len(voltages_raw) or start_sample >= end_sample:
        msg = f"Invalid search window: start={start_sample}, end={end_sample}, len={len(voltages_raw)}"
        logger.error(msg)
        raise ValueError(msg)
    
    # crop data around search window
    voltages_raw = voltages_raw[start_sample:end_sample]
    best_index, _ = normalize_cross_correlation(audio_voltages=voltages_raw, template_voltages=voltages_beep)
    start_time = (best_index/audio_sfreq)+audio_search_start

    logger.info(f"Old beep matching method found task beep at {start_time} in audio file")
    
    return start_time
    
# perform a normalized cross correlation of the template beep across the chosen audio segment 
def normalize_cross_correlation(
    audio_voltages: np.ndarray, 
    template_voltages: np.ndarray
) -> tuple[int, np.ndarray]:
    
    # zero-mean the template voltages
    template_voltages = template_voltages - template_voltages.mean()
    template_norm = np.linalg.norm(template_voltages)

    beep_length = len(template_voltages)
    corr_length = len(audio_voltages) - beep_length +1
    scores = np.empty(corr_length)

    # perform cross correlation
    for sample in range(corr_length):
        window = audio_voltages[sample: sample+beep_length]
        window = window - window.mean()
        numerator = np.dot(window, template_voltages)
        denominator = template_norm*(np.linalg.norm(window))
        if denominator == 0:
            denominator = denominator + 1e-12
        scores[sample] = numerator/denominator

    logger.debug(f"Normalized cross correlation highest value: {np.max(scores)}, highest abs value: {np.max(np.abs(scores))}")
    scores = np.abs(scores)
    best_index = int(np.argmax(scores))
    return best_index, scores


        