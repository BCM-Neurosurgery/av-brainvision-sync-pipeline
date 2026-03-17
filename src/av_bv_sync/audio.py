import numpy as np
import re
from pathlib import Path
from datetime import datetime, timezone
from scipy.io import wavfile
import subprocess
import json
import soundfile as sf

import logging
logger = logging.getLogger("audio_eeg_sync")

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
    
    datetime_to_file = {}

    for file in file_list:
        dt = get_datetime(file)
        if dt not in datetime_to_file:
            datetime_to_file[dt] = file
    
    sorted_pairs = sorted(datetime_to_file.items(), key=lambda pair:pair[0])
    
    sorted_files = [file for dt, file in sorted_pairs]

    return sorted_files   

# stitch files together
def stitch_files(
    sorted_file_list: list[Path]
) -> tuple[int, np.ndarray]:
    
    voltage_chunks = []

    for file in sorted_file_list:
        logger.info(f"Reading wav file: {file}")
        try:
            voltages, sfreq = sf.read(str(file))
        except Exception as e:
            logger.error(f"Failed reading wav file: {file}")
            logger.error(f"Error: {e}")
            raise

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

# quickly get audio file sampling frequency
def get_audio_sfreq(
    audio_file: Path
) -> int:
    return sf.info(audio_file).samplerate

