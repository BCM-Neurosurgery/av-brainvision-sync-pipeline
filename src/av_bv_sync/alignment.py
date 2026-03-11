import numpy as np
from scipy.signal import correlate
import logging
from datetime import datetime, timedelta, timezone
logger = logging.getLogger("audio_eeg_sync")


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