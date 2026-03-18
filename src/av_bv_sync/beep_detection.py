from pathlib import Path
from scipy.io import wavfile
import soundfile as sf
import numpy as np
from scipy.signal import find_peaks, resample_poly, correlate

import logging
logger = logging.getLogger("audio_eeg_sync")

# save wav file with predicted beep time and 60 second buffer
def save_predicted_beep(
    beep_time: float, 
    audio_voltages: np.ndarray, 
    sfreq: int, 
    output_path: Path, 
    buffer_sec=60
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
    sf.write(output_path, beep_segment, sfreq)

# estimate the beep when you don't have any audio file info with a crude fft correlation for candidate times and then a normalized cross correlation of each candidate time
def beep_matching_all(
    voltages_raw: np.ndarray, 
    audio_sfreq: int, 
    beep_file: Path,
    min_gap_sec: float=60.0,
    downsample_factor: int=10,
    local_window_sec: float=10.0

) -> tuple[np.ndarray, np.ndarray]:
    
    logger.info("Performing beep matching all")
    sfreq_original, voltages_beep = wavfile.read(str(beep_file))

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

    coarse_threshold = .4 * np.max(corr)
    min_gap_samples_ds = int(round((min_gap_sec * audio_sfreq) / downsample_factor))

    peak_samples_ds, _ = find_peaks(
        corr, 
        height=coarse_threshold,
        distance = min_gap_samples_ds
    )

    candidate_times_sec = (peak_samples_ds * downsample_factor) / audio_sfreq
    logger.info(f"Coarse stage found {len(peak_samples_ds)} candidates")
    logger.info(f"Candidate times (sec): {candidate_times_sec}")

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

        if best_score >= 0.60:
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
    final_times = np.array(sorted(final_times))
    final_samples = np.round(final_times * sfreq_original).astype(int)
    return final_times, final_samples

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


        