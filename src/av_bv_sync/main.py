from av_bv_sync.ui import select_file, select_dir, confirm_or_select_dir
from av_bv_sync.brainvision import parse_brainvision
from av_bv_sync.audio import parse_audio
from av_bv_sync.alignment import find_window, find_time_alignment, apply_alignment
from av_bv_sync.beep_detection import beep_matching_all, beep_matching_window, save_predicted_beep
from av_bv_sync.video_sync_wrapper import run_pipeline_gui
from pathlib import Path
from datetime import datetime
import logging

# Remaining to dos
    # testing pipeline on at as many sessions as possible
    # test on windows
    # submodule or figure out solution for third party repo
    # bash script to run everything from desktop app

if __name__ == "__main__":

    # define repo absolute path
    PIPELINE_ROOT = Path(__file__).resolve().parents[2]
    ASSETS_DIR = PIPELINE_ROOT / "assets"

    # option for old waveform matching method (arduino independent)
    run_waveform_beep_matching = True

    # option for running arduino beep matching 
    run_arduino_beep_matching = False



    # pull up the file explorer to choose the preprocessed file
    vhdr_path_obj = select_file(
        title="Select a brain vision .vhdr file",
        filetype=[("Brain vision header", "*.vhdr")]
    )

    # guess the audio and video paths based on the initial file selection
    date_dir= vhdr_path_obj.parent.parent
    guessed_audio_path = date_dir / "audio"
    guessed_video_path = date_dir / "video"

    audio_path_obj = confirm_or_select_dir(
        guessed_path=guessed_audio_path,
        title="Select the directory containing the audio files you want to align",
        dir_type="audio"
    )
    video_path_obj = confirm_or_select_dir(
        guessed_path=guessed_video_path,
        title="Select the directory containing the video files you want to align",
        dir_type="video"
    )
    output_path_obj =select_dir(
        title="Select the directory you want to save the aligned files to"
    )
    
    # set up logger
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    log_file_dir = output_path_obj / "logs"
    log_file_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_file_dir / f"audio_eeg_align_{timestamp}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path)
        ]
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger = logging.getLogger("audio_eeg_sync")
    logger.info("Audio eeg alignment started")
    
    # get the exp start and end time and sync pulses from the brain vision file
    bv_start_time, exp_start_time, exp_end_time, bv_sync_pulse_array = parse_brainvision(vhdr_path_object=vhdr_path_obj, use_pulse_flag=run_arduino_beep_matching)

    # branch for running the arduino beep matching vs waveform matching
    if run_arduino_beep_matching == False:
        audio_file_start, audio_sfreq, stitched_voltages_raw= parse_audio(audio_path_object=audio_path_obj, use_pulse_flag=run_arduino_beep_matching)

    else:
        audio_file_start, audio_sfreq, audio_sync_pulse_array, stitched_voltages_pulses, stitched_voltages_raw= parse_audio(audio_path_object=audio_path_obj, use_pulse_flag=run_arduino_beep_matching)

        # perform the correlation to get the relative clock rates and offset for line of best fit
        rel_clock_rates, offset = find_time_alignment(bv_pulse_times=bv_sync_pulse_array, audio_pulse_times=audio_sync_pulse_array)

        # find the start and end sample for audio 
        audio_start_sample = apply_alignment(rel_clock_rates=rel_clock_rates, offset=offset, bv_time=exp_start_time, audio_sfreq=audio_sfreq)
        audio_end_sample = apply_alignment(rel_clock_rates=rel_clock_rates, offset=offset, bv_time=exp_end_time, audio_sfreq=audio_sfreq)

        logger.debug(f"audio_start_sample: {audio_start_sample}")
        logger.debug(f"audio_end_sample: {audio_end_sample}")
        
        predicted_beep_time = (exp_start_time-offset)/rel_clock_rates
        logger.info(f"Predicted {vhdr_path_obj.stem} experiment start time in audio file: {predicted_beep_time} seconds")
        
        output_beep_dir = output_path_obj / "beep_segments"
        output_beep_dir.mkdir(parents=True, exist_ok=True)
        beep_path = output_beep_dir / f"beep_segment_{vhdr_path_obj.stem}_{date_dir.name}.wav"
        save_predicted_beep(beep_time=predicted_beep_time, audio_voltages=stitched_voltages_raw, sfreq=audio_sfreq, output_path=beep_path)
        logger.info(f"Saved predicted beep segment to {beep_path}")

    # optional backup alignment method: use waveform matchingfor beeps
    if run_waveform_beep_matching: 
        # define task beep paths
        prt_paat_beep_path = ASSETS_DIR / "task_beeps" / "PAAT.wav"
        output_waveform_beep_dir = output_path_obj / "old_method_beep_segments"
        output_waveform_beep_dir.mkdir(parents=True, exist_ok=True)

        # output all the found beeps as wav files to a folder if there's no datetime object
        if audio_file_start is None:
            predicted_beep_times = beep_matching_all(voltages_raw=stitched_voltages_raw, audio_sfreq=audio_sfreq, beep_file=prt_paat_beep_path)
            for beep in predicted_beep_times:
                output_waveform_file = output_waveform_beep_dir / f"beep_segment_unknown_experiment_{date_dir.name}_{beep}.wav"
                save_predicted_beep(beep_time=beep, audio_voltages=stitched_voltages_raw, sfreq=audio_sfreq,output_path=output_waveform_file, buffer_sec=10)
                logger.info(f"Old waveform matching: saved {output_waveform_file}")
        
        else:
            # get the audio search window
            window_start_time, window_end_time = find_window(audio_file_start=audio_file_start, bv_file_start=bv_start_time, exp_start_time=exp_start_time)

            # find beeps
            logger.info("Running old beep matching method")
            predicted_matched_beep_time = beep_matching_window(voltages_raw=stitched_voltages_raw, audio_sfreq=audio_sfreq, beep_file=prt_paat_beep_path,
                                                        audio_search_start=window_start_time, audio_search_end=window_end_time)
            logger.info(f"Predicted beep time for old matching method is {predicted_matched_beep_time}")
            output_waveform_file = output_waveform_beep_dir / f"beep_segment_{vhdr_path_obj.stem}_{date_dir.name}.wav"
            save_predicted_beep(beep_time=predicted_matched_beep_time, audio_voltages=stitched_voltages_raw, sfreq=audio_sfreq,output_path=output_waveform_file, buffer_sec=10)
            logger.info(f"Old waveform matching: saved {output_waveform_file}")

#     # run the audio video sync pipeline 
#     run_pipeline_gui(
#     audio_dir=audio_path_obj,
#     video_dir=video_path_obj,
#     out_dir=output_path_obj,
#     site="jamail",
#     audio_sample_start=audio_start_sample,
#     audio_sample_end=audio_end_sample,
# )



    




