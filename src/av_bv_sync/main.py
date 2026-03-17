from av_bv_sync.ui import select_file, select_dir, confirm_or_select_dir, select_analysis_method
from av_bv_sync.brainvision import parse_brainvision
from av_bv_sync.audio import parse_audio, get_audio_sfreq
from av_bv_sync.alignment import find_time_alignment, apply_alignment
from av_bv_sync.beep_detection import beep_matching_all, save_predicted_beep
from av_bv_sync.video_sync_wrapper import run_pipeline_gui
from pathlib import Path
from datetime import datetime
import tkinter as tk
import logging

# Remaining to dos
    # bash script to run everything from desktop app

if __name__ == "__main__":

    # define repo absolute path
    PIPELINE_ROOT = Path(__file__).resolve().parents[2]
    ASSETS_DIR = PIPELINE_ROOT / "assets"

    # instantiate dialog window object
    app_root = tk.Tk()
    app_root.withdraw()

    # prompt user to select the desired analysis method
    selection = select_analysis_method(root=app_root)

    # set the booleans for selection method: 
    run_arduino_beep_matching = (selection == "arduino_beep_matching")
    run_waveform_beep_matching = (selection == "waveform_beep_matching")
    run_waveform_postprocess = (selection == "waveform_postprocessing")

    # option for old waveform matching method (arduino independent)
    if run_arduino_beep_matching:
        # pull up the file explorer to choose the preprocessed file
        vhdr_path_obj = select_file(
            root = app_root,
            title="Select a brain vision .vhdr file",
            filetype=[("Brain vision header", "*.vhdr")]
        )
        guessed_audio_path = vhdr_path_obj.parents[1] / "audio"
    elif run_waveform_postprocess:
        # find the beep file you want to align
        beep_path_obj = select_file(
            root = app_root,
            title="Select the unknown experiment beep segment you want to align",
            filetype=[("Audio file", "*.wav")]
        )
        
        # pull up the file explorer to choose the preprocessed file
        vhdr_path_obj = select_file(
            root = app_root,
            title="Select a brain vision .vhdr file",
            filetype=[("Brain vision header", "*.vhdr")]
        )
        guessed_audio_path = vhdr_path_obj.parents[1] / "audio"
    else:
        guessed_audio_path = None
        

    audio_path_obj = confirm_or_select_dir(
        root=app_root,
        guessed_path=guessed_audio_path,
        title="Select the directory containing the audio files you want to align",
        dir_type="audio"
    )
    video_path_obj = confirm_or_select_dir(
        root=app_root,
        guessed_path=audio_path_obj.parent / "video" / "FLIR",
        title="Select the directory containing the video files you want to align",
        dir_type="video"
    )
    output_path_obj =select_dir(
        root=app_root,
        title="Select the directory you want to save the aligned files to"
    )
    
    app_root.destroy()

    # set up logger
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    date_dir= audio_path_obj.parent.name
    patient_id = audio_path_obj.parents[2].name
    output_path_obj = output_path_obj / date_dir
    pipeline_output_dir = output_path_obj / "av_brainvision_sync_logs"
    log_file_dir = pipeline_output_dir / "logs"
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

    logger.info(f"Found date directory: {date_dir}")
    logger.info(f"Found patient id: {patient_id}")

    # branch for finding beeps with waveform matching
    if run_waveform_beep_matching:
        audio_file_start, audio_sfreq, stitched_voltages_raw= parse_audio(audio_path_object=audio_path_obj, use_pulse_flag=run_arduino_beep_matching)
        # define task beep paths
        prt_paat_beep_path = ASSETS_DIR / "task_beeps" / "PAAT.wav"
        output_waveform_beep_dir = pipeline_output_dir / "waveform_matched_beep_segments"
        output_waveform_beep_dir.mkdir(parents=True, exist_ok=True)

        # output all the found beeps as wav files to a folder if there's no datetime object
        predicted_beep_times, predicted_beep_samples = beep_matching_all(voltages_raw=stitched_voltages_raw, audio_sfreq=audio_sfreq, beep_file=prt_paat_beep_path)
        for i, beep in enumerate(predicted_beep_times):
            output_waveform_file = output_waveform_beep_dir / f"beep_segment_{patient_id}_unknown_exp_{date_dir}_startTime_{beep}_startSample_{predicted_beep_samples[i]}.wav"
            save_predicted_beep(beep_time=beep, audio_voltages=stitched_voltages_raw, sfreq=audio_sfreq,output_path=output_waveform_file)
            logger.info(f"Old waveform matching: saved {output_waveform_file}")
    
    # branch for saving audio/video segments from candidate waveform matched beeps
    elif run_waveform_postprocess:

        # get the start sample from the file name -> beep_segment_{patient_id}_unknown_exp_{date_dir}_startTime_##_startSample_##
        beep_file = beep_path_obj.stem
        parts = beep_file.split("_")
        audio_start_sample = int(parts[-1])

        # get the exp start and end time and sync pulses from the brain vision file
        bv_start_time, exp_start_time, exp_end_time, _ = parse_brainvision(vhdr_path_object=vhdr_path_obj, use_pulse_flag=run_arduino_beep_matching)
        experiment_time = exp_end_time - exp_start_time
        
        audio_sfreq = get_audio_sfreq(beep_path_obj)
        audio_end_sample = audio_start_sample + int(round((experiment_time*audio_sfreq)))
        logger.info(f"Found audio start sample at {audio_start_sample}")
        logger.info(f"Found audio end sample at {audio_end_sample}")

        # # go ahead and run the a/v sync pipeline directly with audio start sample and audio end sample
        logger.info(f"Running video sync pipeline, results will be saved to {output_path_obj}")
        run_pipeline_gui(
            audio_dir=audio_path_obj,
            video_dir=video_path_obj,
            out_dir=output_path_obj / vhdr_path_obj.stem,
            site="jamail",
            audio_sample_start=audio_start_sample,
            audio_sample_end=audio_end_sample,
        )

    # branch for running arduino pulse matching
    else:
        # get the exp start and end time and sync pulses from the brain vision file
        bv_start_time, exp_start_time, exp_end_time, bv_sync_pulse_array = parse_brainvision(vhdr_path_object=vhdr_path_obj, use_pulse_flag=run_arduino_beep_matching)

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
        
        output_beep_dir = pipeline_output_dir / "arduino_matched_beep_segments"
        output_beep_dir.mkdir(parents=True, exist_ok=True)
        beep_path = output_beep_dir / f"beep_segment_{patient_id}_{vhdr_path_obj.stem}_{date_dir}_startSample_{audio_start_sample}_endSample_{audio_end_sample}.wav"
        save_predicted_beep(beep_time=predicted_beep_time, audio_voltages=stitched_voltages_raw, sfreq=audio_sfreq, output_path=beep_path)
        logger.info(f"Saved predicted beep segment to {beep_path}")

        # go ahead and run the a/v sync pipeline directly with audio start sample and audio end sample
        logger.info(f"Running video sync pipeline, results will be saved to {output_path_obj}")
        run_pipeline_gui(
            audio_dir=audio_path_obj,
            video_dir=video_path_obj,
            out_dir=output_path_obj / vhdr_path_obj.stem,
            site="jamail",
            audio_sample_start=audio_start_sample,
            audio_sample_end=audio_end_sample,
        )





    




