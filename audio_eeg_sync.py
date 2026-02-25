from helper_functions import *
from cli_nbu_mvp import run_pipeline_gui
import logging

if __name__ == "__main__":

    # run the beep verification test
    run_test = True

    # pull up the file explorer to choose the preprocessed file
    vhdr_path_obj = select_file(
        title="Select a brain vision .vhdr file",
        filetype=[("Brain vision header", "*.vhdr")]
    )
    audio_path_obj = select_dir(
        title="Select the directory containing the audio files you want to align",
        initial_dir=vhdr_path_obj.parent
    )
    video_path_obj = select_dir(
        title="Select the directory containing the video files you want to align",
        initial_dir=vhdr_path_obj.parent
    )
    output_path_obj =select_dir(
        title="Select the directory you want to save the aligned files to"
    )
    
    # set up logger
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    log_file_path = output_path_obj / f"audio_eeg_align_{timestamp}.log"
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
    exp_start_time, exp_end_time, bv_sync_pulse_array = parse_brainvision(vhdr_path_object=vhdr_path_obj)

    # get the sync pulses from the audio file
    audio_sfreq, audio_sync_pulse_array, stitched_voltages_pulses, stitched_voltages_raw= parse_audio(audio_path_object=audio_path_obj)

    # perform the correlation to get the relative clock rates and offset for line of best fit
    rel_clock_rates, offset = find_time_alignment(bv_pulse_times=bv_sync_pulse_array, audio_pulse_times=audio_sync_pulse_array)

    # find the start and end sample for audio 
    audio_start_sample = apply_alignment(rel_clock_rates=rel_clock_rates, offset=offset, bv_time=exp_start_time, audio_sfreq=audio_sfreq)
    audio_end_sample = apply_alignment(rel_clock_rates=rel_clock_rates, offset=offset, bv_time=exp_end_time, audio_sfreq=audio_sfreq)

    logger.debug(f"audio_start_sample: {audio_start_sample}")
    logger.debug(f"audio_end_sample: {audio_end_sample}")
    
    predicted_beep_time = (exp_start_time-offset)/rel_clock_rates
    logger.info(f"Predicted {vhdr_path_obj.stem} experiment start time in audio file: {predicted_beep_time} seconds")
    
    # optional test to output the +/- 2sec from predicted task beep time as wav file in output/
    if run_test == True:
        date_folder = vhdr_path_obj.parent.parent.name
        beep_path = output_path_obj / f"beep_segment_{vhdr_path_obj.stem}_{date_folder}.wav"
        save_predicted_beep(beep_time=predicted_beep_time, audio_voltages=stitched_voltages_raw, sfreq=audio_sfreq, output_path=beep_path)
        logger.info(f"Saved predicted beep segment to {output_path_obj}")

    # run the audio video sync pipeline 
    run_pipeline_gui(
    audio_dir=audio_path_obj,
    video_dir=video_path_obj,
    out_dir=output_path_obj,
    site="jamail",
    audio_sample_start=audio_start_sample,
    audio_sample_end=audio_end_sample,
)



    




