from helper_functions import *
from cli_nbu_mvp import run_pipeline_gui

if __name__ == "__main__":
    
    # run the beep verification test
    run_test = True

    # # pull up the file explorer to choose the preprocessed file
    # vhdr_path_obj = select_file(
    #     title="Select a brain vision .vhdr file",
    #     filetype=[("Brain vision header", "*.vhdr")]
    # )
    # audio_path_obj = select_file(
    #     title="Select the audio .wav file with sync pulses (greatestNumber-*.wav)",
    #     filetype=[("Audio file", "*.wav")]
    # )
    # video_path_obj = select_dir(
    #     title="Select the directory containing the video files you want to align"
    # )
    # output_path_obj =select_dir(
    #     title="Select the directory you want to save the aligned files to"
    # )

    # # for testing -> manual directory paths
    # # PAAT -> need a test for this
    # vhdr_path_obj= Path("/Users/sophiapouya/Desktop/EEG/PAAT.vhdr")
    # audio_path_obj = Path("/Users/sophiapouya/Desktop/av-eeg-test/AUDIO/20260212/Media/03-260212_1141.wav")
    # video_path_obj = Path("/Users/sophiapouya/Desktop/av-eeg-test/VIDEO/20260212")
    # output_path_obj=Path("/Users/sophiapouya/workspace/bcm/av-brainvision-sync-pipeline/output")

    # # PRT
    # vhdr_path_obj= Path("/Users/sophiapouya/Desktop/EEG/PRT.vhdr")
    # audio_path_obj = Path("/Users/sophiapouya/Desktop/prt_av/AUDIO/20260212/Media/04-260212_1832.wav")
    # video_path_obj = Path("/Users/sophiapouya/Desktop/prt_av/VIDEO/20260212")
    # output_path_obj=Path("/Users/sophiapouya/workspace/bcm/av-brainvision-sync-pipeline/output")

    vhdr_path_obj= Path("/Users/sophiapouya/workspace/bcm/av-eeg-sync/reliability_test/av-eeg-sync-2.vhdr")
    audio_path_obj = Path("/Users/sophiapouya/workspace/bcm/av-eeg-sync/reliability_test/03-260205_1038.wav")
    video_path_obj = Path("/Users/sophiapouya/workspace/bcm/av-eeg-sync/reliability_test/")
    output_path_obj=Path("/Users/sophiapouya/workspace/bcm/av-brainvision-sync-pipeline/output")

    # get the exp start and end time and sync pulses from the brain vision file
    exp_start_time, exp_end_time, bv_sync_pulse_array = parse_brainvision(vhdr_path_object=vhdr_path_obj)

    # get the sync pulses from the audio file
    audio_sfreq, audio_sync_pulse_array = parse_audio(audio_path_object=audio_path_obj)

    # perform the correlation to get the relative clock rates and offset for line of best fit
    rel_clock_rates, offset = find_time_alignment(bv_pulse_times=bv_sync_pulse_array, audio_pulse_times=audio_sync_pulse_array)

    # find the start and end sample for audio 
    audio_start_sample = apply_alignment(rel_clock_rates=rel_clock_rates, offset=offset, bv_time=exp_start_time, audio_sfreq=audio_sfreq)
    audio_end_sample = apply_alignment(rel_clock_rates=rel_clock_rates, offset=offset, bv_time=exp_end_time, audio_sfreq=audio_sfreq)

    print("rel_clock_rates:", rel_clock_rates)
    print("offset:", offset)
    print("audio_start_sample:", audio_start_sample)
    print("audio_end_sample:", audio_end_sample)

    # verify the beep against the old system
    # optional test to see if beep from old way is found with equation  
    if run_test == True:
        audio_file_name = audio_path_obj.name.split("-")[-1]
        primary_audio_file_path = str(audio_path_obj.parent) + "/01-"+audio_file_name
        pat_beep_file = os.path.join("task_beeps", "PAAT.wav")

        matched_beep_time= estimate_beep(audio_file = primary_audio_file_path, beep_file=pat_beep_file, ds_factor=22)
        predicted_beep_time = (exp_start_time-offset)/rel_clock_rates
        diff = abs(matched_beep_time - predicted_beep_time)

        print(f"Verification Test:\n Matched Beep Time (Old Method): {matched_beep_time} seconds\n Predicted Beep Time: {predicted_beep_time} seconds")
        print(f"Predicted experiment start is {diff:.4f} seconds different from matched beep time")

    # run the audio video sync pipeline 
    audio_dir = audio_path_obj.parent
    run_pipeline_gui(
    audio_dir=audio_dir,
    video_dir=video_path_obj,
    out_dir=output_path_obj,
    site="jamail",
    audio_sample_start=audio_start_sample,
    audio_sample_end=audio_end_sample,
)



    




