from helper_functions import *
from cli_nbu_mvp import run_pipeline_gui

# pull up the file explorer and get the file path for the preprocessed mat file 
# pull up the vmrk file 
# pull up the audio file
# find the audio file with the last number (audio file in local time (chicago) ex:2026-2)
# grab the sync pulse audio file and save the pulse array
# get metadata from matlab file 
    # event pulse markers -> brainvision.response_events (Timestamp, EVsample, alignment_response_signal) 
    # start time-> brainvision.brainvision_task_start_timestamp_unix (ms) (when task code 21 appears in event stream)
    # end time -> 10 seconds plus the last Events.Timestamp (ms)

# perform correlation on audio and brain vision pulse arrays (output-> t_bv = rel_clock_rate*t_audio + offset)
# use output of correlation to calculate start and stop time of audio in start sample number, end sample number, file name 
# use output of above into Yewen's script to align audio and video 

# react -> javascript front end 
# docker service?
# electron-> desktop apps 
# BV time
# wall clock time
# audio time -> samples since start of file (and properties of the file -> created time in local time (chicago))

# find the line of best fit to predict 

if __name__ == "__main__":
    
    # pull up the file explorer to choose the preprocessed file
    vhdr_path_obj = select_file(
        title="Select a brain vision .vhdr file",
        filetype=[("Brain vision header", "*.vhdr")]
    )
    audio_path_obj = select_file(
        title="Select the audio .wav file with sync pulses (greatestNumber-*.wav)",
        filetype=[("Audio file", "*.wav")]
    )
    video_path_obj = select_dir(
        title="Select the directory containing the video files you want to align"
    )

    output_path_obj =select_dir(
        title="Select the directory you want to save the aligned files to"
    )

    # # for testing -> manual directory paths
    # vhdr_path_obj= Path("/Users/sophiapouya/Desktop/EEG/PRT.vhdr")
    # audio_path_obj = Path("/Users/sophiapouya/Desktop/prt_av/AUDIO/20260212/Media/04-260212_1832.wav")
    # video_path_obj = Path("/Users/sophiapouya/Desktop/prt_av/VIDEO/20260212")
    # output_path_obj=Path("/Users/sophiapouya/workspace/bcm/av-brainvision-sync-pipeline/output")

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



    




