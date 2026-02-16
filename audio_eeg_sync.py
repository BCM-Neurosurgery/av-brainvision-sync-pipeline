from helper_functions import *

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

    # get the exp start and end time and sync pulses from the brain vision file
    exp_start_time, exp_end_time, bv_sync_pulse_array = parse_brainvision(vhdr_path_object=vhdr_path_obj)

    # get the sync pulses from the audio file
    audio_sfreq, audio_sync_pulse_array = parse_audio(audio_path_object=audio_path_obj)

    # perform the correlation to get the relative clock rates and offset for line of best fit
    rel_clock_rates, offset = find_time_alignment(bv_pulse_times=bv_sync_pulse_array, audio_pulse_times=audio_sync_pulse_array)

    # find the start and end sample for audio 
    audio_start_sample = apply_alignment(rel_clock_rates=rel_clock_rates, offset=offset, bv_time=exp_start_time, audio_sfreq=audio_sfreq)
    audio_end_sample = apply_alignment(rel_clock_rates=rel_clock_rates, offset=offset, bv_time=exp_end_time, audio_sfreq=audio_sfreq)



    




