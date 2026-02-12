from helper_functions import *

# pull up the file explorer and get the file path for the preprocessed mat file
# get metadata from matlab file (brain vision sync marker array, brain vision start & stop???, )
# grab the sync pulse audio file and save the pulse array
# perform correlation on audio and brain vision pulse arrays (output-> t_bv = rel_clock_rate*t_audio + offset)
# use output of correlation to calculate start and stop time of audio (from start and stop time of matlab metadata?)
# use output of above into Yewen's script to align audio and video 


if __name__ == "__main__":
    # pull up the file explorer to choose the preprocessed file
    file_path = select_file()

    




