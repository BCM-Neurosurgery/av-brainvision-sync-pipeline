import tkinter as tk
from tkinter import filedialog
import sys
from pathlib import Path
import scipy.io as sio
from scipy.io import wavfile
import mne
import os

# pulls up the file explorer 
def select_file(title, filetype):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetype
    )
    root.destroy()
    if file_path is None:
        print(f"No file selected for {filetype[0][0]}")
        sys.exit(0)
    return Path(str(file_path))

# parse the brain vision vhdr and vmrk file to get the pulses array and sampling frequency 
def parse_brainvision(vhdr_path_object):
    raw = mne.io.read_raw_brainvision(vhdr_path_object, preload=False)
    sfreq = raw.info['sfreq']
    events, event_id = mne.events_from_annotations(raw)
    # event_id -> {'Reponse name': number of responses }
    # events -> [sample_index, 0, event_code]
    r257_code = event_id['Response/R257']
    r257_rows = events[events[:,2] == r257_code]
    r257_samples = r257_rows[:,0]
    
    return(sfreq, r257_samples)



