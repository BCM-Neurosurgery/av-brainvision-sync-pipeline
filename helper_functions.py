import tkinter as tk
from tkinter import filedialog
import sys
from pathlib import Path
import scipy.io as sio


# pulls up the file explorer to select the preprocessed file
def select_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfile(
        title="Select a preprocessed file",
        filetypes=[("MAT Files", "*.mat")]
    )
    root.destroy()
    if file_path is None:
        print("No file selected. You must select a preprocessed .mat file")
        sys.exit(0)
    return Path(file_path)

# will grab the metadata from matlab file necessary for rest of pipeline
def get_metadata(file_path):
    metadata= {}
    file_contents = sio.loadmat(file_path)

