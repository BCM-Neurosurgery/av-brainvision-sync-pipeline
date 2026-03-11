import logging
from tkinter import filedialog, messagebox
from pathlib import Path
import tkinter as tk
import sys

logger = logging.getLogger("audio_eeg_sync")
# pulls up the file explorer and selects a file
def select_file(
    title: str, 
    filetype: list[tuple[str, str]]
) -> Path:
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetype
    )
    root.destroy()
    if not file_path:
        print(f"No file selected for {filetype[0][0]}")
        sys.exit(0)
    return Path(str(file_path))

# pulls up the file explorer and selects a directory
def select_dir(
    title: str, 
    initial_dir=None
) -> Path:
    
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Select Directory", title)
    dir_path = filedialog.askdirectory(title=title, initialdir=initial_dir)
    root.destroy()

    if not dir_path:
        print("No directory selected")
        sys.exit(0)
    return Path(str(dir_path))

# helper function for confirming directory
def confirm_or_select_dir(
    guessed_path: Path, 
    title: str, 
    dir_type: str
) -> Path: 
    
    correct_dir = messagebox.askyesno(
        "Confirm Directory",
        f"Is the following {dir_type} directory correct? \n {guessed_path}"
    )
    if correct_dir is True:
        return guessed_path
    else:
        correct_dir = select_dir(title=title, initial_dir=guessed_path)
        return correct_dir