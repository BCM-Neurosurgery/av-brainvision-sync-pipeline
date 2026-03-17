import logging
from tkinter import filedialog, messagebox
from pathlib import Path
import tkinter as tk
import sys

logger = logging.getLogger("audio_eeg_sync")

# pulls up the file explorer and selects a file
def select_file(
    root: tk.Tk,
    title: str, 
    filetype: list[tuple[str, str]]
) -> Path:
    
    file_path = filedialog.askopenfilename(
        parent=root,
        title=title,
        filetypes=filetype
    )

    if not file_path:
        print(f"No file selected for {filetype[0][0]}")
        sys.exit(0)
    return Path(str(file_path))

# pulls up the file explorer and selects a directory
def select_dir(
    root: tk.Tk,
    title: str, 
    initial_dir=None
) -> Path:
    
    messagebox.showinfo(
        "Select Directory", 
        title,
        parent=root)
    
    dir_path = filedialog.askdirectory(
        parent=root,
        title=title, 
        initialdir=initial_dir)

    if not dir_path:
        print("No directory selected")
        sys.exit(0)
    return Path(str(dir_path))

# helper function for confirming directory
def confirm_or_select_dir(
    root: tk.Tk,
    guessed_path: Path, 
    title: str, 
    dir_type: str
) -> Path: 
    
    correct_dir = messagebox.askyesno(
        "Confirm Directory",
        f"Is the following {dir_type} directory correct? \n {guessed_path}",
        parent=root
    )
    if correct_dir is True:
        return guessed_path
    else:
        correct_dir = select_dir(
            root=root,
            title=title, 
            initial_dir=guessed_path)
        return correct_dir

def select_analysis_method(
    root: tk.Tk
    )-> str:
    root.withdraw()

    window = tk.Toplevel(root)
    window.title("Select Analysis Method")
    window.geometry("600x400")

    # default method -> arduino beep matching 
    selection = tk.StringVar(value="") 
    tk.Label(
        window, 
        text="Choose an alignment method:"
    ).pack(pady=10)

    tk.Radiobutton(
        window, 
        text="Find Beeps with Arduino Matching (Only applicable to clinic visits after 02/12/2026)", 
        variable=selection, 
        value="arduino_beep_matching"
    ).pack()

    tk.Radiobutton(
        window, 
        text="Part 1: Find Beeps with Waveform Matching", 
        variable=selection, value="waveform_beep_matching"
    ).pack()

    tk.Radiobutton(
        window, 
        text="Part 2: Postprocessing Beeps from Waveform Matching (MUST do Part 1 First)", 
        variable=selection, value="waveform_postprocessing"
    ).pack()

    def on_run() -> None:
        if not selection.get():
            print("No analysis method selected")
            return
        window.destroy()

    tk.Button(
        window,
        text="Run",
        command=on_run
    ).pack(pady=10)

    window.wait_window()

    return selection.get()