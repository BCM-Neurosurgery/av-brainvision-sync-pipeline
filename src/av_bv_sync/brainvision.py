import mne
from datetime import datetime
import numpy as np
import logging
from pathlib import Path
import tempfile

logger = logging.getLogger("audio_eeg_sync")

# find the first stimulus as the experiment start and the last stimulus
def find_experiment_times(
    events: np.ndarray, 
    event_id: dict[str,int]
)-> tuple[int, int]:
    
    stimulus_codes = []
    for label in event_id:
        if label.startswith("Stimulus"):
            stimulus_codes.append(event_id[label])
    stimulus_rows = events[np.isin(events[:,2], stimulus_codes)]
    first_stim_sample = stimulus_rows[0,0]
    last_stim_sample = stimulus_rows[-1,0]
    return first_stim_sample, last_stim_sample

# make sure vhdr file calls out the same name vmrk file
def prep_vhdr(
    vhdr_path_object: Path
) -> Path:
    
    exp_name = vhdr_path_object.stem
    expected_vmrk = f"{exp_name}.vmrk"
    expected_eeg = f"{exp_name}.eeg"

    with open(vhdr_path_object, "r") as f:
        lines = f.readlines()
    
    correct = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("MarkerFile="):
            # check if the correct name is in there
            if exp_name in line:
                correct = correct +1
            else:
                full_vmrk_path = vhdr_path_object.parent / expected_vmrk
                lines[i] = f"MarkerFile={full_vmrk_path}\n"
        elif line.strip().startswith("DataFile="):
            if exp_name in line:
                correct = correct +1
            else:
                full_eeg_path = vhdr_path_object.parent / expected_eeg
                lines[i] = f"DataFile={full_eeg_path}\n"
   
    # if both paths are correct, just return the original path object
    if correct == 2:
        return vhdr_path_object
    
    temp_vhdr_file = tempfile.NamedTemporaryFile(delete=False, suffix=".vhdr")
    
    with open(temp_vhdr_file.name, "w") as f:
        f.writelines(lines)
    
    return Path(temp_vhdr_file.name)

# parse the brain vision vhdr and vmrk file to get the pulses array and sampling frequency 
def parse_brainvision(
    vhdr_path_object: Path, 
    use_pulse_flag: bool, 
    response_id="Response/R257"
) -> tuple[datetime, float, float, np.ndarray | None]:
    
    vhdr_path = prep_vhdr(vhdr_path_object=vhdr_path_object)
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False)
    sfreq = raw.info['sfreq']
    bv_file_start = raw.info['meas_date']   # UTC time
    logger.info(f"Found brainvision file start {bv_file_start}")
    events, event_id = mne.events_from_annotations(raw)
    # event_id -> {'Reponse name': number of responses }
    # events -> [sample_index, 0, event_code]
    # get the start and end time of experiment
    start_sample, end_sample= find_experiment_times(events, event_id)
    start_time = start_sample/sfreq
    end_time = end_sample/sfreq + 10   # pad the end time with 10 seconds after last stimulus
    
    logger.debug(f"Brainvision task start time: {start_time}")
    logger.debug(f"Brainvision task end time: {end_time}")
    
    if use_pulse_flag:
        response_code = event_id[response_id]
        response_rows = events[events[:,2] == response_code]
        response_samples = response_rows[:,0]
        response_times = response_samples/sfreq
        logger.debug(f"Brainvision sync pulse array size {len(response_times)}")
    else:
        response_times = None

    return bv_file_start, start_time, end_time, response_times
