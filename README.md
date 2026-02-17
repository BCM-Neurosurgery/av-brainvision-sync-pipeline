# av-brainvision-sync-pipeline

## Purpose
This pipeline synchronizes audio, video, and brain vision (EEG and task markers) data. 
- Expected inputs: BrainVision vhdr file and audio file (from pulse system input)
- Expected outputs: audio and video file cropped and aligned with brain vision recording 

## Example
Execute audio_eeg_sync.py

## Requirements
- python=3.11
- scipy=1.17.0
- mne=1.11.0
