# av-brainvision-sync-pipeline

## Purpose
This pipeline synchronizes audio, video, and brain vision (EEG and task markers) data. 
- Expected inputs: BrainVision vhdr file and audio file (from pulse system input)
- Expected outputs: audio and video file cropped and aligned with BrainVision recording from an experiment 

## Repository Structure

```text
av-brainvision-sync-pipeline/
├── src/                          # main pipeline code
├── third_party/
│   └── video-sync-nbu/           # submodule for video/audio processing
├── environment.yml               # conda environment
├── pyproject.toml                # package definition
└── README.md
```

## Setup
1. Clone the Repository
`git clone <your-repo-url>`
`cd av-brainvision-sync-pipeline`

2. Initialize the Submodule
`git submodule update --init --recursive`

3. Create the Conda Environment
`conda env create -f environment.yml`
`conda activate av-sync-env`

4. Install FFmpeg
FFmpeg is required for audio/video processing.

- If you are using the provided conda environment, install FFmpeg from `conda-forge`:
    - `conda install -c conda-forge ffmpeg`

- If you are not using conda: 
    - Mac (Homebrew)
    `brew install ffmpeg`

    - Ubuntu/Linux
    `sudo apt update`
    `sudo apt install ffmpeg`

    - Windows
        1. Download FFmpeg zip archive
        2. Extract the archive to a permanent location
        3. Inside the extracted folder, locate the `bin` folder 
        4. Add that `bin` folder to your system `PATH`
        5. Open a new terminal or Command Prompt and verify the `ffmpeg -version` 


5. Install both Packages
`pip install -e .`
`pip install -e third_party/video-sync-nbu`

6. Run the Pipeline
`python -m av_bv_sync.main`