import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent / "third_party" / "video_sync_nbu_main"
sys.path.insert(0,str(PIPELINE_ROOT))

from scripts.cli.cli_nbu import run_pipeline
from scripts.log.logutils import configure_logging

def run_pipeline_gui(
    *,
    audio_dir: Path,
    video_dir: Path,
    out_dir: Path,
    site: str,
    audio_sample_start: int,
    audio_sample_end: int,
    log_level: str = "INFO",
) -> int:
    configure_logging(out_dir, log_level)

    return run_pipeline(
        audio_dir=audio_dir,
        video_dir=video_dir,
        out_dir=out_dir,
        site=site,
        segments=None,
        cameras=None,
        target_pairs=None,
        log_level=log_level,
        skip_decode=False,
        do_split=False,
        overwrite_clips=False,
        resume_from_segment=None,
        time_start=None,
        time_end=None,
        time_zone="UTC",
        audio_sample_start=int(audio_sample_start),
        audio_sample_end=int(audio_sample_end),
        run_id=None,
    )