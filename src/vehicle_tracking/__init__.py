"""Vehicle tracking sub-package for Accident Intel.

Exposes the two primary entry points used by the analysis pipeline:

- :func:`process_video` — runs YOLO11x + BoT-SORT on a video file and writes an
  annotated output video together with a JSON trajectories file.
- :func:`identify_accident_candidates` — post-processes the trajectory data to
  score vehicle pairs by their accident suspicion level using overlap, relative
  closing motion, and abrupt deceleration heuristics.
"""

from .accident_candidates import identify_accident_candidates
from .tracker import process_video

__all__ = ["identify_accident_candidates", "process_video"]
