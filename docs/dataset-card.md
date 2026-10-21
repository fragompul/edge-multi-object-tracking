# Dataset card: MOT17 (as used here)

- **What:** MOT17 (Milan, Leal-Taixé, Reid, Roth, Schindler; arXiv:1603.00831), the MOTChallenge pedestrian tracking benchmark.
- **Subset:** the 7 *training* sequences: MOT17-02, -04, -05, -09, -10, -11, -13 (5,330 frames, 640x480 to 1920x1080, 14-30 fps).
- **Source of the copy:** the public Hugging Face mirror `Lekim89/MOT17` (the official host was unreachable from the
  development environment); only the `-FRCNN` copy of each sequence is fetched since the three detector variants share images.
- **Annotations used:** pedestrian class, `consider` flag = 1, with visibility; no distractor handling.
- **Licence:** CC BY-NC-SA 3.0 (MOTChallenge). The repository's code is MIT; the clips, ground-truth overlays and detections in
  `web/public/data/` and the GIFs in `docs/assets/` are derived from MOT17 and carry its licence.
- **Contents of concern:** real people filmed in public places; faces are visible. The project produces no identities.
- **Splits:** a temporal split (trailing 25% of MOT17-02) for tuning only; evaluation on whole sequences.
