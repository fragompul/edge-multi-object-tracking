# Model card

## Models in this repository

| Model | Source | Licence | Used for |
| --- | --- | --- | --- |
| YOLOv8 n/s/m, YOLO11 n/s/m | Ultralytics, COCO-pretrained | AGPL-3.0 | person detection (class 0) |
| OSNet x0.25 / x0.5 (MSMT17) | Zhou et al. / `torchreid` authors, via Hugging Face `kaiyangzhou/osnet` | MIT | appearance embeddings |
| ResNet18 (ImageNet) | torchvision | BSD-3-Clause | baseline generic embeddings |
| `web/public/models/yolo11n-*.onnx` | exports of YOLO11n | AGPL-3.0 (derivative) | in-browser detection |

No model is trained or fine-tuned in this repository. Checkpoint hashes are pinned in `emot/reid/embedders.py`.

## Intended use

Counting and flow analysis of people in fixed or hand-held video; research on tracker design; edge-deployment studies.

## Out-of-scope use

Identifying individuals; cross-camera tracking; surveillance of persons without a lawful basis; any decision about a
person based on tracker output alone. See [Privacy & ethics](privacy-and-ethics.md).

## Performance

On MOT17 train (7 sequences), HOTA 0.35-0.40 averaged over detectors for the trackers here ([Results](results.md));
detection AP50 0.54-0.64 at 640 px depending on the backbone. Performance degrades with small/distant people, heavy
occlusion, night scenes and low resolution; raising input size helps more than raising model size.

## Known limitations

COCO-pretrained detectors, no fine-tuning; person class only; accuracy numbers are for this protocol and not
leaderboard-comparable; fairness across demographics, clothing or lighting was **not** evaluated.
