"""Builds and executes notebooks/01_results_walkthrough.ipynb (outputs included, so GitHub renders it)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

cells = [
    nbf.v4.new_markdown_cell(
        "# EMOT -- results walkthrough\n\n"
        "A guided read of the committed benchmark (`results/`). Every table and figure is computed from the JSON files; "
        "nothing is typed by hand. For the full write-up see `docs/results.md` and `docs/edge.md`."
    ),
    nbf.v4.new_code_cell(
        "import json, warnings\nfrom pathlib import Path\nimport numpy as np, pandas as pd\n"
        "from IPython.display import Image, display\n"
        "from emot.benchmark.aggregate import (load_tracking_results, load_detection_results, load_edge_benchmark,\n"
        "                                      sequence_table, mean_by_tracker)\n"
        "from emot.benchmark.replay import VARIANTS\n"
        "warnings.filterwarnings('ignore'); pd.set_option('display.width', 160)\n"
        "ROOT = Path('..') if Path('../results').exists() else Path('.')\n"
        "trk = load_tracking_results(ROOT / 'results'); det = load_detection_results(ROOT / 'results')\n"
        "seq = sequence_table(ROOT / 'results'); edge = pd.DataFrame(load_edge_benchmark(ROOT / 'results')['rows'])\n"
        "print(len(trk), 'tracking runs |', trk.sequence.nunique(), 'sequences |', trk.backbone.nunique(), 'backbones |', trk.tracker.nunique(), 'trackers')"
    ),
    nbf.v4.new_markdown_cell("## 1. The data: 7 MOT17 sequences, split by *measured* camera motion\n\nThe `camera` column comes from the optical-flow estimator, not from the dataset's description."),
    nbf.v4.new_code_cell("seq[['sequence','width','height','fps','frames','gt_tracks','mean_people_per_frame','camera_motion_px','camera']]"),
    nbf.v4.new_code_cell("display(Image(ROOT / 'docs/figures/camera_motion.png', width=900))"),
    nbf.v4.new_markdown_cell("## 2. The leaderboard\n\nMean over all sequences and backbones; detectors are shared, so differences are pure association."),
    nbf.v4.new_code_cell("lb = mean_by_tracker(trk).rename(index=lambda t: VARIANTS[t].label)\nlb.round(3)"),
    nbf.v4.new_markdown_cell("### Held-out check (tuned on MOT17-02, so drop it)"),
    nbf.v4.new_code_cell("mean_by_tracker(trk, exclude_sequences=('MOT17-02',)).rename(index=lambda t: VARIANTS[t].label).round(3)"),
    nbf.v4.new_markdown_cell("## 3. What each component buys\n\nPaired per-(sequence, backbone) comparisons: mean HOTA change and how often the first tracker wins."),
    nbf.v4.new_code_cell(
        "cams = dict(zip(seq.sequence, seq.camera))\n"
        "def paired(a, b, metric='hota', cam=None):\n"
        "    d = trk if cam is None else trk[trk.sequence.map(cams) == cam]\n"
        "    pa = d[d.tracker == a].set_index(['sequence','backbone'])[metric]\n"
        "    pb = d[d.tracker == b].set_index(['sequence','backbone'])[metric]\n"
        "    diff = (pa - pb).dropna(); return round(diff.mean(), 4), int((diff > 0).sum()), len(diff)\n"
        "rows = [('CMC vs ByteTrack, moving', *paired('botsort_cmc','bytetrack',cam='moving')),\n"
        "        ('CMC vs ByteTrack, static', *paired('botsort_cmc','bytetrack',cam='static')),\n"
        "        ('OSNet vs ResNet18 (DeepSORT)', *paired('deepsort_osnet','deepsort')),\n"
        "        ('cascade vs single-stage', *paired('deepsort_osnet_cascade','deepsort_osnet')),\n"
        "        ('BoT-SORT full vs ByteTrack', *paired('botsort','bytetrack'))]\n"
        "pd.DataFrame(rows, columns=['comparison','mean ΔHOTA','wins','pairs'])"
    ),
    nbf.v4.new_code_cell("display(Image(ROOT / 'docs/figures/ablation_cmc_reid.png', width=900))\ndisplay(Image(ROOT / 'docs/figures/reid_separation.png', width=800))"),
    nbf.v4.new_markdown_cell("### Sequences are not independent samples\n\nSix backbones share each of the seven scenes, so a bootstrap over *scenes* is the honest uncertainty estimate. With only 7 scenes it is wide -- read the paired deltas as descriptive."),
    nbf.v4.new_code_cell(
        "rng = np.random.default_rng(0)\n"
        "def scene_deltas(a, b, metric='hota'):\n"
        "    d = trk.pivot_table(index='sequence', columns='tracker', values=metric)\n"
        "    return (d[a] - d[b]).values\n"
        "def boot(x, n=5000): return np.percentile([rng.choice(x, len(x)).mean() for _ in range(n)], [2.5, 97.5])\n"
        "for a, b in [('botsort','bytetrack'), ('deepsort_osnet','deepsort'), ('botsort_cmc','bytetrack')]:\n"
        "    x = scene_deltas(a, b); lo, hi = boot(x)\n"
        "    print(f'{VARIANTS[a].label} - {VARIANTS[b].label}: mean {x.mean():+.3f}, 95% bootstrap CI over scenes [{lo:+.3f}, {hi:+.3f}]')"
    ),
    nbf.v4.new_markdown_cell("## 4. Per-scene view"),
    nbf.v4.new_code_cell("display(Image(ROOT / 'docs/figures/heatmap_hota.png', width=900))\ndisplay(Image(ROOT / 'docs/figures/id_switches.png', width=900))"),
    nbf.v4.new_markdown_cell("## 5. Detectors and the edge"),
    nbf.v4.new_code_cell("main = edge[edge.group == 'main']\nmain.pivot_table(index='backbone', columns='variant', values='mean_ms').round(1)"),
    nbf.v4.new_code_cell("main.pivot_table(index='backbone', columns='variant', values='ap50').round(3)"),
    nbf.v4.new_code_cell("display(Image(ROOT / 'docs/figures/edge_latency.png', width=900))\ndisplay(Image(ROOT / 'docs/figures/edge_resolution.png', width=900))\ndisplay(Image(ROOT / 'docs/figures/edge_op_profile.png', width=900))"),
    nbf.v4.new_markdown_cell("---\nSources: `docs/results.md`, `docs/edge.md`, `docs/methodology.md`. Regenerate figures with `make figures`."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path("notebooks/01_results_walkthrough.ipynb")
NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()
nbf.write(nb, out)
print("wrote", out)
