from pathlib import Path

import pytest

from emot.data.mot17 import (
    gt_to_mot_array,
    load_ground_truth,
    load_seqinfo,
    train_val_split,
)

SEQINFO_CONTENT = """[Sequence]
name=MOT17-TEST
imDir=img1
frameRate=30
seqLength=100
imWidth=1920
imHeight=1080
imExt=.jpg
"""

# frame, id, left, top, w, h, conf(consider_flag), class, visibility
GT_ROWS = [
    "1,1,10,10,50,100,1,1,0.9",
    "1,2,200,50,40,90,1,1,0.5",
    "2,1,15,10,50,100,1,1,0.9",
    "1,3,300,50,40,90,0,1,0.9",  # consider_flag=0 -> excluded
    "1,4,300,50,40,90,1,7,0.9",  # class 7 (not pedestrian) -> excluded
]


@pytest.fixture
def seq_dir(tmp_path: Path) -> Path:
    d = tmp_path / "MOT17-TEST"
    (d / "gt").mkdir(parents=True)
    (d / "seqinfo.ini").write_text(SEQINFO_CONTENT)
    (d / "gt" / "gt.txt").write_text("\n".join(GT_ROWS) + "\n")
    return d


def test_load_seqinfo(seq_dir: Path):
    info = load_seqinfo(seq_dir)
    assert info.name == "MOT17-TEST"
    assert info.frame_rate == 30
    assert info.seq_length == 100
    assert info.im_width == 1920
    assert info.im_height == 1080


def test_load_ground_truth_filters_invalid_and_non_pedestrian_rows(seq_dir: Path):
    df = load_ground_truth(seq_dir)
    # The class-7 row and the consider_flag=0 row must be excluded, leaving
    # 3 rows: id=1 in frames 1 and 2, id=2 in frame 1.
    assert len(df) == 3
    assert set(df["id"]) == {1, 2}
    assert (df["id"] == 1).sum() == 2


def test_min_visibility_filter(seq_dir: Path):
    df = load_ground_truth(seq_dir, min_visibility=0.6)
    # Only id=1's two rows (visibility 0.9) pass; id=2 (0.5) is dropped.
    assert set(df["id"]) == {1}


def test_gt_to_mot_array_shape_and_columns(seq_dir: Path):
    df = load_ground_truth(seq_dir)
    array = gt_to_mot_array(df, frame_idx=1)
    assert array.shape == (2, 5)  # 2 pedestrians in frame 1
    assert set(array[:, 0]) == {1, 2}


def test_gt_to_mot_array_empty_frame(seq_dir: Path):
    df = load_ground_truth(seq_dir)
    array = gt_to_mot_array(df, frame_idx=99)
    assert array.shape == (0, 5)


def test_train_val_split_is_temporal_not_random(seq_dir: Path):
    info = load_seqinfo(seq_dir)
    train, val = train_val_split(info, val_fraction=0.25)
    assert list(train) == list(range(1, 76))
    assert list(val) == list(range(76, 101))
    assert train[-1] < val[0]  # strictly temporal, no overlap
