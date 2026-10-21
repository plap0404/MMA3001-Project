"""Tests for scripts/box_stats.py, using a small synthetic dataset."""

import pytest

from box_stats import format_table, read_boxes, summarise_class

# (width, height) as fractions of the image.
TINY = [(0.02, 0.02)] * 3          # 12.8 x 12.8 px at 640 -> small
BIG = [(0.9, 0.8)]                  # most of the image
MIXED = TINY + BIG


def write_split(root, split, files):
    labels = root / split / "labels"
    labels.mkdir(parents=True)
    for stem, text in files.items():
        (labels / f"{stem}.txt").write_text(text)


def test_reads_boxes_grouped_by_class(tmp_path):
    write_split(tmp_path, "train", {"a": "1 0.5 0.5 0.2 0.1\n3 0.5 0.5 0.9 0.8"})
    write_split(tmp_path, "test", {"b": "1 0.5 0.5 0.3 0.3"})
    boxes = read_boxes(tmp_path, ("train", "valid", "test"))
    assert sorted(boxes) == [1, 3]
    assert len(boxes[1]) == 2 and len(boxes[3]) == 1


def test_missing_split_is_skipped(tmp_path):
    write_split(tmp_path, "train", {"a": "0 0.5 0.5 0.1 0.1"})
    assert read_boxes(tmp_path, ("train", "valid", "test")) == {0: [(0.1, 0.1)]}


def test_polygon_line_raises(tmp_path):
    write_split(tmp_path, "train", {"a": "0 0.1 0.1 0.2 0.1 0.2 0.2"})
    with pytest.raises(ValueError):
        read_boxes(tmp_path, ("train",))


def test_median_area_is_width_times_height():
    assert summarise_class([(0.2, 0.1)], 640)["median_area_frac"] == pytest.approx(0.02)


def test_median_resists_a_few_large_boxes():
    summary = summarise_class(MIXED, 640)
    assert summary["median_area_frac"] < summary["mean_area_frac"]
    assert summary["median_area_frac"] == pytest.approx(0.0004)


def test_pixel_conversion_follows_imgsz():
    assert summarise_class([(0.05, 0.05)], 640)["median_w_px"] == pytest.approx(32.0)
    assert summarise_class([(0.05, 0.05)], 960)["median_w_px"] == pytest.approx(48.0)


def test_small_object_count_and_fraction():
    summary = summarise_class(MIXED, 640)
    assert summary["small_boxes"] == 3
    assert summary["small_fraction"] == pytest.approx(0.75)


def test_a_box_small_at_640_may_not_be_at_960():
    sizes = [(0.04, 0.04)]  # 25.6 px at 640, 38.4 px at 960
    assert summarise_class(sizes, 640)["small_boxes"] == 1
    assert summarise_class(sizes, 960)["small_boxes"] == 0


def test_empty_class_raises():
    with pytest.raises(ValueError):
        summarise_class([], 640)


def test_table_has_a_row_per_class():
    summaries = {"unsealed": summarise_class(BIG, 640), "loose-meat": summarise_class(TINY, 640)}
    lines = format_table(summaries, 640).splitlines()
    assert len(lines) == 4                      # header, separator, two classes
    assert lines[2].startswith("| loose-meat")  # sorted by name
