"""Tests for scripts/filter_labels.py, using a tiny fake dataset."""

import pytest
import yaml

from filter_labels import (
    filter_label_lines,
    filter_split,
    parse_label_line,
    polygon_to_box,
    write_data_yaml,
)

NAMES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]


def make_split(split_dir, files):
    """Create a fake split: one dummy image and one label file per entry."""
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "labels").mkdir(parents=True)
    for stem, text in files.items():
        (split_dir / "images" / f"{stem}.jpg").write_bytes(b"fake image")
        (split_dir / "labels" / f"{stem}.txt").write_text(text)


def test_removes_only_dropped_class():
    lines = ["1 0.5 0.5 0.1 0.1", "4 0.2 0.2 0.05 0.01", "3 0.5 0.5 1 1"]
    assert filter_label_lines(lines, {4})[0] == ["1 0.5 0.5 0.1 0.1", "3 0.5 0.5 1 1"]


def test_other_boxes_unchanged_and_in_order():
    lines = ["3 0.1 0.2 0.3 0.4", "0 0.5 0.6 0.7 0.8"]
    assert filter_label_lines(lines, {4})[0] == lines


def test_blank_lines_ignored():
    assert filter_label_lines(["", "   ", "2 0.1 0.1 0.1 0.1"], {4})[0] == ["2 0.1 0.1 0.1 0.1"]


def test_malformed_line_raises():
    with pytest.raises(ValueError):
        filter_label_lines(["1 0.5 0.5"], {4})


def test_split_counts_and_no_class_4_left(tmp_path):
    make_split(tmp_path / "in", {
        "mixed": "1 0.5 0.5 0.1 0.1\n4 0.2 0.2 0.05 0.01",   # keeps 1 box
        "wrinkle_only": "4 0.2 0.2 0.05 0.01",              # dropped
        "normal": "",                                        # kept, empty
        "no_newline": "3 0.5 0.5 1 1",                       # like Roboflow files
    })
    counts = filter_split(tmp_path / "in", tmp_path / "out", {4}, drop_left_empty=True)

    assert counts == {"kept": 3, "dropped": 1, "boxes_removed": 2, "polygons": {}}
    out_labels = sorted((tmp_path / "out" / "labels").glob("*.txt"))
    out_images = sorted((tmp_path / "out" / "images").glob("*.jpg"))
    assert [p.stem for p in out_labels] == [p.stem for p in out_images]
    for label in out_labels:
        for line in label.read_text().splitlines():
            assert line.split()[0] != "4"


def test_wrinkle_only_image_kept_when_option_off(tmp_path):
    make_split(tmp_path / "in", {"wrinkle_only": "4 0.2 0.2 0.05 0.01"})
    counts = filter_split(tmp_path / "in", tmp_path / "out", {4}, drop_left_empty=False)
    assert counts["kept"] == 1
    assert (tmp_path / "out" / "labels" / "wrinkle_only.txt").read_text() == ""


def test_missing_label_raises(tmp_path):
    make_split(tmp_path / "in", {"a": "1 0.5 0.5 0.1 0.1"})
    (tmp_path / "in" / "labels" / "a.txt").unlink()
    with pytest.raises(FileNotFoundError):
        filter_split(tmp_path / "in", tmp_path / "out", {4}, drop_left_empty=True)


def test_data_yaml_has_four_classes_and_clean_paths(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "in" / "data.yaml").write_text(yaml.safe_dump({"names": NAMES}))
    write_data_yaml(tmp_path / "in", tmp_path / "out", {4})

    config = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
    assert config["nc"] == 4
    assert config["names"] == NAMES[:4]
    assert config["train"] == "train/images"
    assert "path" not in config


def test_dropping_a_middle_class_is_refused(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "data.yaml").write_text(yaml.safe_dump({"names": NAMES}))
    with pytest.raises(ValueError):
        write_data_yaml(tmp_path / "in", tmp_path, {1})


# Polygon handling. The real dataset contains lines like this one
# (a closed 6-point outline), mixed in with ordinary box lines.
REAL_POLYGON = (
    "4 0.3117284722222222 0.8806583333333333 0.38734583333333333 0.7870370370370371 "
    "0.37345694444444444 0.7664611111111111 0.375 0.7623453703703703 "
    "0.2993826388888889 0.871399074074074 0.3117284722222222 0.8806583333333333"
)


def test_polygon_to_box_square():
    square = [0.2, 0.2, 0.6, 0.2, 0.6, 0.8, 0.2, 0.8]
    assert polygon_to_box(square) == pytest.approx([0.4, 0.5, 0.4, 0.6])


def test_polygon_points_outside_image_are_clipped():
    assert polygon_to_box([-0.1, 0.5, 0.5, 0.0, 0.5, 1.2]) == pytest.approx([0.25, 0.5, 0.5, 1.0])


def test_real_polygon_line_parses():
    class_id, box, was_polygon = parse_label_line(REAL_POLYGON)
    assert class_id == 4 and was_polygon
    x, y, w, h = box
    # Box must enclose every point of the outline.
    assert x - w / 2 == pytest.approx(0.2993826388888889)
    assert x + w / 2 == pytest.approx(0.38734583333333333)


@pytest.mark.parametrize("bad", ["1 0.5 0.5", "1 0.1 0.2 0.3 0.4 0.5", "1 0.1 0.2 0.3 0.4 0.5 0.6 0.7"])
def test_invalid_lines_raise(bad):
    with pytest.raises(ValueError):
        parse_label_line(bad)


def test_polygon_of_kept_class_becomes_box():
    polygon = "1 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8"
    kept, converted = filter_label_lines([polygon, REAL_POLYGON], {4})
    assert converted == {1: 1}
    assert len(kept) == 1 and len(kept[0].split()) == 5
    assert kept[0] == "1 0.400000 0.500000 0.400000 0.600000"
