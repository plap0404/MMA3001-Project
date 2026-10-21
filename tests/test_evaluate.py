"""Tests for scripts/evaluate.py, using stand-in metrics instead of a model."""

from evaluate import build_rows, format_table

NAMES = {0: "loose-meat", 1: "packaging-error", 2: "twisted-meat", 3: "unsealed"}
# (precision, recall, mAP50, mAP50-95) for each entry of ap_class_index
FAKE = {0: (0.8, 0.6, 0.7, 0.4), 1: (0.5, 0.4, 0.45, 0.2)}


def fake_class_result(position):
    return FAKE[position]


def test_rows_cover_every_class_even_when_absent():
    rows = build_rows(NAMES, [1, 3], fake_class_result, [3, 29, 7, 52])
    assert {r["class"] for r in rows} == set(NAMES.values())


def test_present_classes_get_their_metrics():
    rows = build_rows(NAMES, [1, 3], fake_class_result, [3, 29, 7, 52])
    by_name = {r["class"]: r for r in rows}
    assert by_name["packaging-error"]["precision"] == 0.8   # position 0
    assert by_name["unsealed"]["recall"] == 0.4             # position 1
    assert by_name["unsealed"]["boxes"] == 52


def test_absent_classes_have_no_metrics_but_keep_box_counts():
    rows = build_rows(NAMES, [1, 3], fake_class_result, [3, 29, 7, 52])
    by_name = {r["class"]: r for r in rows}
    assert by_name["loose-meat"]["mAP50"] is None
    assert by_name["loose-meat"]["boxes"] == 3


def test_box_counts_are_indexed_by_class_id_not_position():
    # Regression guard: using the position instead of the class id would give 3.
    rows = build_rows(NAMES, [1], fake_class_result, [3, 29, 7, 52])
    assert {r["class"]: r["boxes"] for r in rows}["packaging-error"] == 29


def test_table_has_a_row_per_class_plus_overall():
    rows = build_rows(NAMES, [1, 3], fake_class_result, [3, 29, 7, 52])
    overall = {"precision": 0.65, "recall": 0.5, "mAP50": 0.575, "mAP50-95": 0.3}
    lines = format_table(rows, overall).splitlines()
    assert len(lines) == 2 + len(NAMES) + 1
    assert lines[-1].startswith("| **all** | **91** |")


def test_table_shows_missing_metrics_as_dash():
    rows = build_rows(NAMES, [1], fake_class_result, [3, 29, 7, 52])
    overall = {"precision": 0.8, "recall": 0.6, "mAP50": 0.7, "mAP50-95": 0.4}
    table = format_table(rows, overall)
    assert "| loose-meat | 3 | - | - | - | - |" in table
