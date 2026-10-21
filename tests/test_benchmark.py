"""Tests for scripts/benchmark.py, using fixed timing lists."""

import pytest

from benchmark import format_report, summarise_times, trays_per_minute

TIMES = [10.0, 12.0, 11.0, 50.0, 10.5]  # 50.0 is an outlier, as happens in practice


def test_median_resists_the_outlier():
    summary = summarise_times(TIMES)
    assert summary["median_ms"] == 11.0
    assert summary["mean_ms"] > summary["median_ms"]  # the mean is dragged up


def test_min_max_and_count():
    summary = summarise_times(TIMES)
    assert (summary["min_ms"], summary["max_ms"], summary["n"]) == (10.0, 50.0, 5)


def test_p95_is_at_the_top_of_the_range():
    assert summarise_times(TIMES)["p95_ms"] == 50.0


def test_p95_never_runs_off_the_end():
    # Nearest-rank indexing must stay inside the list for any length.
    for length in range(1, 40):
        summarise_times([float(i + 1) for i in range(length)])


def test_throughput_is_derived_from_the_median():
    assert summarise_times(TIMES)["images_per_second"] == pytest.approx(1000 / 11.0)


def test_trays_per_minute():
    assert trays_per_minute(20.0) == pytest.approx(3000.0)   # 50/s
    assert trays_per_minute(300.0) == pytest.approx(200.0)


def test_empty_timings_raise():
    with pytest.raises(ValueError):
        summarise_times([])


def test_non_positive_timing_raises():
    # A zero or negative time means the timer failed, not that the model is instant.
    with pytest.raises(ValueError):
        summarise_times([10.0, 0.0])


def test_report_names_every_stage_and_the_throughput():
    stages = {s: summarise_times(TIMES) for s in
              ("preprocess", "inference", "postprocess", "total")}
    report = format_report("run_x", "cpu", 640, stages, {"parameters": 3_200_000, "gflops": 8.7})
    for stage in ("preprocess", "inference", "postprocess", "total"):
        assert stage in report
    assert "3,200,000 parameters" in report
    assert "8.7 GFLOPs" in report
    assert "trays/minute" in report
