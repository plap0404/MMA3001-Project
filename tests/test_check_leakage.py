"""Tests for scripts/check_leakage.py, using small synthetic images."""

import numpy as np
import pytest
from PIL import Image

from check_leakage import dhash, hamming_matrix, nearest_frame, parse_frame, sampling_step


@pytest.mark.parametrize("name, expected", [
    ("frame_with_red_10200_jpg.rf.3f9a1c.jpg", ("frame_with_red", 10200)),
    ("frame_with_red_10200.jpg", ("frame_with_red", 10200)),
    ("cam2_frame_100_jpg.rf.ab12.jpg", ("cam2_frame", 100)),  # uses the LAST number
    ("tray.jpg", None),
])
def test_parse_frame(name, expected):
    assert parse_frame(name) == expected


def make_image(path, shift=0, seed=1):
    """Save a random grey texture, optionally shifted sideways by some pixels."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(120, 200), dtype=np.uint8)
    Image.fromarray(np.roll(base, shift, axis=1)).save(path)


def test_identical_images_have_distance_zero(tmp_path):
    make_image(tmp_path / "a.png")
    make_image(tmp_path / "b.png")
    assert hamming_matrix(dhash(tmp_path / "a.png")[None], dhash(tmp_path / "b.png")[None])[0, 0] == 0


def test_similar_images_closer_than_different_ones(tmp_path):
    make_image(tmp_path / "a.png", seed=1)
    make_image(tmp_path / "a_moved.png", shift=2, seed=1)
    make_image(tmp_path / "other.png", seed=2)
    h = np.stack([dhash(tmp_path / n) for n in ("a.png", "a_moved.png", "other.png")])
    d = hamming_matrix(h, h)
    assert d[0, 1] < d[0, 2]


def test_hash_length():
    # 16 x 16 = 256 bits by default; independent of image size.
    assert dhash.__defaults__[0] ** 2 == 256


def test_hamming_matrix_shape_and_values():
    a = np.array([[True, False, True], [False, False, False]])
    b = np.array([[True, False, True], [True, True, True], [False, False, True]])
    expected = np.array([[0, 1, 1], [2, 3, 1]])
    assert np.array_equal(hamming_matrix(a, b), expected)


def test_nearest_frame_same_video_only():
    train = [("vid_a", 100), ("vid_b", 203), ("vid_a", 180), None]
    assert nearest_frame(("vid_a", 200), train) == (2, 20)  # vid_b 203 is ignored


def test_nearest_frame_no_match():
    assert nearest_frame(("vid_c", 5), [("vid_a", 5)]) == (None, None)
    assert nearest_frame(None, [("vid_a", 5)]) == (None, None)


def test_sampling_step_every_10th_frame():
    frames = [("v", n) for n in range(0, 200, 10)] + [None]
    assert sampling_step(frames) == 10


def test_sampling_step_ignores_jumps_between_videos():
    frames = [("a", 0), ("a", 5), ("a", 10), ("b", 9000), ("b", 9005)]
    assert sampling_step(frames) == 5
