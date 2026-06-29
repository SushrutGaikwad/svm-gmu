# tests/test_realworld.py
import importlib.util
import sys
from pathlib import Path

import numpy as np

_EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXP_DIR))
_spec = importlib.util.spec_from_file_location("_realworld", _EXP_DIR / "_realworld.py")
RW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RW)


def test_augment_image_zero_is_identity():
    img = np.zeros((28, 28))
    img[8:14, 6:10] = 1.0  # asymmetric patch
    rng = np.random.default_rng(0)
    out = RW.augment_image(img, rng, rot_deg=0.0, max_shift=0.0)
    assert out.shape == (28, 28)
    assert np.allclose(out, img, atol=1e-6)


def test_augment_image_rotation_changes_asymmetric_patch():
    img = np.zeros((28, 28))
    img[8:14, 6:10] = 1.0
    rng = np.random.default_rng(0)
    out = RW.augment_image(img, rng, rot_deg=90.0, max_shift=0.0)
    assert not np.allclose(out, img)
