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


def test_augmentation_cloud_shape_and_zero_range():
    img = np.zeros(784)
    img[300:320] = 1.0
    rng = np.random.default_rng(0)
    cloud = RW.augmentation_cloud(img, n_aug=16, rng=rng, rot_range=0.0, max_shift=0.0)
    assert cloud.shape == (16, 784)
    assert np.allclose(cloud, img[None, :], atol=1e-6)


def test_moment_gaussian_diag():
    cloud = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    g = RW.moment_gaussian(cloud, "diag")
    assert g["weights"].shape == (1,) and np.isclose(g["weights"][0], 1.0)
    assert np.allclose(g["means"][0], [1.0, 1.0])
    assert g["covariances"].shape == (1, 2)
    assert np.allclose(g["covariances"][0], [1.0, 1.0])


def test_moment_gaussian_full_shape():
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(50, 3))
    g = RW.moment_gaussian(cloud, "full")
    assert g["covariances"].shape == (1, 3, 3)


def test_iso_gaussian_is_isotropic():
    cloud = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
    g = RW.iso_gaussian(cloud)
    assert g["covariances"].shape == (1, 2)
    assert np.allclose(g["covariances"][0], [2.0, 2.0])  # mean of per-dim var (4, 0)
