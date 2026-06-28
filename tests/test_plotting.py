"""Tests for svm_gmu.plotting helpers that do not require rendering."""

import matplotlib

from svm_gmu.plotting import use_latex_serif


def test_use_latex_serif_sets_rcparams():
    use_latex_serif()
    rc = matplotlib.rcParams
    assert rc["text.usetex"] is True
    assert rc["font.family"] == ["serif"]
    assert rc["pgf.texsystem"] == "pdflatex"
    assert rc["pgf.rcfonts"] is False
    assert "lmodern" in rc["pgf.preamble"]
    assert "lmodern" in rc["text.latex.preamble"]
