"""End-to-end pipeline test with synthetic board image."""
from __future__ import annotations

import pytest

from shogi_vision.pieces import INITIAL_SFEN, initial_board
from shogi_vision.pipeline import image_to_sfen, image_to_sfen_with_diagnostics
from shogi_vision.piece_classifier import TemplateClassifier
from shogi_vision.synthetic import render_board


@pytest.fixture(scope="module")
def initial_image():
    board, *_ = initial_board()
    return render_board(board, size=900)


@pytest.fixture(scope="module")
def clf():
    return TemplateClassifier(cell_size=100)


def test_initial_position_sfen(initial_image, clf):
    """Pipeline must reproduce INITIAL_SFEN from a synthetic rendering."""
    result = image_to_sfen(initial_image, classifier=clf)
    assert result == INITIAL_SFEN


def test_empty_board_pipeline():
    from shogi_vision.pieces import empty_board, Player
    board = empty_board()
    img = render_board(board, size=900)
    clf = TemplateClassifier(cell_size=100)
    result = image_to_sfen(img, classifier=clf, turn=Player.BLACK)
    assert result == "9/9/9/9/9/9/9/9/9 b - 1"


def test_output_is_string(initial_image, clf):
    result = image_to_sfen(initial_image, classifier=clf)
    assert isinstance(result, str)
    parts = result.split()
    assert len(parts) == 4
    assert parts[1] in ("b", "w")


def test_diagnostics_output(initial_image, clf):
    result = image_to_sfen_with_diagnostics(initial_image, classifier=clf)
    assert result["sfen"] == INITIAL_SFEN
    assert result["board_size"] == [900, 900]
    for key in (
        "load_image_s",
        "detect_board_s",
        "segment_cells_s",
        "classify_cells_s",
        "build_sfen_s",
        "total_s",
    ):
        assert key in result["timings"]
        assert result["timings"][key] >= 0.0
