"""Gating test: TemplateClassifier must recognise all 28 piece types correctly."""
from __future__ import annotations

import pytest

from shogi_vision.piece_classifier import TemplateClassifier
from shogi_vision.pieces import PIECE_TYPES, Player, piece_for_player
from shogi_vision.synthetic import render_single_piece


@pytest.fixture(scope="module")
def clf():
    return TemplateClassifier(cell_size=100)


@pytest.mark.parametrize("piece_type", PIECE_TYPES)
def test_black_piece_recognised(clf, piece_type):
    sfen = piece_for_player(piece_type, Player.BLACK)
    cell = render_single_piece(sfen, cell_size=100)
    result = clf.classify(cell)
    assert result == sfen, (
        f"Expected {sfen!r} for Black {piece_type}, got {result!r}"
    )


@pytest.mark.parametrize("piece_type", PIECE_TYPES)
def test_white_piece_recognised(clf, piece_type):
    sfen = piece_for_player(piece_type, Player.WHITE)
    cell = render_single_piece(sfen, cell_size=100)
    result = clf.classify(cell)
    assert result == sfen, (
        f"Expected {sfen!r} for White {piece_type}, got {result!r}"
    )


def test_empty_cell_returns_none(clf):
    import numpy as np
    empty = np.ones((100, 100, 3), dtype=np.uint8) * 255
    assert clf.classify(empty) is None
