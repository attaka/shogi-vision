"""Tests that classifiers return correct uppercase/lowercase SFEN symbols."""
from __future__ import annotations

import pytest

from shogi_vision.piece_classifier import TemplateClassifier
from shogi_vision.pieces import Player, piece_for_player
from shogi_vision.synthetic import render_single_piece


@pytest.fixture(scope="module")
def clf():
    return TemplateClassifier(cell_size=100)


@pytest.mark.parametrize("piece_type", ["K", "R", "B", "G", "S", "N", "L", "P"])
def test_black_piece_is_uppercase(clf, piece_type):
    sfen_black = piece_for_player(piece_type, Player.BLACK)
    cell = render_single_piece(sfen_black, cell_size=100)
    result = clf.classify(cell)
    assert result is not None
    first = result.lstrip("+")
    assert first.isupper(), f"Black {piece_type} should return uppercase, got {result!r}"


@pytest.mark.parametrize("piece_type", ["K", "R", "B", "G", "S", "N", "L", "P"])
def test_white_piece_is_lowercase(clf, piece_type):
    sfen_white = piece_for_player(piece_type, Player.WHITE)
    cell = render_single_piece(sfen_white, cell_size=100)
    result = clf.classify(cell)
    assert result is not None
    first = result.lstrip("+")
    assert first.islower(), f"White {piece_type} should return lowercase, got {result!r}"
