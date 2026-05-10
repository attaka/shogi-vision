"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from shogi_vision.pieces import initial_board, sfen_to_board, INITIAL_SFEN
from shogi_vision.synthetic import render_board


@pytest.fixture(scope="session")
def initial_position_board():
    """Return (board_array, turn, hands, move_count) for the initial shogi position."""
    return initial_board()


@pytest.fixture(scope="session")
def initial_board_image(initial_position_board):
    """Return a synthetic BGR image of the initial shogi position."""
    board, *_ = initial_position_board
    return render_board(board, size=900)


@pytest.fixture(scope="session")
def template_classifier():
    """Return a TemplateClassifier instance (no Tesseract required)."""
    from shogi_vision.piece_classifier import TemplateClassifier
    return TemplateClassifier(cell_size=100)
