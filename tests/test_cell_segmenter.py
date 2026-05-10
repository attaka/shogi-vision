"""Tests for cell segmentation."""
from __future__ import annotations

import numpy as np
import pytest

from shogi_vision.cell_segmenter import segment_cells, cell_size, BOARD_RANKS, BOARD_FILES


def _make_board_image(size: int = 900) -> np.ndarray:
    return np.ones((size, size, 3), dtype=np.uint8) * 255


def test_cell_count():
    img = _make_board_image()
    cells = segment_cells(img)
    assert len(cells) == BOARD_RANKS
    assert all(len(row) == BOARD_FILES for row in cells)


def test_cell_sizes():
    size = 900
    img = _make_board_image(size)
    cells = segment_cells(img)
    expected_h = size // BOARD_RANKS
    expected_w = size // BOARD_FILES
    for row in cells:
        for cell in row:
            assert cell.shape[0] == expected_h
            assert cell.shape[1] == expected_w


def test_cell_content():
    """Verify that segmentation preserves pixel values."""
    img = np.zeros((900, 900, 3), dtype=np.uint8)
    # Paint top-left cell red
    img[0:100, 0:100] = [0, 0, 255]  # BGR red

    cells = segment_cells(img)
    tl = cells[0][0]
    assert tl.shape == (100, 100, 3)
    # Mean blue channel of top-left cell should be ~255
    assert tl[:, :, 2].mean() > 200


def test_cell_size_helper():
    img = _make_board_image(900)
    h, w = cell_size(img)
    assert h == 100
    assert w == 100
