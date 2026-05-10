"""Tests for board detection."""
from __future__ import annotations

import numpy as np
import pytest

from shogi_vision.board_detector import detect_board, BOARD_OUTPUT_SIZE


def test_output_size(initial_board_image):
    result = detect_board(initial_board_image)
    assert result.shape == (BOARD_OUTPUT_SIZE, BOARD_OUTPUT_SIZE, 3)


def test_output_is_bgr(initial_board_image):
    result = detect_board(initial_board_image)
    assert result.dtype == np.uint8
    assert result.ndim == 3


def test_full_image_fallback():
    """detect_board should handle images with no clear quadrilateral."""
    blank = np.ones((500, 500, 3), dtype=np.uint8) * 200
    result = detect_board(blank)
    assert result.shape == (BOARD_OUTPUT_SIZE, BOARD_OUTPUT_SIZE, 3)


def test_board_with_border():
    """Synthetic board wrapped in a white border should still return correct size."""
    import cv2
    from shogi_vision.pieces import initial_board
    from shogi_vision.synthetic import render_board

    board, *_ = initial_board()
    board_img = render_board(board, size=700)
    # Add a white border (simulating a photo margin)
    bordered = cv2.copyMakeBorder(board_img, 30, 30, 30, 30,
                                   cv2.BORDER_CONSTANT, value=[255, 255, 255])
    result = detect_board(bordered)
    assert result.shape == (BOARD_OUTPUT_SIZE, BOARD_OUTPUT_SIZE, 3)
