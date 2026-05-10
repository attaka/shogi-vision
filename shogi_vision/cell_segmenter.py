"""Segment a perspective-corrected board image into 81 individual cells."""
from __future__ import annotations

import numpy as np

BOARD_RANKS = 9
BOARD_FILES = 9


def segment_cells(board_image: np.ndarray) -> list[list[np.ndarray]]:
    """Split a square board image into a 9×9 grid of cell images.

    Parameters
    ----------
    board_image:
        A square BGR numpy array of the perspective-corrected board
        (typically ``900 × 900``).

    Returns
    -------
    cells : list[list[np.ndarray]]
        ``cells[rank_idx][file_idx]`` — the cell image at that position.
        rank_idx 0 = rank 1 (white's back rank, top of image).
        file_idx 0 = file 9 (left side of diagram).
    """
    h, w = board_image.shape[:2]
    cell_h = h // BOARD_RANKS
    cell_w = w // BOARD_FILES

    cells: list[list[np.ndarray]] = []
    for rank in range(BOARD_RANKS):
        row: list[np.ndarray] = []
        y0 = rank * cell_h
        y1 = (rank + 1) * cell_h if rank < BOARD_RANKS - 1 else h
        for file in range(BOARD_FILES):
            x0 = file * cell_w
            x1 = (file + 1) * cell_w if file < BOARD_FILES - 1 else w
            row.append(board_image[y0:y1, x0:x1].copy())
        cells.append(row)
    return cells


def cell_size(board_image: np.ndarray) -> tuple[int, int]:
    """Return (cell_height, cell_width) for a given board image."""
    h, w = board_image.shape[:2]
    return h // BOARD_RANKS, w // BOARD_FILES
