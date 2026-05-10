"""Visualisation helpers for debugging board detection and piece classification."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .piece_classifier import Classifier
from .cell_segmenter import segment_cells, BOARD_RANKS, BOARD_FILES


def draw_overlay(
    board_img: np.ndarray,
    cells: list[list[np.ndarray]],
    clf: Classifier,
) -> np.ndarray:
    """Return a copy of *board_img* annotated with the predicted piece labels.

    Each cell shows the classifier's prediction (SFEN symbol or '·' for empty).
    """
    out = board_img.copy()
    h, w = out.shape[:2]
    cell_h = h // BOARD_RANKS
    cell_w = w // BOARD_FILES

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cell_h / 80.0
    thickness = max(1, int(font_scale * 2))

    for rank_idx in range(BOARD_RANKS):
        for file_idx in range(BOARD_FILES):
            cell = cells[rank_idx][file_idx]
            label: Optional[str] = clf.classify(cell)
            text = label if label else "·"

            x0 = file_idx * cell_w
            y0 = rank_idx * cell_h

            # Draw grid cell border
            cv2.rectangle(out, (x0, y0), (x0 + cell_w, y0 + cell_h),
                          (0, 200, 0), 1)

            # Put label text
            tw, th = cv2.getTextSize(text, font, font_scale, thickness)[0]
            tx = x0 + (cell_w - tw) // 2
            ty = y0 + (cell_h + th) // 2
            color = (200, 0, 0) if label and label[0].isupper() else (0, 0, 200)
            cv2.putText(out, text, (tx, ty), font, font_scale, color, thickness,
                        cv2.LINE_AA)

    return out
