"""Detect and perspective-correct a shogi board in a photo."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# Output board size after perspective correction
BOARD_OUTPUT_SIZE = 900


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk and return a BGR numpy array."""
    img = Image.open(str(path)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def detect_board(image: np.ndarray) -> np.ndarray:
    """Detect the shogi board in *image* and return a square perspective-corrected view.

    The returned image is always ``BOARD_OUTPUT_SIZE × BOARD_OUTPUT_SIZE`` pixels (BGR).

    Strategy
    --------
    1. Edge detection via Canny.
    2. Find the largest closed quadrilateral contour (the board frame).
    3. Apply a perspective transform so the board fills the output square.

    If no quadrilateral is found the full image is returned (resized to square).
    """
    corners = _find_board_corners(image)
    return _apply_perspective(image, corners, BOARD_OUTPUT_SIZE)


# ── corner detection ──────────────────────────────────────────────────────────

def _find_board_corners(image: np.ndarray) -> np.ndarray:
    """Return 4 ordered corners (TL, TR, BR, BL) of the board as float32 array."""
    h, w = image.shape[:2]
    image_area = float(h * w)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Close small gaps so the board outline forms one contour
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # RETR_EXTERNAL → only outermost contours (skip interior grid cells)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_corners: Optional[np.ndarray] = None
    best_area = 0.0

    # Only accept quadrilaterals occupying a substantial fraction of the image —
    # this rejects e.g. a single grid cell being mistaken for the board.
    MIN_AREA_FRACTION = 0.30

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * MIN_AREA_FRACTION:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_area = area
            best_corners = approx.reshape(4, 2).astype(np.float32)

    if best_corners is None:
        # Fallback: assume the image already shows just the board (digital screenshot).
        best_corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )

    return _order_corners(best_corners)


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Reorder 4 points to (top-left, top-right, bottom-right, bottom-left)."""
    s = corners.sum(axis=1)
    d = np.diff(corners, axis=1).ravel()
    return np.array(
        [
            corners[np.argmin(s)],   # TL: smallest x+y
            corners[np.argmin(d)],   # TR: smallest x-y
            corners[np.argmax(s)],   # BR: largest x+y
            corners[np.argmax(d)],   # BL: largest x-y
        ],
        dtype=np.float32,
    )


def _apply_perspective(image: np.ndarray, corners: np.ndarray, size: int) -> np.ndarray:
    """Warp the four *corners* to a ``size × size`` square."""
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, (size, size))
