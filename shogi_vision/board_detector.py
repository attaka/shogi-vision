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
    """Return 4 ordered corners (TL, TR, BR, BL) of the board as float32 array.

    Strategy (in order):
      1. Hough-line grid detector — finds 10 evenly-spaced horizontal +
         10 evenly-spaced vertical lines forming the 9x9 grid.
         Robust against side panels (持ち駒 area) and labels.
      2. Largest-quadrilateral contour — fallback for boards that have a
         clear outer frame but a noisy interior.
      3. Full image — last-resort fallback.
    """
    grid = _find_grid_corners_hough(image)
    if grid is not None:
        return _order_corners(grid)

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


# ── Projection-profile based 9x9 grid detector ────────────────────────────────

def _find_grid_corners_hough(image: np.ndarray) -> Optional[np.ndarray]:
    """Detect a 9x9 shogi grid using axis-aligned edge projection profiles.

    Sums edge pixels along each row (resp. column) to score how much of
    that row (resp. column) lies on a horizontal (resp. vertical) line.
    Peaks correspond to grid lines. Then searches for 10 evenly-spaced
    peaks on each axis. This works even when pieces obscure half of each
    grid line because the remaining cells still contribute to the peak.

    Returns 4 corners of the grid region (unordered), or None if a
    regular 9x9 lattice could not be confidently located.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarise (foreground = dark pixels: grid lines + glyphs)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 5,
    )

    # Morphological opening with long horizontal / vertical kernels keeps
    # only structures longer than the kernel — i.e., grid lines —
    # while suppressing kanji glyphs which span only ~one cell.
    h_kernel_len = max(int(w * 0.25), 20)
    v_kernel_len = max(int(h * 0.25), 20)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    row_proj = horiz.sum(axis=1).astype(np.float32)
    col_proj = vert.sum(axis=0).astype(np.float32)

    h_thresh = max(float(row_proj.max()) * 0.3, 1.0)
    v_thresh = max(float(col_proj.max()) * 0.3, 1.0)
    h_peaks = _find_peaks_1d(row_proj, threshold=h_thresh, min_dist=h * 0.05)
    v_peaks = _find_peaks_1d(col_proj, threshold=v_thresh, min_dist=w * 0.05)

    grid_h = _find_evenly_spaced(h_peaks, count=10, min_spacing=h * 0.05)
    grid_v = _find_evenly_spaced(v_peaks, count=10, min_spacing=w * 0.05)
    if grid_h is None or grid_v is None:
        return None

    span_h = grid_h[-1] - grid_h[0]
    span_v = grid_v[-1] - grid_v[0]
    if abs(span_h - span_v) > 0.15 * max(span_h, span_v):
        return None  # not square enough

    return np.array(
        [
            [grid_v[0], grid_h[0]],
            [grid_v[-1], grid_h[0]],
            [grid_v[-1], grid_h[-1]],
            [grid_v[0], grid_h[-1]],
        ],
        dtype=np.float32,
    )


def _find_peaks_1d(
    signal: np.ndarray, threshold: float, min_dist: float
) -> list[float]:
    """Find peak positions in `signal` above `threshold`.

    Adjacent indices above threshold are merged into a single peak at the
    local maximum (this collapses 2-pixel-thick line responses into a
    single position). Stronger peaks suppress weaker ones within
    `min_dist`.
    """
    n = len(signal)
    raw_peaks: list[int] = []
    i = 0
    while i < n:
        if signal[i] >= threshold:
            j = i
            while j < n and signal[j] >= threshold:
                j += 1
            run = signal[i:j]
            raw_peaks.append(i + int(np.argmax(run)))
            i = j
        else:
            i += 1

    # Non-max suppression: keep stronger peaks, drop weaker ones within min_dist
    raw_peaks.sort(key=lambda p: -float(signal[p]))
    kept: list[int] = []
    for p in raw_peaks:
        if all(abs(p - k) >= min_dist for k in kept):
            kept.append(p)
    return sorted(float(p) for p in kept)


def _find_evenly_spaced(
    positions: list[float], count: int, min_spacing: float
) -> Optional[list[float]]:
    """Search for `count` evenly-spaced positions among `positions`.

    For every (first, last) candidate pair (including pairs of count-1
    points from which the count-th is extrapolated), check how many of
    the expected positions have a near match.  Returns the best matching
    set if at least count-2 positions are inliers.

    The count-1 extrapolation handles the common case where the last grid
    boundary lies at the image edge and is not detected as a peak.
    """
    n = len(positions)
    if n < count - 1:
        return None

    best_score = -1
    best: Optional[list[float]] = None

    # Try both count points and count-1 points (extrapolate the last one)
    for search_count in (count, count - 1):
        if n < search_count:
            continue
        for i in range(n):
            for j in range(i + search_count - 1, n):
                spacing = (positions[j] - positions[i]) / (search_count - 1)
                if spacing < min_spacing:
                    continue
                tol = spacing * 0.20
                picked: list[float] = []
                score = 0
                for k in range(count):
                    expected = positions[i] + k * spacing
                    nearest = min(positions, key=lambda p: abs(p - expected))
                    if abs(nearest - expected) <= tol:
                        picked.append(nearest)
                        score += 1
                    else:
                        picked.append(expected)
                if score > best_score:
                    best_score = score
                    best = picked

    if best is None or best_score < count - 2:
        return None
    return best


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
