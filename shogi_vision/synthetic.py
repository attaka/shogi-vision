"""Render a synthetic shogi board image using PIL (for testing)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pieces import (
    PIECE_TO_KANJI,
    Board,
    BOARD_RANKS,
    BOARD_FILES,
    player_of,
    piece_type_of,
    Player,
)

_IPA_FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"

# Fallback font paths in order of preference.
# Includes Linux (IPAGothic / WQY) and macOS (Hiragino / PingFang) defaults.
_FONT_CANDIDATES = [
    _IPA_FONT_PATH,
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    # macOS — Hiragino is the standard Japanese font shipped with the OS
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError(
        "No CJK font found. Install IPAGothic (Linux) or rely on Hiragino "
        "(macOS). Tried: " + ", ".join(_FONT_CANDIDATES)
    )


def render_board(board: Board, size: int = 900) -> np.ndarray:
    """Render *board* as a BGR numpy array of shape ``(size, size, 3)``.

    Black's pieces (先手) are drawn upright; White's pieces (後手) are
    drawn rotated 180° — matching the standard Japanese shogi diagram format.

    The board fills the entire image with no rank/file labels, making it
    straightforward for the board detector to find the grid.

    Parameters
    ----------
    board:
        9×9 array from ``pieces.py`` — ``board[rank_idx][file_idx]``
        where rank_idx 0 = rank 1 (top/white's side), file_idx 0 = file 9.
    size:
        Output image side length in pixels (square).

    Returns
    -------
    np.ndarray
        BGR numpy array suitable for OpenCV.
    """
    cell = size // BOARD_RANKS  # cell size in pixels
    font_size = int(cell * 0.55)
    font = _get_font(font_size)

    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw grid lines first
    for i in range(BOARD_RANKS + 1):
        y = i * cell
        draw.line([(0, y), (size, y)], fill=(0, 0, 0), width=2)
    for j in range(BOARD_FILES + 1):
        x = j * cell
        draw.line([(x, 0), (x, size)], fill=(0, 0, 0), width=2)

    # Draw pieces — paste only dark (kanji) pixels using a luminance mask
    # so the grid lines underneath are preserved
    _black = Image.new("RGB", (cell, cell), color=(0, 0, 0))
    for rank_idx in range(BOARD_RANKS):
        for file_idx in range(BOARD_FILES):
            piece = board[rank_idx][file_idx]
            if piece is None or piece == "":
                continue

            ptype = piece_type_of(piece)
            kanji = PIECE_TO_KANJI.get(ptype, "？")
            owner = player_of(piece)

            # Render kanji on a white cell — only dark pixels will be pasted
            cell_img = Image.new("RGB", (cell, cell), color=(255, 255, 255))
            cdraw = ImageDraw.Draw(cell_img)

            bbox = cdraw.textbbox((0, 0), kanji, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = (cell - tw) // 2 - bbox[0]
            ty = (cell - th) // 2 - bbox[1]
            cdraw.text((tx, ty), kanji, fill=(0, 0, 0), font=font)

            if owner == Player.WHITE:
                cell_img = cell_img.rotate(180)

            # mask: bright where the kanji is dark → paste only those pixels
            mask = Image.fromarray(
                (255 - np.array(cell_img.convert("L"))).astype(np.uint8)
            )
            img.paste(_black, (file_idx * cell, rank_idx * cell), mask=mask)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_single_piece(
    piece: str, cell_size: int = 100, white_background: bool = True
) -> np.ndarray:
    """Render a single piece into a ``cell_size × cell_size`` BGR image.

    Useful for building template libraries.
    """
    font_size = int(cell_size * 0.55)
    font = _get_font(font_size)

    bg = (255, 255, 255) if white_background else (240, 220, 180)
    img = Image.new("RGB", (cell_size, cell_size), color=bg)
    draw = ImageDraw.Draw(img)

    ptype = piece_type_of(piece)
    if ptype is None:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    kanji = PIECE_TO_KANJI.get(ptype, "？")
    owner = player_of(piece)

    bbox = draw.textbbox((0, 0), kanji, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (cell_size - tw) // 2 - bbox[0]
    ty = (cell_size - th) // 2 - bbox[1]
    draw.text((tx, ty), kanji, fill=(0, 0, 0), font=font)

    if owner == Player.WHITE:
        img = img.rotate(180)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
