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
    sfen_to_board,
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

    # Draw grid lines
    for i in range(BOARD_RANKS + 1):
        y = i * cell
        draw.line([(0, y), (size, y)], fill=(0, 0, 0), width=2)
    for j in range(BOARD_FILES + 1):
        x = j * cell
        draw.line([(x, 0), (x, size)], fill=(0, 0, 0), width=2)

    # Draw pieces
    for rank_idx in range(BOARD_RANKS):
        for file_idx in range(BOARD_FILES):
            piece = board[rank_idx][file_idx]
            if piece is None or piece == "":
                continue

            ptype = piece_type_of(piece)
            kanji = PIECE_TO_KANJI.get(ptype, "？")
            owner = player_of(piece)

            # Render kanji into a temporary cell-sized image
            cell_img = Image.new("RGB", (cell, cell), color=(255, 255, 255))
            cdraw = ImageDraw.Draw(cell_img)

            # Measure text to center it
            bbox = cdraw.textbbox((0, 0), kanji, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = (cell - tw) // 2 - bbox[0]
            ty = (cell - th) // 2 - bbox[1]
            cdraw.text((tx, ty), kanji, fill=(0, 0, 0), font=font)

            # White pieces are rotated 180°
            if owner == Player.WHITE:
                cell_img = cell_img.rotate(180)

            # Paste into main image
            x0 = file_idx * cell
            y0 = rank_idx * cell
            img.paste(cell_img, (x0, y0))

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


def render_sfen_pretty(
    sfen: str,
    size: int = 900,
    board_color: tuple[int, int, int] = (235, 197, 132),
) -> np.ndarray:
    """Render a cleaner board image from SFEN.

    This renderer is inspired by lightweight GUI board styles (e.g. ShogiHome):
    wood-like board background and pentagonal piece silhouettes. It is suitable
    for generating visually clean demo images before running image->SFEN tests.

    If ``cshogi`` is installed, the input SFEN is validated via ``cshogi.Board``.
    """
    try:
        import cshogi  # type: ignore
    except ImportError:
        cshogi = None

    if cshogi is not None:
        # Fail fast on invalid SFEN when cshogi is available.
        b = cshogi.Board()
        b.set_sfen(sfen)
        _ = b.sfen()

    board, _, _, _ = sfen_to_board(sfen)
    cell = size // BOARD_RANKS
    font_size = int(cell * 0.44)
    font = _get_font(font_size)

    img = Image.new("RGBA", (size, size), color=(*board_color, 255))
    # NOTE: RGBA is required for alpha_composite with RGBA piece layers.
    draw = ImageDraw.Draw(img)

    # grid
    for i in range(BOARD_RANKS + 1):
        y = i * cell
        draw.line([(0, y), (size, y)], fill=(45, 35, 20), width=2)
    for j in range(BOARD_FILES + 1):
        x = j * cell
        draw.line([(x, 0), (x, size)], fill=(45, 35, 20), width=2)

    margin = int(cell * 0.08)

    for rank_idx in range(BOARD_RANKS):
        for file_idx in range(BOARD_FILES):
            piece = board[rank_idx][file_idx]
            if not piece:
                continue

            owner = player_of(piece)
            ptype = piece_type_of(piece)
            kanji = PIECE_TO_KANJI.get(ptype, "？")

            x0 = file_idx * cell + margin
            y0 = rank_idx * cell + margin
            x1 = (file_idx + 1) * cell - margin
            y1 = (rank_idx + 1) * cell - margin
            cx = (x0 + x1) // 2

            # simple pentagonal shogi piece silhouette
            poly = [
                (cx, y0),
                (x1, y0 + int(cell * 0.22)),
                (x1 - int(cell * 0.08), y1),
                (x0 + int(cell * 0.08), y1),
                (x0, y0 + int(cell * 0.22)),
            ]

            piece_img = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
            pdraw = ImageDraw.Draw(piece_img)
            local_poly = [(px - file_idx * cell, py - rank_idx * cell) for px, py in poly]
            pdraw.polygon(local_poly, fill=(248, 232, 186), outline=(60, 45, 25), width=2)

            bbox = pdraw.textbbox((0, 0), kanji, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = (cell - tw) // 2 - bbox[0]
            ty = int(cell * 0.38) - th // 2 - bbox[1]
            pdraw.text((tx, ty), kanji, fill=(20, 20, 20), font=font)

            if owner == Player.WHITE:
                piece_img = piece_img.rotate(180)

            # Use mask paste for compatibility across PIL versions/modes.
            img.paste(piece_img, (file_idx * cell, rank_idx * cell), piece_img)

    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
