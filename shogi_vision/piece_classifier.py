"""Piece classification from cell images.

Two concrete implementations are provided:

* ``TesseractClassifier`` — uses Tesseract OCR (requires the ``tesseract``
  binary and the ``jpn`` traineddata to be installed).
* ``TemplateClassifier`` — uses OpenCV template matching against kanji images
  generated from IPAGothic at startup (no external binary needed).

Use ``auto_classifier()`` to get the best available classifier automatically.
"""
from __future__ import annotations

import shutil
from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np

from .pieces import KANJI_TO_PIECE, PIECE_TYPES, Player, piece_for_player

# ── tuneable constants ────────────────────────────────────────────────────────
EMPTY_STD_THRESHOLD = 15   # inner-ROI std-dev below this → empty cell
INNER_ROI_FRACTION = 0.70  # crop factor when measuring emptiness
OCR_SCALE = 2.5            # upsample factor before feeding Tesseract

KANJI_WHITELIST = "王玉飛角金銀桂香歩龍馬全圭杏と"


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class Classifier(Protocol):
    """Classify a single cell image into a SFEN piece symbol or ``None``."""

    def classify(self, cell: np.ndarray) -> Optional[str]:
        """Return the SFEN piece symbol (e.g. ``'K'``, ``'p'``, ``'+R'``),
        or ``None`` for an empty cell."""
        ...


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_empty(cell: np.ndarray) -> bool:
    """Return True if *cell* contains no piece (judged by pixel std-dev)."""
    h, w = cell.shape[:2]
    margin_h = int(h * (1 - INNER_ROI_FRACTION) / 2)
    margin_w = int(w * (1 - INNER_ROI_FRACTION) / 2)
    roi = cell[margin_h:h - margin_h, margin_w:w - margin_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) < EMPTY_STD_THRESHOLD


def _rot180(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_180)


def _preprocess_for_ocr(cell: np.ndarray) -> np.ndarray:
    """Upscale and binarise a cell image for reliable single-glyph OCR."""
    h, w = cell.shape[:2]
    big = cv2.resize(cell, (int(w * OCR_SCALE), int(h * OCR_SCALE)),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ── TesseractClassifier ───────────────────────────────────────────────────────

class TesseractClassifier:
    """Classify pieces using Tesseract OCR in Japanese single-glyph mode.

    For each non-empty cell, OCR is attempted at both 0° and 180°.
    - 0° valid → Black piece (先手)
    - 180° valid → White piece (後手)
    - Both valid → higher confidence wins; ties broken by rank position if
      ``rank_hint`` is passed, otherwise Black is assumed.
    - Neither valid → None (treated as empty or unknown)
    """

    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore
            self._tess = pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract is not installed. "
                "Run: pip install pytesseract"
            ) from exc
        if not shutil.which("tesseract"):
            raise RuntimeError(
                "tesseract binary not found. "
                "Run: apt-get install tesseract-ocr tesseract-ocr-jpn"
            )

    def classify(self, cell: np.ndarray, rank_idx: int = 4) -> Optional[str]:
        if _is_empty(cell):
            return None
        from PIL import Image  # lazy import to keep startup fast
        img = _preprocess_for_ocr(cell)
        img_pil = Image.fromarray(img)
        img180 = Image.fromarray(_preprocess_for_ocr(_rot180(cell)))

        kanji0, conf0 = self._ocr(img_pil)
        kanji180, conf180 = self._ocr(img180)

        valid0 = kanji0 in KANJI_TO_PIECE
        valid180 = kanji180 in KANJI_TO_PIECE

        if valid0 and not valid180:
            return piece_for_player(KANJI_TO_PIECE[kanji0], Player.BLACK)
        if valid180 and not valid0:
            return piece_for_player(KANJI_TO_PIECE[kanji180], Player.WHITE)
        if valid0 and valid180:
            # Both orientations produced a valid kanji — use confidence
            if conf0 > conf180:
                return piece_for_player(KANJI_TO_PIECE[kanji0], Player.BLACK)
            if conf180 > conf0:
                return piece_for_player(KANJI_TO_PIECE[kanji180], Player.WHITE)
            # Same confidence: White pieces are more common in ranks 0-3
            player = Player.WHITE if rank_idx < 4 else Player.BLACK
            kanji = kanji180 if player == Player.WHITE else kanji0
            return piece_for_player(KANJI_TO_PIECE[kanji], player)
        return None  # no valid kanji detected

    def _ocr(self, img_pil) -> tuple[str, float]:
        """Run Tesseract on a PIL image, return (kanji_str, confidence)."""
        config = (
            f"--psm 10 -l jpn "
            f"-c tessedit_char_whitelist={KANJI_WHITELIST}"
        )
        data = self._tess.image_to_data(
            img_pil, config=config,
            output_type=self._tess.Output.DICT,
        )
        texts = [t.strip() for t in data["text"] if t.strip()]
        confs = [
            float(c) for t, c in zip(data["text"], data["conf"])
            if t.strip() and c != "-1"
        ]
        if not texts:
            return "", 0.0
        # Take the token with highest confidence
        best_idx = int(np.argmax(confs)) if confs else 0
        return texts[best_idx] if best_idx < len(texts) else "", (confs[best_idx] if confs else 0.0)


# ── TemplateClassifier ────────────────────────────────────────────────────────

class TemplateClassifier:
    """Classify pieces via OpenCV template matching.

    Templates are rendered at startup from IPAGothic (no external binary
    needed).  This makes the classifier fully deterministic and suitable for
    CI environments where Tesseract is unavailable.
    """

    def __init__(self, cell_size: int = 100) -> None:
        self._cell_size = cell_size
        self._templates: list[tuple[str, np.ndarray]] = []
        self._build_templates()

    def _build_templates(self) -> None:
        from .synthetic import render_single_piece

        for pt in PIECE_TYPES:
            for player in (Player.BLACK, Player.WHITE):
                sfen = piece_for_player(pt, player)
                tmpl = render_single_piece(sfen, cell_size=self._cell_size)
                gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                self._templates.append((sfen, gray))

    def classify(self, cell: np.ndarray) -> Optional[str]:
        if _is_empty(cell):
            return None

        cell_resized = cv2.resize(cell, (self._cell_size, self._cell_size))
        gray = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2GRAY)

        best_score = -1.0
        best_sfen: Optional[str] = None

        for sfen, tmpl in self._templates:
            result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            if score > best_score:
                best_score = score
                best_sfen = sfen

        # Require a minimum match quality to avoid misclassifying empty cells
        if best_score < 0.4:
            return None
        return best_sfen


# ── factory ───────────────────────────────────────────────────────────────────

def auto_classifier(cell_size: int = 100) -> Classifier:
    """Return the best available classifier.

    Uses ``TesseractClassifier`` if the ``tesseract`` binary is on PATH,
    otherwise falls back to ``TemplateClassifier``.
    """
    if shutil.which("tesseract"):
        try:
            return TesseractClassifier()
        except RuntimeError:
            pass
    return TemplateClassifier(cell_size=cell_size)
