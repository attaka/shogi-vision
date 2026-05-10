"""End-to-end pipeline: image file → SFEN string."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .board_detector import detect_board, load_image
from .cell_segmenter import segment_cells
from .piece_classifier import Classifier, auto_classifier
from .pieces import Player, board_to_sfen, empty_board


def image_to_sfen(
    image: Union[str, Path, np.ndarray],
    *,
    classifier: Optional[Classifier] = None,
    turn: Player = Player.BLACK,
    move_count: int = 1,
) -> str:
    """Convert a shogi board image to a SFEN string.

    Parameters
    ----------
    image:
        File path (str or Path) or a BGR numpy array.
    classifier:
        A :class:`~shogi_vision.piece_classifier.Classifier` instance.
        Defaults to ``auto_classifier()``.
    turn:
        Whose turn it is (used in the SFEN string).  Defaults to Black.
    move_count:
        Move number written into the SFEN string.

    Returns
    -------
    str
        A four-part SFEN string, e.g.
        ``"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"``
    """
    if classifier is None:
        classifier = auto_classifier()

    # Load image
    if isinstance(image, (str, Path)):
        bgr = load_image(image)
    else:
        bgr = np.asarray(image)

    # Detect and perspective-correct the board
    board_img = detect_board(bgr)

    # Segment into 81 cells
    cells = segment_cells(board_img)
    cell_h = board_img.shape[0] // 9

    # Classify each cell
    board = empty_board()
    for rank_idx, row in enumerate(cells):
        for file_idx, cell in enumerate(row):
            if hasattr(classifier, "classify"):
                # Pass rank_idx hint to TesseractClassifier when available
                import inspect
                sig = inspect.signature(classifier.classify)
                if "rank_idx" in sig.parameters:
                    piece = classifier.classify(cell, rank_idx=rank_idx)
                else:
                    piece = classifier.classify(cell)
            else:
                piece = classifier.classify(cell)
            board[rank_idx][file_idx] = piece

    return board_to_sfen(board, turn=turn, move_count=move_count)
