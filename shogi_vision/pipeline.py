"""End-to-end pipeline: image file → SFEN string."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional, Union

import numpy as np

from .board_detector import detect_board, load_image
from .cell_segmenter import segment_cells
from .piece_classifier import Classifier, auto_classifier
from .pieces import Player, board_to_sfen, empty_board


def image_to_sfen_with_diagnostics(
    image: Union[str, Path, np.ndarray],
    *,
    classifier: Optional[Classifier] = None,
    turn: Player = Player.BLACK,
    move_count: int = 1,
) -> dict:
    """Convert a shogi board image to SFEN with stage diagnostics.

    Returns a dictionary containing the final SFEN and per-stage timings in
    seconds. This is intended as phase-1 instrumentation for error analysis
    and regression tracking.
    """
    if classifier is None:
        classifier = auto_classifier()

    timings: dict[str, float] = {}

    t0 = perf_counter()
    if isinstance(image, (str, Path)):
        bgr = load_image(image)
    else:
        bgr = np.asarray(image)
    timings["load_image_s"] = perf_counter() - t0

    t1 = perf_counter()
    board_img = detect_board(bgr)
    timings["detect_board_s"] = perf_counter() - t1

    t2 = perf_counter()
    cells = segment_cells(board_img)
    timings["segment_cells_s"] = perf_counter() - t2

    t3 = perf_counter()
    board = empty_board()
    for rank_idx, row in enumerate(cells):
        for file_idx, cell in enumerate(row):
            if hasattr(classifier, "classify"):
                import inspect
                sig = inspect.signature(classifier.classify)
                if "rank_idx" in sig.parameters:
                    piece = classifier.classify(cell, rank_idx=rank_idx)
                else:
                    piece = classifier.classify(cell)
            else:
                piece = classifier.classify(cell)
            board[rank_idx][file_idx] = piece
    timings["classify_cells_s"] = perf_counter() - t3

    t4 = perf_counter()
    sfen = board_to_sfen(board, turn=turn, move_count=move_count)
    timings["build_sfen_s"] = perf_counter() - t4
    timings["total_s"] = sum(timings.values())

    return {
        "sfen": sfen,
        "board_size": [int(board_img.shape[1]), int(board_img.shape[0])],
        "timings": timings,
    }


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
    result = image_to_sfen_with_diagnostics(
        image,
        classifier=classifier,
        turn=turn,
        move_count=move_count,
    )
    return result["sfen"]
