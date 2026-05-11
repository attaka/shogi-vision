"""Command-line interface: shogi-vision <image> [options]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shogi-vision",
        description="Convert a shogi board image to SFEN notation.",
    )
    p.add_argument("image", type=Path, help="Path to the board image.")
    p.add_argument(
        "--turn", choices=["b", "w"], default="b",
        help="Whose turn it is (b=Black/先手, w=White/後手). Default: b",
    )
    p.add_argument(
        "--move", type=int, default=1,
        help="Move number to embed in SFEN. Default: 1",
    )
    p.add_argument(
        "--classifier", choices=["auto", "tesseract", "template"],
        default="auto",
        help=(
            "Piece classifier to use. "
            "'auto' uses Template by default; use 'tesseract' explicitly for OCR. "
            "Default: auto"
        ),
    )
    p.add_argument(
        "--debug", type=Path, default=None, metavar="PATH",
        help="Save a debug overlay image to PATH.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    from .piece_classifier import (
        TesseractClassifier,
        TemplateClassifier,
        auto_classifier,
    )
    from .pieces import Player
    from .pipeline import image_to_sfen

    if args.classifier == "tesseract":
        try:
            clf = TesseractClassifier()
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    elif args.classifier == "template":
        clf = TemplateClassifier()
    else:
        clf = auto_classifier()

    turn = Player.BLACK if args.turn == "b" else Player.WHITE

    try:
        sfen = image_to_sfen(args.image, classifier=clf, turn=turn, move_count=args.move)
    except Exception as exc:
        print(f"Error processing image: {exc}", file=sys.stderr)
        sys.exit(1)

    print(sfen)

    if args.debug:
        _save_debug(args.image, args.debug, clf, turn, args.move)


def _save_debug(
    image_path: Path, out_path: Path, clf, turn, move_count: int
) -> None:
    """Write a debug overlay image showing the detected grid and piece labels."""
    from .board_detector import load_image, detect_board
    from .cell_segmenter import segment_cells
    from .debug import draw_overlay
    import cv2

    bgr = load_image(image_path)
    board_img = detect_board(bgr)
    cells = segment_cells(board_img)
    overlay = draw_overlay(board_img, cells, clf)
    cv2.imwrite(str(out_path), overlay)
    print(f"Debug image saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
