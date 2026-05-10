"""CLI integration test via subprocess."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import pytest

from shogi_vision.pieces import INITIAL_SFEN, initial_board
from shogi_vision.synthetic import render_board


@pytest.fixture(scope="module")
def initial_image_path(tmp_path_factory):
    board, *_ = initial_board()
    img = render_board(board, size=900)
    p = tmp_path_factory.mktemp("cli") / "initial.png"
    cv2.imwrite(str(p), img)
    return p


def test_cli_initial_position(initial_image_path):
    result = subprocess.run(
        [sys.executable, "-m", "shogi_vision.cli", str(initial_image_path),
         "--classifier", "template"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == INITIAL_SFEN


def test_cli_missing_file():
    result = subprocess.run(
        [sys.executable, "-m", "shogi_vision.cli", "nonexistent.png",
         "--classifier", "template"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0


def test_cli_turn_white(initial_image_path):
    result = subprocess.run(
        [sys.executable, "-m", "shogi_vision.cli", str(initial_image_path),
         "--classifier", "template", "--turn", "w"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    parts = result.stdout.strip().split()
    assert parts[1] == "w"
