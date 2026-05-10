"""Tests for SFEN encoding / decoding logic."""
from __future__ import annotations

import pytest

from shogi_vision.pieces import (
    INITIAL_SFEN,
    Player,
    board_to_sfen,
    empty_board,
    initial_board,
    piece_for_player,
    piece_type_of,
    player_of,
    sfen_to_board,
)


def test_initial_sfen_roundtrip():
    board, turn, hands, move_count = sfen_to_board(INITIAL_SFEN)
    result = board_to_sfen(board, turn=turn, hands=hands, move_count=move_count)
    assert result == INITIAL_SFEN


def test_initial_board_shape():
    board, _, _, _ = initial_board()
    assert len(board) == 9
    assert all(len(row) == 9 for row in board)


def test_initial_board_rank1():
    board, _, _, _ = initial_board()
    # rank 1 (index 0): White's back rank — l n s g k g s n l
    rank1 = board[0]
    assert rank1 == ["l", "n", "s", "g", "k", "g", "s", "n", "l"]


def test_initial_board_rank9():
    board, _, _, _ = initial_board()
    # rank 9 (index 8): Black's back rank — L N S G K G S N L
    rank9 = board[8]
    assert rank9 == ["L", "N", "S", "G", "K", "G", "S", "N", "L"]


def test_empty_board_sfen():
    board = empty_board()
    sfen = board_to_sfen(board, turn=Player.BLACK, move_count=1)
    assert sfen == "9/9/9/9/9/9/9/9/9 b - 1"


def test_player_of():
    assert player_of("K") == Player.BLACK
    assert player_of("k") == Player.WHITE
    assert player_of("+R") == Player.BLACK
    assert player_of("+r") == Player.WHITE
    assert player_of(None) is None
    assert player_of("") is None


def test_piece_type_of():
    assert piece_type_of("K") == "K"
    assert piece_type_of("k") == "K"
    assert piece_type_of("+r") == "+R"
    assert piece_type_of("+R") == "+R"
    assert piece_type_of(None) is None


def test_piece_for_player():
    assert piece_for_player("K", Player.BLACK) == "K"
    assert piece_for_player("K", Player.WHITE) == "k"
    assert piece_for_player("+R", Player.BLACK) == "+R"
    assert piece_for_player("+R", Player.WHITE) == "+r"


def test_hands_roundtrip():
    hands = {"R": 1, "r": 2, "P": 3}
    board = empty_board()
    sfen = board_to_sfen(board, turn=Player.WHITE, hands=hands, move_count=5)
    _, turn_out, hands_out, mc = sfen_to_board(sfen)
    assert turn_out == Player.WHITE
    assert mc == 5
    assert hands_out.get("R", 0) == 1
    assert hands_out.get("r", 0) == 2
    assert hands_out.get("P", 0) == 3


def test_promoted_pieces():
    board = empty_board()
    board[0][0] = "+R"  # Black's dragon at rank 1, file 9
    sfen = board_to_sfen(board, turn=Player.BLACK)
    assert sfen.startswith("+R")
    board2, _, _, _ = sfen_to_board(sfen)
    assert board2[0][0] == "+R"


def test_invalid_sfen_raises():
    with pytest.raises(ValueError):
        sfen_to_board("not a valid sfen")


def test_sfen_with_hand_no_pieces():
    board, turn, hands, mc = sfen_to_board("9/9/9/9/9/9/9/9/9 b - 1")
    assert hands == {}
    assert board_to_sfen(board, turn, hands, mc) == "9/9/9/9/9/9/9/9/9 b - 1"
