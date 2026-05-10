"""SFEN notation encoding/decoding and piece definitions."""
from __future__ import annotations

from enum import Enum
from typing import Optional

# Board dimensions
BOARD_RANKS = 9
BOARD_FILES = 9

# board[rank_idx][file_idx]
# rank_idx: 0 = rank 1 (white's back rank), 8 = rank 9 (black's back rank)
# file_idx: 0 = file 9 (left in diagram), 8 = file 1 (right in diagram)
Board = list[list[Optional[str]]]


class Player(Enum):
    BLACK = "b"  # 先手
    WHITE = "w"  # 後手


# All piece types in SFEN (Black's notation)
PIECE_TYPES: list[str] = [
    "K", "R", "B", "G", "S", "N", "L", "P",
    "+R", "+B", "+S", "+N", "+L", "+P",
]

# Kanji on each piece face (used for OCR-based detection)
KANJI_TO_PIECE: dict[str, str] = {
    "王": "K", "玉": "K",
    "飛": "R", "龍": "+R",
    "角": "B", "馬": "+B",
    "金": "G",
    "銀": "S", "全": "+S",
    "桂": "N", "圭": "+N",
    "香": "L", "杏": "+L",
    "歩": "P", "と": "+P",
}

# Piece type to display kanji (Black's piece face)
PIECE_TO_KANJI: dict[str, str] = {
    "K": "王",
    "R": "飛", "+R": "龍",
    "B": "角", "+B": "馬",
    "G": "金",
    "S": "銀", "+S": "全",
    "N": "桂", "+N": "圭",
    "L": "香", "+L": "杏",
    "P": "歩", "+P": "と",
}

# Pieces that can be held in hand (unpromoted only, no king)
HAND_PIECE_ORDER: list[str] = ["R", "B", "G", "S", "N", "L", "P"]

# Standard initial board position
INITIAL_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def piece_for_player(piece_type: str, player: Player) -> str:
    """Return the SFEN symbol for a piece type and player."""
    if player == Player.BLACK:
        return piece_type
    if piece_type.startswith("+"):
        return "+" + piece_type[1:].lower()
    return piece_type.lower()


def player_of(sfen_piece: str) -> Optional[Player]:
    """Return the player who owns a SFEN piece symbol."""
    if sfen_piece is None or sfen_piece == "":
        return None
    if sfen_piece.startswith("+"):
        return Player.BLACK if sfen_piece[1].isupper() else Player.WHITE
    return Player.BLACK if sfen_piece[0].isupper() else Player.WHITE


def piece_type_of(sfen_piece: str) -> Optional[str]:
    """Return the piece type (uppercase, possibly with '+') from a SFEN symbol."""
    if sfen_piece is None or sfen_piece == "":
        return None
    if sfen_piece.startswith("+"):
        return "+" + sfen_piece[1:].upper()
    return sfen_piece.upper()


def board_to_sfen(
    board: Board,
    turn: Player = Player.BLACK,
    hands: Optional[dict[str, int]] = None,
    move_count: int = 1,
) -> str:
    """Convert a 9x9 board array to a complete SFEN string.

    Board convention: board[rank_idx][file_idx]
      rank_idx 0 = rank 1 (top/white's side)
      file_idx 0 = file 9 (left in diagram)
    """
    rows: list[str] = []
    for rank_idx in range(BOARD_RANKS):
        row_str = ""
        empty = 0
        for file_idx in range(BOARD_FILES):
            cell = board[rank_idx][file_idx]
            if cell is None or cell == "":
                empty += 1
            else:
                if empty:
                    row_str += str(empty)
                    empty = 0
                row_str += cell
        if empty:
            row_str += str(empty)
        rows.append(row_str)

    board_str = "/".join(rows)
    turn_str = turn.value
    hands_str = _encode_hands(hands or {})
    return f"{board_str} {turn_str} {hands_str} {move_count}"


def sfen_to_board(sfen: str) -> tuple[Board, Player, dict[str, int], int]:
    """Parse a SFEN string into (board, turn, hands, move_count)."""
    parts = sfen.strip().split()
    if len(parts) != 4:
        raise ValueError(f"Invalid SFEN: expected 4 parts, got {len(parts)}: {sfen!r}")

    board_str, turn_str, hands_str, move_str = parts
    board = _decode_board(board_str)
    turn = Player.BLACK if turn_str == "b" else Player.WHITE
    hands = _decode_hands(hands_str)
    move_count = int(move_str)
    return board, turn, hands, move_count


def empty_board() -> Board:
    """Return an empty 9x9 board."""
    return [[None] * BOARD_FILES for _ in range(BOARD_RANKS)]


def initial_board() -> tuple[Board, Player, dict[str, int], int]:
    """Return the standard shogi starting position."""
    return sfen_to_board(INITIAL_SFEN)


# ── internal helpers ──────────────────────────────────────────────────────────

def _decode_board(board_str: str) -> Board:
    rows = board_str.split("/")
    if len(rows) != BOARD_RANKS:
        raise ValueError(f"Expected {BOARD_RANKS} ranks, got {len(rows)}")

    board: Board = []
    for row_str in rows:
        row: list[Optional[str]] = []
        i = 0
        while i < len(row_str):
            ch = row_str[i]
            if ch == "+":
                if i + 1 >= len(row_str):
                    raise ValueError(f"Trailing '+' in SFEN row: {row_str!r}")
                row.append("+" + row_str[i + 1])
                i += 2
            elif ch.isdigit():
                row.extend([None] * int(ch))
                i += 1
            else:
                row.append(ch)
                i += 1
        if len(row) != BOARD_FILES:
            raise ValueError(f"Expected {BOARD_FILES} files, got {len(row)} in row {row_str!r}")
        board.append(row)
    return board


def _encode_hands(hands: dict[str, int]) -> str:
    """Encode hands dict to SFEN.

    Key format: 'R', 'r', 'B', 'b', etc. (SFEN piece symbol without promotion)
    or 'black_R', 'white_r' style.
    """
    if not hands:
        return "-"

    result = ""
    for piece in HAND_PIECE_ORDER:
        count = hands.get(piece, 0) + hands.get(f"black_{piece}", 0)
        if count > 0:
            if count > 1:
                result += str(count)
            result += piece

    for piece in HAND_PIECE_ORDER:
        lp = piece.lower()
        count = hands.get(lp, 0) + hands.get(f"white_{piece}", 0)
        if count > 0:
            if count > 1:
                result += str(count)
            result += lp

    return result or "-"


def _decode_hands(hands_str: str) -> dict[str, int]:
    if hands_str == "-":
        return {}

    hands: dict[str, int] = {}
    i = 0
    while i < len(hands_str):
        count = 0
        while i < len(hands_str) and hands_str[i].isdigit():
            count = count * 10 + int(hands_str[i])
            i += 1
        if count == 0:
            count = 1
        if i < len(hands_str):
            piece = hands_str[i]
            hands[piece] = hands.get(piece, 0) + count
            i += 1
    return hands
