# shogi-vision

将棋盤画像 (デジタルスクリーンショット) を SFEN 文字列に変換する Python パッケージ。

```
$ shogi-vision board.png
lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
```

## インストール

```bash
# Tesseract (OCR) — 本物の画像を扱うとき必要
sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn

# Python パッケージ
pip install -e .[dev]
```

## CLI

```bash
shogi-vision <image>                       # 自動 (TemplateClassifier)
shogi-vision <image> --classifier template # IPAGothic テンプレマッチ (合成画像向け)
shogi-vision <image> --classifier tesseract # Tesseract OCR を明示指定
shogi-vision <image> --turn w --move 42    # SFEN メタ情報を上書き
shogi-vision <image> --debug overlay.png   # 検出結果をオーバレイ表示
```

## Python API

```python
from shogi_vision import image_to_sfen, render_board
from shogi_vision.pieces import initial_board

# 画像 → SFEN
sfen = image_to_sfen("board.png")

# 局面 → 合成画像 (テスト用)
board, *_ = initial_board()
img = render_board(board, size=900)
```

## アーキテクチャ

```
画像 ──▶ board_detector ──▶ cell_segmenter ──▶ piece_classifier ──▶ pieces.board_to_sfen ──▶ SFEN
        (透視補正 900x900)   (9x9 セル抽出)     (1セル → 駒記号)
```

| モジュール | 役割 |
|---|---|
| `shogi_vision.pieces` | SFEN エンコード/デコード、駒種定義 |
| `shogi_vision.board_detector` | OpenCV で盤面を検出して 900×900 に正規化 |
| `shogi_vision.cell_segmenter` | 盤画像を 9×9 セルに分割 |
| `shogi_vision.piece_classifier` | 各セル → SFEN 駒記号 (Tesseract OCR / テンプレマッチ) |
| `shogi_vision.pipeline` | 全工程を結合 |
| `shogi_vision.synthetic` | テスト用の合成盤レンダラ (PIL + IPAGothic) |
| `shogi_vision.cli` | コマンドラインインターフェース |
| `shogi_vision.debug` | 予測結果のオーバレイ可視化 |

## 駒記号 (SFEN)

```
K=王玉  R=飛  B=角  G=金  S=銀  N=桂  L=香  P=歩
+R=龍   +B=馬       +S=全  +N=圭  +L=杏  +P=と
```

大文字 = 先手 (Black, 黒), 小文字 = 後手 (White, 白)。
盤の向きは **先手視点固定** (黒が下、白が上) を仮定する。

## テスト

```bash
pytest -v
```

合成盤画像を生成して end-to-end で動作確認する。Tesseract が未インストールでも
TemplateClassifier 経由で全テスト (71件) が通る。

## 既知の制約

- **Tesseract の単字認識限界**: `jpn` モデルは連続テキスト用に学習されているため、
  孤立した漢字 (特に `王` `銀` `香` `角`) で認識精度が落ちる。`auto` は
  TemplateClassifier を使う。OCR 経路を試す場合だけ `--classifier tesseract` を指定。
- **持ち駒は未対応**: SFEN の手駒部分は常に `-` を出力する。
- **盤の向き判定なし**: 先手視点に固定。逆向きの画像は事前に回転が必要。
