from shogi_vision.synthetic import render_sfen_pretty


def test_render_sfen_pretty_shape():
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    img = render_sfen_pretty(sfen, size=900)
    assert img.shape == (900, 900, 3)

