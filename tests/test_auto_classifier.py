from shogi_vision.piece_classifier import TemplateClassifier, auto_classifier


def test_auto_classifier_defaults_to_template():
    assert isinstance(auto_classifier(), TemplateClassifier)
