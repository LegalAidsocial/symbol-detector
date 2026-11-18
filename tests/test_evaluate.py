from symbol_detector.evaluate import ground_truth_verdict


def test_positive_is_banned() -> None:
    row = {"label": "positive", "symbol": "swastika", "group": "swastika_positive_0"}
    assert ground_truth_verdict(row) == "banned"


def test_religious_swastika_hard_negative_is_needs_context() -> None:
    row = {
        "label": "hard_negative",
        "symbol": "swastika",
        "group": "swastika_hard_negative_0",
    }
    assert ground_truth_verdict(row) == "needs_context"


def test_other_swastika_hard_negative_is_clean() -> None:
    row = {
        "label": "hard_negative",
        "symbol": "swastika",
        "group": "swastika_hard_negative_1",
    }
    assert ground_truth_verdict(row) == "clean"


def test_other_symbol_hard_negative_is_clean() -> None:
    row = {
        "label": "hard_negative",
        "symbol": "odal_rune",
        "group": "odal_rune_hard_negative_0",
    }
    assert ground_truth_verdict(row) == "clean"


def test_easy_negative_is_clean() -> None:
    row = {"label": "easy_negative", "symbol": "none", "group": "easy_negative_42"}
    assert ground_truth_verdict(row) == "clean"
