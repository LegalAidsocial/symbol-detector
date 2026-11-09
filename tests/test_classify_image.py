from symbol_detector.classify_image import verdict_for


def test_banned_symbol_is_banned() -> None:
    assert verdict_for("swastika") == "banned"
    assert verdict_for("ss_deaths_head") == "banned"
    assert verdict_for("hamas_flag") == "banned"


def test_religious_swastika_needs_context_not_banned() -> None:
    assert verdict_for("swastika_religious") == "needs_context"


def test_none_class_is_clean() -> None:
    assert verdict_for("none") == "clean"
