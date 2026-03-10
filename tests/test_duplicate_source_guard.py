import weather_ensemble as we


def test_dedupe_models_by_source_deterministic_winner(monkeypatch):
    fake_models = {
        "dup_low": {
            "endpoint": "https://example.test/v1/forecast",
            "params": {"models": "same-token"},
            "recommended_for_be": False,
            "max_days": 3,
            "tier": "global",
        },
        "dup_high": {
            "endpoint": "https://example.test/v1/forecast",
            "params": {"models": "same-token"},
            "recommended_for_be": True,
            "max_days": 10,
            "tier": "short",
        },
        "unique": {
            "endpoint": "https://example.test/v1/forecast",
            "params": {"models": "other-token"},
            "recommended_for_be": True,
            "max_days": 5,
            "tier": "medium",
        },
    }
    monkeypatch.setattr(we, "WEATHER_MODELS", fake_models)

    deduped, dropped = we.dedupe_models_by_source(["dup_low", "dup_high", "unique"])

    assert deduped == ["dup_high", "unique"]
    assert dropped == ["dup_low"]


def test_dedupe_models_by_source_repeated_same_id_unique_once(monkeypatch):
    fake_models = {
        "repeat": {
            "endpoint": "https://example.test/v1/forecast",
            "params": {"models": "same-token"},
            "recommended_for_be": True,
            "max_days": 7,
            "tier": "short",
        },
    }
    monkeypatch.setattr(we, "WEATHER_MODELS", fake_models)

    deduped, dropped = we.dedupe_models_by_source(["repeat", "repeat", "repeat"])

    assert deduped == ["repeat"]
    assert dropped == ["repeat", "repeat"]


def test_dedupe_models_by_source_same_source_collision_keeps_winner(monkeypatch):
    fake_models = {
        "loser": {
            "endpoint": "https://example.test/v1/forecast",
            "params": {"models": "same-token"},
            "recommended_for_be": False,
            "max_days": 3,
            "tier": "global",
        },
        "winner": {
            "endpoint": "https://example.test/v1/forecast",
            "params": {"models": "same-token"},
            "recommended_for_be": True,
            "max_days": 10,
            "tier": "short",
        },
    }
    monkeypatch.setattr(we, "WEATHER_MODELS", fake_models)

    deduped, dropped = we.dedupe_models_by_source(["loser", "winner"])

    assert deduped == ["winner"]
    assert dropped == ["loser"]


def test_dedupe_models_by_source_mixed_repeats_and_collisions(monkeypatch):
    fake_models = {
        "repeat": {
            "endpoint": "https://example.test/v1/forecast",
            "params": {"models": "token-a"},
            "recommended_for_be": True,
            "max_days": 8,
            "tier": "short",
        },
        "alt": {
            "endpoint": "https://example.test/v1/forecast",
            "params": {"models": "token-a"},
            "recommended_for_be": False,
            "max_days": 4,
            "tier": "global",
        },
        "other": {
            "endpoint": "https://example.test/v1/forecast",
            "params": {"models": "token-b"},
            "recommended_for_be": True,
            "max_days": 5,
            "tier": "medium",
        },
    }
    monkeypatch.setattr(we, "WEATHER_MODELS", fake_models)

    deduped, dropped = we.dedupe_models_by_source(["repeat", "alt", "repeat", "other", "repeat"])

    assert deduped == ["repeat", "other"]
    assert dropped == ["alt", "repeat", "repeat"]


def test_dedupe_models_by_source_order_is_deterministic(monkeypatch):
    fake_models = {
        "b": {
            "endpoint": "https://example.test/v1/forecast",
            "params": {"models": "token-b"},
            "recommended_for_be": True,
            "max_days": 5,
            "tier": "medium",
        },
        "a": {
            "endpoint": "https://example.test/v1/forecast",
            "params": {"models": "token-a"},
            "recommended_for_be": True,
            "max_days": 5,
            "tier": "medium",
        },
    }
    monkeypatch.setattr(we, "WEATHER_MODELS", fake_models)

    deduped, dropped = we.dedupe_models_by_source(["b", "a", "b", "a"])

    assert deduped == ["b", "a"]
    assert dropped == ["b", "a"]
