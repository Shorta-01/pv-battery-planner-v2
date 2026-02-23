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
