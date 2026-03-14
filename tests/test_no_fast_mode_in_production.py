import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt

import backend_api


def test_run_path_disables_fast_mode_without_debug_opt_in(monkeypatch) -> None:
    state = backend_api.BackendState()
    captured = {}

    def _fake_build_ensemble_forecast(**kwargs):
        captured.update(kwargs)
        raise ValueError("stop after capture")

    monkeypatch.setattr(backend_api, "build_ensemble_forecast", _fake_build_ensemble_forecast)
    monkeypatch.delenv("PVBP_ENABLE_DEBUG_FAST_MODE", raising=False)

    try:
        state._run(
            target_date=dt.date.today() + dt.timedelta(days=1),
            soc_percent=50.0,
            yesterday_kwh=10.0,
            buffer_percent=0.0,
            user_max_ac_kw=3.0,
            weather_models=None,
            forecast_mode="auto",
            ensemble_method="weighted",
            pv_uncertainty=False,
            fast_mode=True,
        )
    except ValueError as exc:
        assert "stop after capture" in str(exc)
    else:
        raise AssertionError("Expected early stop")

    assert captured.get("fast_mode") is False
