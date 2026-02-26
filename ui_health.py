from __future__ import annotations


def should_hard_stop(result: dict) -> tuple[bool, str, list[str]]:
    status = str(result.get("status") or "").strip().lower()
    warnings_raw = result.get("warnings")
    warnings = warnings_raw if isinstance(warnings_raw, list) else []
    warnings_count = int(result.get("warnings_count") or 0)
    fallback = [f"{warnings_count} warning(s) recorded"] if warnings_count else []

    hard_stop_statuses = {"error", "failed"}

    if status in hard_stop_statuses:
        return True, "Forecast failed", (warnings or fallback)

    if status == "degraded":
        return False, "Forecast completed with warnings", (warnings or fallback)

    return False, "", warnings


def model_indicators(model_id: str, weather_ensemble: dict | None) -> dict:
    if not weather_ensemble:
        return {"fetch": "na", "data": "na", "tooltip": "No diagnostics available."}

    failed = weather_ensemble.get("failed_model_reasons") or {}
    if model_id in failed:
        return {
            "fetch": "fail",
            "data": "na",
            "tooltip": str(failed[model_id]),
        }

    selected = set(weather_ensemble.get("selected_models") or [])
    if model_id not in selected:
        return {"fetch": "na", "data": "na", "tooltip": "Not selected for this run."}

    meta = (weather_ensemble.get("fetch_meta_by_model") or {}).get(model_id) or {}
    missing_overlap = int(meta.get("missing_hours_overlap") or 0)
    missing_vars = (weather_ensemble.get("missing_vars_by_model") or {}).get(model_id) or []
    pv_critical = {
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
    }
    pv_missing = [v for v in missing_vars if v in pv_critical]

    if missing_overlap == 0 and not pv_missing:
        return {
            "fetch": "ok",
            "data": "ok",
            "tooltip": "Coverage OK; PV-critical fields OK.",
        }

    return {
        "fetch": "ok",
        "data": "warn",
        "tooltip": f"missing_hours_overlap={missing_overlap}; pv_missing={pv_missing}",
    }
