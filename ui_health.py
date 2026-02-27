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


def compute_weather_health(
    weather_ensemble: dict | None,
    *,
    status: str,
    attempted_models: list[str] | None,
) -> tuple[str, str]:
    """
    returns (level, tooltip_text)
    level in {"ok","warn","fail","na"}
    tooltip_text is short multi-line string (<=6 lines)
    """
    if not weather_ensemble or not isinstance(weather_ensemble, dict):
        return ("na", "")

    status_l = (status or "").strip().lower()
    failed = weather_ensemble.get("failed_model_reasons", {}) or {}
    fetch_meta = weather_ensemble.get("fetch_meta_by_model", {}) or {}
    missing_vars = weather_ensemble.get("missing_vars_by_model", {}) or {}

    models = attempted_models or weather_ensemble.get("selected_models") or []
    models = [str(m) for m in models if isinstance(m, str)]

    critical_vars = {
        "direct_normal_irradiance",
        "diffuse_radiation",
        "shortwave_radiation",
    }

    detail_lines: list[str] = []
    has_overlap_or_critical_missing = False
    for model in models:
        if model in failed:
            detail_lines.append(f"{model}: fetch failed ({failed.get(model)})")
            continue

        meta = fetch_meta.get(model, {}) or {}
        missing_overlap = meta.get("missing_hours_overlap") or meta.get("missing_overlap") or 0
        try:
            missing_overlap_n = int(missing_overlap)
        except (TypeError, ValueError):
            missing_overlap_n = 0
        if missing_overlap_n:
            has_overlap_or_critical_missing = True
            detail_lines.append(f"{model}: missing hours overlap: {missing_overlap_n}")

        model_missing_vars = missing_vars.get(model, []) or []
        model_missing_vars = [str(v) for v in model_missing_vars if isinstance(v, str)]
        critical_missing = [v for v in model_missing_vars if v in critical_vars]
        vars_to_report = critical_missing or model_missing_vars[:3]
        if vars_to_report:
            if critical_missing:
                has_overlap_or_critical_missing = True
            detail_lines.append(f"{model}: missing vars: {','.join(vars_to_report)}")

    if status_l in {"error", "failed"} or (models and len(failed) == len(models)):
        level = "fail"
    elif failed or has_overlap_or_critical_missing:
        level = "warn"
    else:
        level = "ok"

    if level == "ok":
        return ("ok", "Weather: OK")

    header = "Weather: failed" if level == "fail" else "Weather: warnings"
    lines = [header] + detail_lines[:5]
    return (level, "\n".join(lines[:6]))
