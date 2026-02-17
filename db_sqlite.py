from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import uuid
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TIMEZONE = "Europe/Brussels"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: str) -> None:
    db_file = Path(db_path)
    _ensure_parent(db_file)
    with _connect(str(db_file)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS forecast_runs (
                run_id TEXT PRIMARY KEY,
                target_date TEXT NOT NULL,
                run_at_utc TEXT NOT NULL,
                run_type TEXT,
                status TEXT,
                timezone TEXT,
                charge_kw REAL,
                cutoff_soc REAL,
                pv_forecast_kwh REAL,
                cons_forecast_kwh REAL,
                warnings_count INTEGER,
                inputs_used_json TEXT,
                weather_ensemble_json TEXT,
                pv_p10_kwh REAL,
                pv_p50_kwh REAL,
                pv_p90_kwh REAL,
                run_duration_ms INTEGER,
                config_schema_version INTEGER,
                soc_at_22_used REAL,
                yesterday_kwh_used REAL,
                planner_version TEXT,
                config_hash TEXT,
                config_json TEXT,
                warnings_json TEXT,
                pv_quality TEXT,
                created_at_utc TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_forecast_runs_target_date ON forecast_runs (target_date);
            CREATE INDEX IF NOT EXISTS idx_forecast_runs_run_at ON forecast_runs (run_at_utc);

            CREATE TABLE IF NOT EXISTS forecast_hourly (
                run_id TEXT NOT NULL,
                ts_local TEXT NOT NULL,
                pv_kwh REAL,
                pv_total_unclipped_kwh REAL,
                pv_east_kwh REAL,
                pv_south_kwh REAL,
                pv_clipped_kwh REAL,
                load_kwh REAL,
                grid_import_kwh REAL,
                grid_export_kwh REAL,
                batt_charge_kwh REAL,
                batt_discharge_kwh REAL,
                soc_pct REAL,
                PRIMARY KEY (run_id, ts_local)
            );
            CREATE INDEX IF NOT EXISTS idx_forecast_hourly_ts ON forecast_hourly (ts_local);

            -- Future FusionSolar ingestion table. Join with forecast_hourly on ts_local.
            CREATE TABLE IF NOT EXISTS actual_hourly (
                source TEXT NOT NULL,
                ts_local TEXT NOT NULL,
                pv_kwh REAL,
                load_kwh REAL,
                grid_import_kwh REAL,
                grid_export_kwh REAL,
                soc_pct REAL,
                completeness REAL,
                PRIMARY KEY (source, ts_local)
            );
            CREATE INDEX IF NOT EXISTS idx_actual_hourly_ts ON actual_hourly (ts_local);

            -- Future forecast-vs-actual score table. Join on score_date and run_id.
            CREATE TABLE IF NOT EXISTS daily_scores (
                run_id TEXT PRIMARY KEY,
                score_date TEXT NOT NULL,
                pv_mae_kwh REAL,
                pv_mape REAL,
                load_mae_kwh REAL,
                load_mape REAL,
                soc_mae_pct REAL,
                import_error_kwh REAL,
                export_error_kwh REAL,
                created_at_utc TEXT NOT NULL
            );
            """
        )

        existing_cols = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(forecast_hourly)").fetchall()
        }
        for col_name in [
            "pv_total_unclipped_kwh",
            "pv_east_kwh",
            "pv_south_kwh",
            "pv_clipped_kwh",
        ]:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE forecast_hourly ADD COLUMN {col_name} REAL")

        forecast_runs_existing_cols = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(forecast_runs)").fetchall()
        }
        for col_name, col_type in [
            ("status", "TEXT"),
            ("warnings_count", "INTEGER"),
            ("inputs_used_json", "TEXT"),
            ("weather_ensemble_json", "TEXT"),
            ("pv_p10_kwh", "REAL"),
            ("pv_p50_kwh", "REAL"),
            ("pv_p90_kwh", "REAL"),
            ("run_duration_ms", "INTEGER"),
            ("config_schema_version", "INTEGER"),
        ]:
            if col_name not in forecast_runs_existing_cols:
                conn.execute(f"ALTER TABLE forecast_runs ADD COLUMN {col_name} {col_type}")

        conn.execute(
            """
            UPDATE forecast_hourly
            SET
                pv_total_unclipped_kwh = COALESCE(pv_total_unclipped_kwh, pv_kwh, 0.0),
                pv_east_kwh = COALESCE(pv_east_kwh, 0.0),
                pv_south_kwh = COALESCE(pv_south_kwh, pv_kwh, 0.0),
                pv_clipped_kwh = COALESCE(pv_clipped_kwh, 0.0)
            WHERE
                pv_total_unclipped_kwh IS NULL
                OR pv_east_kwh IS NULL
                OR pv_south_kwh IS NULL
                OR pv_clipped_kwh IS NULL
            """
        )


def compute_config_hash(config: dict) -> str:
    serialized = json.dumps(config or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _iso_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _parse_df(payload: dict, key: str) -> pd.DataFrame:
    raw = payload.get(key)
    if not isinstance(raw, dict):
        return pd.DataFrame()
    try:
        return pd.read_json(StringIO(json.dumps(raw)), orient="split")
    except ValueError:
        return pd.DataFrame()


def _parse_series(payload: dict, key: str) -> pd.Series:
    frame = _parse_df(payload, key)
    if "value" in frame.columns:
        return frame["value"]
    return pd.Series(dtype=float)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_local_hour_str(value: Any) -> str | None:
    stamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(stamp):
        return None
    if hasattr(stamp, "to_pydatetime"):
        stamp = stamp.to_pydatetime()
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.replace(tzinfo=None)
    return stamp.strftime("%Y-%m-%dT%H:00:00")


def _normalize_hourly(payload: dict) -> list[dict]:
    pv_df = _parse_df(payload, "pv")
    flows_df = _parse_df(payload, "flows")
    soc_series = _parse_series(payload, "soc")

    base_index = None
    for candidate in (pv_df.index, flows_df.index, soc_series.index):
        if len(candidate) > 0:
            base_index = candidate
            break
    if base_index is None:
        return []

    rows: list[dict] = []
    for ts in base_index:
        ts_local = _to_local_hour_str(ts)
        if not ts_local:
            continue

        pv_kwh = _safe_float(pv_df.at[ts, "pv_total_kwh"]) if "pv_total_kwh" in pv_df.columns and ts in pv_df.index else None
        pv_total_unclipped_kwh = _safe_float(pv_df.at[ts, "pv_total_unclipped_kwh"]) if "pv_total_unclipped_kwh" in pv_df.columns and ts in pv_df.index else pv_kwh
        pv_east_kwh = _safe_float(pv_df.at[ts, "pv_east_kwh"]) if "pv_east_kwh" in pv_df.columns and ts in pv_df.index else 0.0
        pv_south_kwh = _safe_float(pv_df.at[ts, "pv_south_kwh"]) if "pv_south_kwh" in pv_df.columns and ts in pv_df.index else pv_kwh
        pv_clipped_kwh = _safe_float(pv_df.at[ts, "pv_clipped_kwh"]) if "pv_clipped_kwh" in pv_df.columns and ts in pv_df.index else 0.0
        load_kwh = _safe_float(pv_df.at[ts, "load_kwh"]) if "load_kwh" in pv_df.columns and ts in pv_df.index else None
        grid_import = _safe_float(flows_df.at[ts, "grid_import_kwh"]) if "grid_import_kwh" in flows_df.columns and ts in flows_df.index else None
        grid_export = _safe_float(flows_df.at[ts, "grid_export_kwh"]) if "grid_export_kwh" in flows_df.columns and ts in flows_df.index else None
        batt_charge = _safe_float(flows_df.at[ts, "batt_charge_kwh"]) if "batt_charge_kwh" in flows_df.columns and ts in flows_df.index else None
        batt_discharge = _safe_float(flows_df.at[ts, "batt_discharge_kwh"]) if "batt_discharge_kwh" in flows_df.columns and ts in flows_df.index else None

        soc_pct = None
        if "soc_end_pct" in flows_df.columns and ts in flows_df.index:
            soc_pct = _safe_float(flows_df.at[ts, "soc_end_pct"])
        elif ts in soc_series.index:
            soc_val = _safe_float(soc_series.at[ts])
            if soc_val is not None:
                soc_pct = soc_val * 100.0 if soc_val <= 1.0 else soc_val

        rows.append(
            {
                "ts_local": ts_local,
                "pv_kwh": pv_kwh,
                "pv_total_unclipped_kwh": pv_total_unclipped_kwh,
                "pv_east_kwh": pv_east_kwh,
                "pv_south_kwh": pv_south_kwh,
                "pv_clipped_kwh": pv_clipped_kwh,
                "load_kwh": load_kwh,
                "grid_import_kwh": grid_import,
                "grid_export_kwh": grid_export,
                "batt_charge_kwh": batt_charge,
                "batt_discharge_kwh": batt_discharge,
                "soc_pct": soc_pct,
            }
        )
    return rows


def insert_forecast_run(db_path: str, payload: dict) -> None:
    if not isinstance(payload, dict):
        return

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    inputs_used = payload.get("inputs_used") if isinstance(payload.get("inputs_used"), dict) else {}
    if not inputs_used:
        inputs_used = {
            "soc_at_22_percent": payload.get("soc_at_22_percent"),
            "yesterday_consumption_kwh": payload.get("yesterday_consumption_kwh"),
        }
    weather_ensemble = payload.get("weather_ensemble") if isinstance(payload.get("weather_ensemble"), dict) else {}

    run_id = str(payload.get("run_id") or uuid.uuid4())
    target_date = str(payload.get("target_date") or "")
    if not target_date:
        return

    run_at_utc = str(payload.get("run_at_utc") or _iso_utc_now())
    timezone = str(payload.get("timezone") or payload.get("system_snapshot", {}).get("timezone") or DEFAULT_TIMEZONE)

    hourly_rows = _normalize_hourly(payload)

    pv_forecast_kwh = _safe_float(metrics.get("pv_forecast_kwh"))
    cons_forecast_kwh = _safe_float(metrics.get("cons_forecast_kwh"))
    if pv_forecast_kwh is None:
        pv_forecast_kwh = float(sum((row.get("pv_kwh") or 0.0) for row in hourly_rows))
    if cons_forecast_kwh is None:
        cons_forecast_kwh = float(sum((row.get("load_kwh") or 0.0) for row in hourly_rows))

    cutoff_soc = _safe_float(metrics.get("cutoff_soc"))
    if cutoff_soc is not None and cutoff_soc <= 1.0:
        cutoff_soc = cutoff_soc * 100.0

    config_obj = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    config_hash = payload.get("config_hash") or compute_config_hash(config_obj)
    config_json = payload.get("config_json")
    if config_json is None:
        config_json = json.dumps(config_obj, sort_keys=True)

    warnings_json = json.dumps(payload.get("warnings", []))
    inputs_used_json = json.dumps(inputs_used, sort_keys=True)
    weather_ensemble_json = json.dumps(weather_ensemble, sort_keys=True)
    pv_quality = payload.get("pv_quality")
    pv_quality_text = json.dumps(pv_quality) if isinstance(pv_quality, dict) else None

    row_data = {
        "run_id": run_id,
        "target_date": target_date,
        "run_at_utc": run_at_utc,
        "run_type": payload.get("run_type"),
        "status": payload.get("status"),
        "timezone": timezone,
        "charge_kw": _safe_float(metrics.get("charge_kw")),
        "cutoff_soc": cutoff_soc,
        "pv_forecast_kwh": pv_forecast_kwh,
        "cons_forecast_kwh": cons_forecast_kwh,
        "warnings_count": int(payload.get("warnings_count") or len(payload.get("warnings", []))),
        "inputs_used_json": inputs_used_json,
        "weather_ensemble_json": weather_ensemble_json,
        "pv_p10_kwh": _safe_float((payload.get("pv_totals_kwh") or {}).get("p10")),
        "pv_p50_kwh": _safe_float((payload.get("pv_totals_kwh") or {}).get("p50")),
        "pv_p90_kwh": _safe_float((payload.get("pv_totals_kwh") or {}).get("p90")),
        "run_duration_ms": int(payload.get("run_duration_ms")) if payload.get("run_duration_ms") is not None else None,
        "config_schema_version": payload.get("config_schema_version"),
        "soc_at_22_used": _safe_float(inputs_used.get("soc_at_22_percent")),
        "yesterday_kwh_used": _safe_float(inputs_used.get("yesterday_consumption_kwh")),
        "planner_version": payload.get("planner_version"),
        "config_hash": config_hash,
        "config_json": config_json,
        "warnings_json": warnings_json,
        "pv_quality": pv_quality_text,
        "created_at_utc": payload.get("created_at_utc") or run_at_utc,
    }

    with _connect(db_path) as conn:
        conn.execute("BEGIN")
        conn.execute(
            """
            INSERT OR REPLACE INTO forecast_runs (
                run_id, target_date, run_at_utc, run_type, status, timezone,
                charge_kw, cutoff_soc, pv_forecast_kwh, cons_forecast_kwh, warnings_count,
                inputs_used_json, weather_ensemble_json, pv_p10_kwh, pv_p50_kwh, pv_p90_kwh,
                run_duration_ms, config_schema_version,
                soc_at_22_used, yesterday_kwh_used, planner_version,
                config_hash, config_json, warnings_json, pv_quality, created_at_utc
            ) VALUES (
                :run_id, :target_date, :run_at_utc, :run_type, :status, :timezone,
                :charge_kw, :cutoff_soc, :pv_forecast_kwh, :cons_forecast_kwh, :warnings_count,
                :inputs_used_json, :weather_ensemble_json, :pv_p10_kwh, :pv_p50_kwh, :pv_p90_kwh,
                :run_duration_ms, :config_schema_version,
                :soc_at_22_used, :yesterday_kwh_used, :planner_version,
                :config_hash, :config_json, :warnings_json, :pv_quality, :created_at_utc
            )
            """,
            row_data,
        )
        conn.execute("DELETE FROM forecast_hourly WHERE run_id = ?", (run_id,))
        if hourly_rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO forecast_hourly (
                    run_id, ts_local, pv_kwh, pv_total_unclipped_kwh, pv_east_kwh, pv_south_kwh, pv_clipped_kwh, load_kwh,
                    grid_import_kwh, grid_export_kwh,
                    batt_charge_kwh, batt_discharge_kwh, soc_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        row["ts_local"],
                        row["pv_kwh"],
                        row["pv_total_unclipped_kwh"],
                        row["pv_east_kwh"],
                        row["pv_south_kwh"],
                        row["pv_clipped_kwh"],
                        row["load_kwh"],
                        row["grid_import_kwh"],
                        row["grid_export_kwh"],
                        row["batt_charge_kwh"],
                        row["batt_discharge_kwh"],
                        row["soc_pct"],
                    )
                    for row in hourly_rows
                ],
            )
        conn.commit()


def fetch_recent_run_summaries(db_path: str, limit: int = 30) -> list[dict]:
    limit = max(1, int(limit))
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT target_date, run_at_utc, run_type, charge_kw, cutoff_soc,
                   pv_forecast_kwh, cons_forecast_kwh
            FROM forecast_runs
            ORDER BY run_at_utc DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    items: list[dict] = []
    for row in rows:
        items.append(
            {
                "target_date": row["target_date"],
                "metrics": {
                    "charge_kw": float(row["charge_kw"] or 0.0),
                    "cutoff_soc": float(row["cutoff_soc"] or 0.0) / 100.0,
                    "pv_forecast_kwh": float(row["pv_forecast_kwh"] or 0.0),
                    "cons_forecast_kwh": float(row["cons_forecast_kwh"] or 0.0),
                },
                "run_at": row["run_at_utc"],
                "run_type": row["run_type"] or "manual",
            }
        )
    items.reverse()
    return items


def _summary_from_row(row: sqlite3.Row) -> dict:
    return {
        "target_date": row["target_date"],
        "metrics": {
            "charge_kw": float(row["charge_kw"] or 0.0),
            "cutoff_soc": float(row["cutoff_soc"] or 0.0) / 100.0,
            "pv_forecast_kwh": float(row["pv_forecast_kwh"] or 0.0),
            "cons_forecast_kwh": float(row["cons_forecast_kwh"] or 0.0),
        },
        "run_at": row["run_at_utc"],
        "run_type": row["run_type"] or "manual",
    }


def fetch_history_latest_per_day(db_path: str, limit_days: int | None = None) -> list[dict]:
    params: list[Any] = []
    where_clauses = ["rn = 1"]
    if limit_days is not None:
        safe_days = max(1, int(limit_days))
        where_clauses.append(
            "target_date IN (SELECT target_date FROM forecast_runs GROUP BY target_date ORDER BY target_date DESC LIMIT ?)"
        )
        params.append(safe_days)

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            WITH ranked AS (
                SELECT
                    run_id,
                    target_date,
                    run_at_utc,
                    created_at_utc,
                    run_type,
                    charge_kw,
                    cutoff_soc,
                    pv_forecast_kwh,
                    cons_forecast_kwh,
                    ROW_NUMBER() OVER (
                        PARTITION BY target_date
                        ORDER BY run_at_utc DESC, COALESCE(created_at_utc, run_at_utc) DESC, run_id DESC
                    ) AS rn
                FROM forecast_runs
            )
            SELECT target_date, run_at_utc, run_type, charge_kw, cutoff_soc, pv_forecast_kwh, cons_forecast_kwh
            FROM ranked
            WHERE {' AND '.join(where_clauses)}
            ORDER BY target_date ASC
            """,
            params,
        ).fetchall()
    return [_summary_from_row(row) for row in rows]


def fetch_history_all_runs(db_path: str, limit_days: int | None = None) -> list[dict]:
    params: list[Any] = []
    date_filter_sql = ""
    if limit_days is not None:
        safe_days = max(1, int(limit_days))
        date_filter_sql = "WHERE target_date IN (SELECT target_date FROM forecast_runs GROUP BY target_date ORDER BY target_date DESC LIMIT ?)"
        params.append(safe_days)

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT target_date, run_at_utc, run_type, charge_kw, cutoff_soc, pv_forecast_kwh, cons_forecast_kwh
            FROM forecast_runs
            {date_filter_sql}
            ORDER BY target_date ASC, run_at_utc ASC, COALESCE(created_at_utc, run_at_utc) ASC, run_id ASC
            """,
            params,
        ).fetchall()
    return [_summary_from_row(row) for row in rows]


def fetch_latest_full_run(db_path: str) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT run_id, target_date, run_at_utc, run_type, timezone,
                   charge_kw, cutoff_soc, pv_forecast_kwh, cons_forecast_kwh,
                   warnings_json
            FROM forecast_runs
            ORDER BY run_at_utc DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        hourly_rows = conn.execute(
            """
            SELECT ts_local, pv_kwh, pv_total_unclipped_kwh, pv_east_kwh, pv_south_kwh, pv_clipped_kwh,
                   load_kwh, grid_import_kwh, grid_export_kwh,
                   batt_charge_kwh, batt_discharge_kwh, soc_pct
            FROM forecast_hourly
            WHERE run_id = ?
            ORDER BY ts_local ASC
            """,
            (row["run_id"],),
        ).fetchall()

    hourly = pd.DataFrame([dict(r) for r in hourly_rows])
    if hourly.empty:
        return {
            "run_id": row["run_id"],
            "target_date": row["target_date"],
            "metrics": {
                "charge_kw": float(row["charge_kw"] or 0.0),
                "cutoff_soc": float(row["cutoff_soc"] or 0.0) / 100.0,
                "pv_forecast_kwh": float(row["pv_forecast_kwh"] or 0.0),
                "cons_forecast_kwh": float(row["cons_forecast_kwh"] or 0.0),
            },
            "warnings": json.loads(row["warnings_json"] or "[]"),
            "run_at": row["run_at_utc"],
            "run_type": row["run_type"] or "manual",
        }

    idx = pd.to_datetime(hourly["ts_local"], errors="coerce")
    pv_df = pd.DataFrame(index=idx)
    pv_df["pv_total_kwh"] = pd.to_numeric(hourly["pv_kwh"], errors="coerce").fillna(0.0)
    pv_df["pv_total_unclipped_kwh"] = pd.to_numeric(hourly.get("pv_total_unclipped_kwh"), errors="coerce").fillna(pv_df["pv_total_kwh"])
    pv_df["pv_east_kwh"] = pd.to_numeric(hourly.get("pv_east_kwh"), errors="coerce").fillna(0.0)
    pv_df["pv_south_kwh"] = pd.to_numeric(hourly.get("pv_south_kwh"), errors="coerce").fillna(pv_df["pv_total_kwh"])
    pv_df["pv_clipped_kwh"] = pd.to_numeric(hourly.get("pv_clipped_kwh"), errors="coerce").fillna(0.0)
    pv_df["load_kwh"] = pd.to_numeric(hourly["load_kwh"], errors="coerce").fillna(0.0)

    flows_df = pd.DataFrame(index=idx)
    flows_df["grid_import_kwh"] = pd.to_numeric(hourly["grid_import_kwh"], errors="coerce").fillna(0.0)
    flows_df["grid_export_kwh"] = pd.to_numeric(hourly["grid_export_kwh"], errors="coerce").fillna(0.0)
    flows_df["batt_charge_kwh"] = pd.to_numeric(hourly["batt_charge_kwh"], errors="coerce").fillna(0.0)
    flows_df["batt_discharge_kwh"] = pd.to_numeric(hourly["batt_discharge_kwh"], errors="coerce").fillna(0.0)
    flows_df["soc_end_pct"] = pd.to_numeric(hourly["soc_pct"], errors="coerce").fillna(0.0)

    soc_series = (pd.to_numeric(hourly["soc_pct"], errors="coerce").fillna(0.0) / 100.0)
    soc_series.index = idx

    return {
        "run_id": row["run_id"],
        "target_date": row["target_date"],
        "pv": json.loads(pv_df.to_json(date_format="iso", orient="split")),
        "flows": json.loads(flows_df.to_json(date_format="iso", orient="split")),
        "soc": json.loads(soc_series.to_frame(name="value").to_json(date_format="iso", orient="split")),
        "metrics": {
            "charge_kw": float(row["charge_kw"] or 0.0),
            "cutoff_soc": float(row["cutoff_soc"] or 0.0) / 100.0,
            "pv_forecast_kwh": float(row["pv_forecast_kwh"] or 0.0),
            "cons_forecast_kwh": float(row["cons_forecast_kwh"] or 0.0),
        },
        "warnings": json.loads(row["warnings_json"] or "[]"),
        "run_at": row["run_at_utc"],
        "run_type": row["run_type"] or "manual",
    }
