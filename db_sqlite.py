from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
import sqlite3
import uuid
from io import StringIO
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import pandas as pd

from error_logging import MAX_ERROR_BODY_CHARS, iso_utc_now, trim

DEFAULT_TIMEZONE = "Europe/Brussels"
CURRENT_CONFIG_SCHEMA_VERSION = 1
MAX_PROVIDER_RESPONSE_CHARS = 250_000
logger = logging.getLogger(__name__)

_SQLITE_PROFILE_ENV_VAR = "PVBP_SQLITE_PROFILE"
_SQLITE_DEFAULT_PROFILE = "laptop"
_SQLITE_PROFILE_ALIASES = {
    "": _SQLITE_DEFAULT_PROFILE,
    "laptop": "laptop",
    "default": "laptop",
    "pi": "pi",
    "rpi": "pi",
    "raspberrypi": "pi",
    "raspberry_pi": "pi",
}
_SQLITE_PROFILE_PRAGMAS: dict[str, dict[str, int]] = {
    "pi": {
        "cache_size": -32768,
        "mmap_size": 33554432,
        "wal_autocheckpoint": 256,
        "journal_size_limit": 67108864,
    },
    "laptop": {
        "cache_size": -131072,
        "mmap_size": 268435456,
        "wal_autocheckpoint": 1000,
        "journal_size_limit": 134217728,
    },
}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_sqlite_profile() -> str:
    raw = os.getenv(_SQLITE_PROFILE_ENV_VAR, "")
    return _SQLITE_PROFILE_ALIASES.get(str(raw).strip().lower(), _SQLITE_DEFAULT_PROFILE)


def _apply_sqlite_pragmas(conn: sqlite3.Connection, *, profile: str) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")

    profile_pragmas = _SQLITE_PROFILE_PRAGMAS[profile]
    conn.execute(f"PRAGMA cache_size={int(profile_pragmas['cache_size'])};")
    conn.execute(f"PRAGMA mmap_size={int(profile_pragmas['mmap_size'])};")
    conn.execute(f"PRAGMA wal_autocheckpoint={int(profile_pragmas['wal_autocheckpoint'])};")
    conn.execute(f"PRAGMA journal_size_limit={int(profile_pragmas['journal_size_limit'])};")


def get_sqlite_pragma_snapshot(db_path: str) -> dict[str, Any]:
    with _connect(db_path) as conn:
        return {
            "profile": _resolve_sqlite_profile(),
            "journal_mode": str(conn.execute("PRAGMA journal_mode;").fetchone()[0]),
            "synchronous": int(conn.execute("PRAGMA synchronous;").fetchone()[0]),
            "busy_timeout": int(conn.execute("PRAGMA busy_timeout;").fetchone()[0]),
            "temp_store": int(conn.execute("PRAGMA temp_store;").fetchone()[0]),
            "cache_size": int(conn.execute("PRAGMA cache_size;").fetchone()[0]),
            "mmap_size": int(conn.execute("PRAGMA mmap_size;").fetchone()[0]),
            "wal_autocheckpoint": int(conn.execute("PRAGMA wal_autocheckpoint;").fetchone()[0]),
            "journal_size_limit": int(conn.execute("PRAGMA journal_size_limit;").fetchone()[0]),
            "foreign_keys": int(conn.execute("PRAGMA foreign_keys;").fetchone()[0]),
        }


@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        _apply_sqlite_pragmas(conn, profile=_resolve_sqlite_profile())
        with conn:
            yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


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
                pv_week_ahead_json TEXT,
                pv_p10_kwh REAL,
                pv_p50_kwh REAL,
                pv_p90_kwh REAL,
                run_duration_ms INTEGER,
                config_schema_version INTEGER,
                soc_at_22_used REAL,
                soc_now_used REAL,
                yesterday_kwh_used REAL,
                planner_version TEXT,
                config_hash TEXT,
                config_json TEXT,
                warnings_json TEXT,
                pv_quality TEXT,
                created_at_utc TEXT,
                mode TEXT,
                requested_days INTEGER,
                models_used_json TEXT,
                ensemble_method TEXT,
                weights_used_json TEXT,
                config_snapshot_json TEXT,
                input_snapshot_json TEXT
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

            CREATE TABLE IF NOT EXISTS weather_hourly_by_model (
                run_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                ts_local TEXT NOT NULL,
                ghi REAL,
                dni REAL,
                dhi REAL,
                dni_source TEXT,
                dhi_source TEXT,
                temp_c REAL,
                wind_ms REAL,
                cloud_pct REAL,
                weather_code INTEGER,
                PRIMARY KEY (run_id, model_id, ts_local)
            );
            CREATE INDEX IF NOT EXISTS idx_weather_hourly_model_ts ON weather_hourly_by_model (model_id, ts_local);
            CREATE INDEX IF NOT EXISTS idx_weather_hourly_run_model ON weather_hourly_by_model (run_id, model_id);

            CREATE TABLE IF NOT EXISTS pv_hourly_by_model (
                run_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                ts_local TEXT NOT NULL,
                pv_east_kwh REAL,
                pv_south_kwh REAL,
                pv_total_kwh REAL,
                pv_unclipped_kwh REAL,
                pv_clipped_kwh REAL,
                dc_kw REAL,
                ac_kw REAL,
                PRIMARY KEY (run_id, model_id, ts_local)
            );
            CREATE INDEX IF NOT EXISTS idx_pv_hourly_model_ts ON pv_hourly_by_model (model_id, ts_local);
            CREATE INDEX IF NOT EXISTS idx_pv_hourly_run_model ON pv_hourly_by_model (run_id, model_id);

            CREATE TABLE IF NOT EXISTS run_ensemble_hourly (
                run_id TEXT NOT NULL,
                ts_local TEXT NOT NULL,
                pv_kwh_p50 REAL,
                pv_kwh_p10 REAL,
                pv_kwh_p90 REAL,
                weather_code_model_id TEXT,
                weather_code INTEGER,
                PRIMARY KEY (run_id, ts_local)
            );
            CREATE INDEX IF NOT EXISTS idx_run_ensemble_hourly_run_ts ON run_ensemble_hourly (run_id, ts_local);

            CREATE TABLE IF NOT EXISTS provider_payloads (
                run_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                fetched_at_utc TEXT,
                endpoint TEXT,
                params_json TEXT,
                response_headers_json TEXT,
                response_json TEXT,
                http_status INTEGER,
                latency_ms INTEGER,
                PRIMARY KEY (run_id, model_id)
            );

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
                created_at_utc TEXT NOT NULL,
                pv_rmse_kwh REAL,
                pv_bias_kwh REAL,
                pv_daily_forecast_kwh REAL,
                pv_daily_actual_kwh REAL,
                pv_daily_error_kwh REAL,
                pv_hourly_points INTEGER,
                source TEXT,
                model_scores_json TEXT
            );
                        CREATE TABLE IF NOT EXISTS backtest_daily_scores (
                score_date TEXT NOT NULL,
                model_id TEXT NOT NULL,
                source TEXT NOT NULL,
                pv_forecast_kwh REAL,
                pv_actual_kwh REAL,
                pv_mae_kwh REAL,
                pv_rmse_kwh REAL,
                pv_bias_kwh REAL,
                pv_daily_error_kwh REAL,
                pv_hourly_points INTEGER,
                created_at_utc TEXT NOT NULL,
                PRIMARY KEY (score_date, model_id, source)
            );
            CREATE INDEX IF NOT EXISTS idx_backtest_daily_scores_date ON backtest_daily_scores (score_date);

            CREATE TABLE IF NOT EXISTS error_events (
                error_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                source TEXT NOT NULL,
                severity TEXT NOT NULL,
                error_type TEXT NOT NULL,
                "where" TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                context_json TEXT,
                fixed INTEGER NOT NULL DEFAULT 0,
                fixed_at_utc TEXT,
                dedupe_key TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_error_events_created_at ON error_events (created_at_utc);
            CREATE INDEX IF NOT EXISTS idx_error_events_fixed ON error_events (fixed, created_at_utc);
            CREATE INDEX IF NOT EXISTS idx_error_events_dedupe ON error_events (dedupe_key, created_at_utc);
            CREATE INDEX IF NOT EXISTS idx_error_events_type ON error_events (error_type, created_at_utc);
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
            ("soc_now_used", "REAL"),
            ("warnings_count", "INTEGER"),
            ("inputs_used_json", "TEXT"),
            ("weather_ensemble_json", "TEXT"),
            ("pv_week_ahead_json", "TEXT"),
            ("pv_p10_kwh", "REAL"),
            ("pv_p50_kwh", "REAL"),
            ("pv_p90_kwh", "REAL"),
            ("run_duration_ms", "INTEGER"),
            ("config_schema_version", "INTEGER"),
            ("mode", "TEXT"),
            ("requested_days", "INTEGER"),
            ("models_used_json", "TEXT"),
            ("ensemble_method", "TEXT"),
            ("weights_used_json", "TEXT"),
            ("config_snapshot_json", "TEXT"),
            ("input_snapshot_json", "TEXT"),
        ]:
            if col_name not in forecast_runs_existing_cols:
                conn.execute(f"ALTER TABLE forecast_runs ADD COLUMN {col_name} {col_type}")


        daily_scores_existing_cols = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(daily_scores)").fetchall()
        }
        for col_name, col_type in [
            ("pv_rmse_kwh", "REAL"),
            ("pv_bias_kwh", "REAL"),
            ("pv_daily_forecast_kwh", "REAL"),
            ("pv_daily_actual_kwh", "REAL"),
            ("pv_daily_error_kwh", "REAL"),
            ("pv_hourly_points", "INTEGER"),
            ("source", "TEXT"),
            ("model_scores_json", "TEXT"),
        ]:
            if col_name not in daily_scores_existing_cols:
                conn.execute(f"ALTER TABLE daily_scores ADD COLUMN {col_name} {col_type}")



        weather_existing_cols = {str(row["name"]) for row in conn.execute("PRAGMA table_info(weather_hourly_by_model)").fetchall()}
        for col_name, col_type in [
            ("dni_source", "TEXT"),
            ("dhi_source", "TEXT"),
            ("weather_code", "INTEGER"),
        ]:
            if col_name not in weather_existing_cols:
                conn.execute(f"ALTER TABLE weather_hourly_by_model ADD COLUMN {col_name} {col_type}")

        pv_model_existing_cols = {str(row["name"]) for row in conn.execute("PRAGMA table_info(pv_hourly_by_model)").fetchall()}
        for col_name, col_type in [
            ("dc_kw", "REAL"),
            ("ac_kw", "REAL"),
        ]:
            if col_name not in pv_model_existing_cols:
                conn.execute(f"ALTER TABLE pv_hourly_by_model ADD COLUMN {col_name} {col_type}")
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
        conn.execute("PRAGMA optimize;")


def compute_config_hash(config: dict) -> str:
    serialized = json.dumps(config or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _iso_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def insert_error_event(
    db_path: str,
    *,
    source: str,
    severity: str,
    error_type: str,
    where: str,
    title: str,
    body: str,
    context: dict | None = None,
    dedupe_key: str | None = None,
    dedupe_window_seconds: int = 300,
) -> str:
    created_at_utc = iso_utc_now()
    error_id = uuid.uuid4().hex
    body_trimmed = trim(str(body or ""), MAX_ERROR_BODY_CHARS)
    context_json = json.dumps(context, sort_keys=True, default=str) if isinstance(context, dict) else None

    with _connect(db_path) as conn:
        if dedupe_key:
            candidate = conn.execute(
                """
                SELECT error_id
                FROM error_events
                WHERE dedupe_key = ?
                  AND fixed = 0
                  AND strftime('%s', created_at_utc) >= strftime('%s', ?) - ?
                ORDER BY created_at_utc DESC
                LIMIT 1
                """,
                (dedupe_key, created_at_utc, int(max(0, dedupe_window_seconds))),
            ).fetchone()
            if candidate:
                return str(candidate["error_id"])

        conn.execute(
            """
            INSERT INTO error_events (
                error_id, created_at_utc, source, severity, error_type, "where", title, body,
                context_json, fixed, fixed_at_utc, dedupe_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)
            """,
            (error_id, created_at_utc, source, severity, error_type, where, title, body_trimmed, context_json, dedupe_key),
        )
    return error_id


def fetch_error_events(db_path: str, *, limit: int | None = None, include_fixed: bool = True) -> list[dict]:
    sql = (
        "SELECT error_id, created_at_utc, source, severity, error_type, \"where\", title, fixed "
        "FROM error_events "
    )
    params: list[Any] = []
    if not include_fixed:
        sql += "WHERE fixed = 0 "
    sql += "ORDER BY created_at_utc DESC"
    if limit is not None and int(limit) > 0:
        sql += " LIMIT ?"
        params.append(int(limit))

    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def fetch_error_event_by_id(db_path: str, error_id: str) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM error_events WHERE error_id = ?", (error_id,)).fetchone()
    return dict(row) if row else None


def set_error_fixed(db_path: str, *, error_id: str, fixed: bool) -> None:
    fixed_at_utc = iso_utc_now() if fixed else None
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE error_events SET fixed = ?, fixed_at_utc = ? WHERE error_id = ?",
            (1 if fixed else 0, fixed_at_utc, error_id),
        )


def delete_error_event(db_path: str, *, error_id: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM error_events WHERE error_id = ?", (error_id,))


def delete_all_error_events(db_path: str, *, only_fixed: bool = False) -> int:
    with _connect(db_path) as conn:
        if only_fixed:
            cursor = conn.execute("DELETE FROM error_events WHERE fixed = 1")
        else:
            cursor = conn.execute("DELETE FROM error_events")
    return int(cursor.rowcount or 0)


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
        out = float(value)
        return None if math.isnan(out) else out
    except (TypeError, ValueError):
        return None


def _trim_text(raw: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(raw) <= max_chars:
        return raw
    suffix = f"... [truncated {len(raw) - max_chars} chars]"
    keep = max(0, max_chars - len(suffix))
    return raw[:keep] + suffix




def _replace_nan_with_none(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {k: _replace_nan_with_none(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_replace_nan_with_none(v) for v in payload]
    if isinstance(payload, float) and math.isnan(payload):
        return None
    return payload

def _safe_json_dumps(payload: Any, *, max_chars: int | None = None) -> str | None:
    if isinstance(payload, str):
        serialized = payload
    else:
        try:
            serialized = json.dumps(_replace_nan_with_none(payload), sort_keys=True, default=str, allow_nan=False)
        except Exception:
            logger.exception(
                "db_sqlite json_serialize_failed payload_type=%s",
                type(payload).__name__,
            )
            return None
    if max_chars is None or len(serialized) <= max_chars:
        return serialized
    fallback = {
        "_truncated": True,
        "original_length": len(serialized),
        "preview": _trim_text(serialized, max_chars=max(0, max_chars - 128)),
    }
    try:
        return json.dumps(fallback, sort_keys=True)
    except Exception:
        logger.exception("db_sqlite json_truncation_fallback_serialize_failed")
        return None


def _normalize_provider_payload_rows(run_id: str, payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    raw = payload.get("provider_payloads_by_model")
    if not isinstance(raw, dict):
        return []
    rows: list[tuple[Any, ...]] = []
    for model_id, record in raw.items():
        if not isinstance(record, dict):
            continue
        rows.append((
            run_id,
            str(model_id),
            str(record.get("fetched_at_utc") or ""),
            str(record.get("endpoint") or ""),
            _safe_json_dumps(record.get("params")),
            _safe_json_dumps(record.get("response_headers")),
            _safe_json_dumps(record.get("response_json"), max_chars=MAX_PROVIDER_RESPONSE_CHARS),
            int(record.get("http_status")) if record.get("http_status") is not None else None,
            int(record.get("latency_ms")) if record.get("latency_ms") is not None else None,
        ))
    return rows


def _required_actual_hourly_columns() -> tuple[str, ...]:
    return (
        "ts_local",
        "pv_kwh",
        "load_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "soc_pct",
    )


def normalize_actual_hourly_row(row: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
    required = _required_actual_hourly_columns()
    row_keys = set(row.keys())
    required_set = set(required)
    if strict and row_keys != required_set:
        missing = sorted(required_set - row_keys)
        extra = sorted(row_keys - required_set)
        parts: list[str] = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        raise ValueError(f"Invalid actual row columns: {' '.join(parts)}")

    ts_local = _to_local_hour_str(row.get("ts_local"))
    if not ts_local:
        raise ValueError("Invalid ts_local value")

    return {
        "ts_local": ts_local,
        "pv_kwh": _safe_float(row.get("pv_kwh")),
        "load_kwh": _safe_float(row.get("load_kwh")),
        "grid_import_kwh": _safe_float(row.get("grid_import_kwh")),
        "grid_export_kwh": _safe_float(row.get("grid_export_kwh")),
        "soc_pct": _safe_float(row.get("soc_pct")),
    }


def insert_actual_hourly_rows(db_path: str, rows: list[dict[str, Any]], *, source: str = "manual_csv") -> int:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each actual row must be an object")
        normalized.append(normalize_actual_hourly_row(row, strict=True))

    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO actual_hourly (
                source,
                ts_local,
                pv_kwh,
                load_kwh,
                grid_import_kwh,
                grid_export_kwh,
                soc_pct,
                completeness
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    source,
                    row["ts_local"],
                    row["pv_kwh"],
                    row["load_kwh"],
                    row["grid_import_kwh"],
                    row["grid_export_kwh"],
                    row["soc_pct"],
                    None,
                )
                for row in normalized
            ],
        )
        conn.commit()
    return len(normalized)


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


def _normalize_weather_by_model(payload: dict) -> list[tuple[str, str, float | None, float | None, float | None, str, str, float | None, float | None, float | None, int | None]]:
    weather_by_model = payload.get("weather_by_model") if isinstance(payload.get("weather_by_model"), dict) else {}
    derived_by_model = payload.get("derived_irradiance_by_model") if isinstance(payload.get("derived_irradiance_by_model"), dict) else {}
    rows: list[tuple[str, str, float | None, float | None, float | None, str, str, float | None, float | None, float | None, int | None]] = []
    for model_id, frame_payload in weather_by_model.items():
        if not isinstance(model_id, str):
            continue
        model_df = _parse_df({"frame": frame_payload}, "frame")
        if model_df.empty:
            continue
        derived = bool(derived_by_model.get(model_id, False))
        source_label = "derived" if derived else "native"
        for ts in model_df.index:
            ts_local = _to_local_hour_str(ts)
            if not ts_local:
                continue
            rows.append(
                (
                    model_id,
                    ts_local,
                    _safe_float(model_df.at[ts, "ghi_wm2"]) if "ghi_wm2" in model_df.columns else None,
                    _safe_float(model_df.at[ts, "dni_wm2"]) if "dni_wm2" in model_df.columns else None,
                    _safe_float(model_df.at[ts, "dhi_wm2"]) if "dhi_wm2" in model_df.columns else None,
                    source_label,
                    source_label,
                    _safe_float(model_df.at[ts, "temp_air_c"]) if "temp_air_c" in model_df.columns else None,
                    _safe_float(model_df.at[ts, "wind_speed_ms"]) if "wind_speed_ms" in model_df.columns else None,
                    _safe_float(model_df.at[ts, "cloud_cover_pct"]) if "cloud_cover_pct" in model_df.columns else None,
                    int(model_df.at[ts, "weather_code"]) if "weather_code" in model_df.columns and _safe_float(model_df.at[ts, "weather_code"]) is not None else None,
                )
            )
    return rows


def _normalize_pv_by_model(payload: dict) -> list[tuple[str, str, float | None, float | None, float | None, float | None, float | None, float | None, float | None]]:
    pv_by_model = payload.get("pv_by_model") if isinstance(payload.get("pv_by_model"), dict) else {}
    rows: list[tuple[str, str, float | None, float | None, float | None, float | None, float | None, float | None, float | None]] = []
    for model_id, frame_payload in pv_by_model.items():
        if not isinstance(model_id, str):
            continue
        model_df = _parse_df({"frame": frame_payload}, "frame")
        if model_df.empty:
            continue
        for ts in model_df.index:
            ts_local = _to_local_hour_str(ts)
            if not ts_local:
                continue
            rows.append(
                (
                    model_id,
                    ts_local,
                    _safe_float(model_df.at[ts, "pv_east_kwh"]) if "pv_east_kwh" in model_df.columns else None,
                    _safe_float(model_df.at[ts, "pv_south_kwh"]) if "pv_south_kwh" in model_df.columns else None,
                    _safe_float(model_df.at[ts, "pv_total_kwh"]) if "pv_total_kwh" in model_df.columns else None,
                    _safe_float(model_df.at[ts, "pv_total_unclipped_kwh"]) if "pv_total_unclipped_kwh" in model_df.columns else None,
                    _safe_float(model_df.at[ts, "pv_clipped_kwh"]) if "pv_clipped_kwh" in model_df.columns else None,
                    _safe_float(model_df.at[ts, "pv_dc_available_kw"]) if "pv_dc_available_kw" in model_df.columns else None,
                    _safe_float(model_df.at[ts, "pv_total_kwh"]) if "pv_total_kwh" in model_df.columns else None,
                )
            )
    return rows


def _normalize_ensemble_hourly(payload: dict) -> list[tuple[str, float | None, float | None, float | None, str | None, int | None]]:
    pv_df = _parse_df(payload, "pv")
    if pv_df.empty:
        return []
    source_model = payload.get("tomorrow_weather_code_source_model_id")
    rows: list[tuple[str, float | None, float | None, float | None, str | None, int | None]] = []
    for ts in pv_df.index:
        ts_local = _to_local_hour_str(ts)
        if not ts_local:
            continue
        code = None
        if "weather_code" in pv_df.columns:
            wc = _safe_float(pv_df.at[ts, "weather_code"])
            code = int(wc) if wc is not None else None
        rows.append((
            ts_local,
            _safe_float(pv_df.at[ts, "pv_total_kwh"]) if "pv_total_kwh" in pv_df.columns else None,
            _safe_float(pv_df.at[ts, "pv_total_low_kwh"]) if "pv_total_low_kwh" in pv_df.columns else None,
            _safe_float(pv_df.at[ts, "pv_total_high_kwh"]) if "pv_total_high_kwh" in pv_df.columns else None,
            str(source_model) if source_model else None,
            code,
        ))
    return rows

def insert_forecast_run(db_path: str, payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    inputs_used = payload.get("inputs_used") if isinstance(payload.get("inputs_used"), dict) else {}
    if not inputs_used:
        soc_fallback = payload.get("soc_now_percent", payload.get("soc_at_22_percent"))
        inputs_used = {
            **inputs_used,
            "soc_now_percent": soc_fallback,
            "soc_at_22_percent": soc_fallback,
            "yesterday_consumption_kwh": payload.get("yesterday_consumption_kwh"),
        }
    weather_ensemble = payload.get("weather_ensemble") if isinstance(payload.get("weather_ensemble"), dict) else {}

    run_id = str(payload.get("run_id") or uuid.uuid4())
    target_date = str(payload.get("target_date") or "")
    if not target_date:
        raise ValueError("target_date is required")

    run_at_utc = str(payload.get("run_at_utc") or _iso_utc_now())
    timezone = str(payload.get("timezone") or payload.get("system_snapshot", {}).get("timezone") or DEFAULT_TIMEZONE)

    hourly_rows = _normalize_hourly(payload)
    weather_rows = _normalize_weather_by_model(payload)
    pv_model_rows = _normalize_pv_by_model(payload)
    ensemble_rows = _normalize_ensemble_hourly(payload)
    provider_payload_rows = _normalize_provider_payload_rows(run_id, payload)

    cutoff_soc = _safe_float(metrics.get("cutoff_soc"))
    if cutoff_soc is not None and cutoff_soc <= 1.0:
        cutoff_soc *= 100.0
    elif cutoff_soc is None:
        cutoff_soc = _safe_float(payload.get("cutoff_soc"))

    pv_forecast_kwh = _safe_float(metrics.get("pv_forecast_kwh"))
    if pv_forecast_kwh is None:
        pv_forecast_kwh = _safe_float(payload.get("pv_forecast_kwh"))

    cons_forecast_kwh = _safe_float(metrics.get("cons_forecast_kwh"))
    if cons_forecast_kwh is None:
        cons_forecast_kwh = _safe_float(payload.get("cons_forecast_kwh"))

    config_obj = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    config_hash = payload.get("config_hash") or compute_config_hash(config_obj)
    config_json = payload.get("config_json")
    if not isinstance(config_json, str):
        config_json = json.dumps(_replace_nan_with_none(config_obj), sort_keys=True, allow_nan=False)

    warnings_json = json.dumps(_replace_nan_with_none(payload.get("warnings", [])), allow_nan=False)

    pv_week_ahead = payload.get("pv_week_ahead") if isinstance(payload.get("pv_week_ahead"), list) else []
    pv_week_ahead_json = json.dumps(_replace_nan_with_none(pv_week_ahead), allow_nan=False)
    pv_quality = payload.get("pv_quality")
    models_used = payload.get("tomorrow_models_used") if isinstance(payload.get("tomorrow_models_used"), list) else []
    if not models_used and isinstance(weather_ensemble, dict):
        maybe_selected = weather_ensemble.get("selected_models")
        if isinstance(maybe_selected, list):
            models_used = [str(model_id) for model_id in maybe_selected]
    weights_used = (weather_ensemble.get("weights_used") if isinstance(weather_ensemble, dict) else None)
    requested_days_raw = payload.get("requested_days")
    if requested_days_raw is None and isinstance(weather_ensemble, dict):
        requested_days_raw = weather_ensemble.get("requested_days")
    try:
        requested_days = max(1, int(requested_days_raw))
    except (TypeError, ValueError):
        requested_days = 1
    pv_quality_text = json.dumps(_replace_nan_with_none(pv_quality), allow_nan=False) if isinstance(pv_quality, dict) else None
    config_schema_version_raw = payload.get("config_schema_version")
    try:
        config_schema_version = int(config_schema_version_raw)
    except (TypeError, ValueError):
        config_schema_version = CURRENT_CONFIG_SCHEMA_VERSION

    row_data = {
        "mode": payload.get("forecast_mode_effective"),
        "requested_days": requested_days,
        "models_used_json": json.dumps(_replace_nan_with_none(models_used), sort_keys=True, allow_nan=False),
        "ensemble_method": weather_ensemble.get("ensemble_method") if isinstance(weather_ensemble, dict) else None,
        "weights_used_json": json.dumps(_replace_nan_with_none(weights_used), sort_keys=True, allow_nan=False),
        "config_snapshot_json": json.dumps(_replace_nan_with_none(config_obj), sort_keys=True, allow_nan=False),
        "input_snapshot_json": json.dumps(_replace_nan_with_none(inputs_used), sort_keys=True, allow_nan=False),
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
        "inputs_used_json": json.dumps(_replace_nan_with_none(inputs_used), sort_keys=True, allow_nan=False),
        "weather_ensemble_json": json.dumps(_replace_nan_with_none(weather_ensemble), sort_keys=True, allow_nan=False),
        "pv_week_ahead_json": pv_week_ahead_json,
        "pv_p10_kwh": _safe_float((payload.get("pv_totals_kwh") or {}).get("p10")),
        "pv_p50_kwh": _safe_float((payload.get("pv_totals_kwh") or {}).get("p50")),
        "pv_p90_kwh": _safe_float((payload.get("pv_totals_kwh") or {}).get("p90")),
        "run_duration_ms": int(payload.get("run_duration_ms")) if payload.get("run_duration_ms") is not None else None,
        "config_schema_version": config_schema_version,
        "soc_at_22_used": _safe_float(inputs_used.get("soc_at_22_percent")),
        "soc_now_used": _safe_float(inputs_used.get("soc_now_percent")),
        "yesterday_kwh_used": _safe_float(inputs_used.get("yesterday_consumption_kwh")),
        "planner_version": payload.get("planner_version"),
        "config_hash": config_hash,
        "config_json": config_json,
        "warnings_json": warnings_json,
        "pv_quality": pv_quality_text,
        "created_at_utc": payload.get("created_at_utc") or run_at_utc,
    }

    try:
        with _connect(db_path) as conn:
            conn.execute("BEGIN")
            conn.execute(
                """
                INSERT OR REPLACE INTO forecast_runs (
                    run_id, target_date, run_at_utc, run_type, status, timezone,
                    charge_kw, cutoff_soc, pv_forecast_kwh, cons_forecast_kwh, warnings_count,
                    inputs_used_json, weather_ensemble_json, pv_week_ahead_json, pv_p10_kwh, pv_p50_kwh, pv_p90_kwh,
                    run_duration_ms, config_schema_version,
                    soc_at_22_used, soc_now_used, yesterday_kwh_used, planner_version,
                    config_hash, config_json, warnings_json, pv_quality, created_at_utc,
                    mode, requested_days, models_used_json, ensemble_method, weights_used_json,
                    config_snapshot_json, input_snapshot_json
                ) VALUES (
                    :run_id, :target_date, :run_at_utc, :run_type, :status, :timezone,
                    :charge_kw, :cutoff_soc, :pv_forecast_kwh, :cons_forecast_kwh, :warnings_count,
                    :inputs_used_json, :weather_ensemble_json, :pv_week_ahead_json, :pv_p10_kwh, :pv_p50_kwh, :pv_p90_kwh,
                    :run_duration_ms, :config_schema_version,
                    :soc_at_22_used, :soc_now_used, :yesterday_kwh_used, :planner_version,
                    :config_hash, :config_json, :warnings_json, :pv_quality, :created_at_utc,
                    :mode, :requested_days, :models_used_json, :ensemble_method, :weights_used_json,
                    :config_snapshot_json, :input_snapshot_json
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

            conn.execute("DELETE FROM weather_hourly_by_model WHERE run_id = ?", (run_id,))
            if weather_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO weather_hourly_by_model (
                        run_id, model_id, ts_local, ghi, dni, dhi, dni_source, dhi_source, temp_c, wind_ms, cloud_pct, weather_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            model_id,
                            ts_local,
                            ghi,
                            dni,
                            dhi,
                            dni_source,
                            dhi_source,
                            temp_c,
                            wind_ms,
                            cloud_pct,
                            weather_code,
                        )
                        for model_id, ts_local, ghi, dni, dhi, dni_source, dhi_source, temp_c, wind_ms, cloud_pct, weather_code in weather_rows
                    ],
                )

            conn.execute("DELETE FROM pv_hourly_by_model WHERE run_id = ?", (run_id,))
            if pv_model_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO pv_hourly_by_model (
                        run_id, model_id, ts_local, pv_east_kwh, pv_south_kwh, pv_total_kwh, pv_unclipped_kwh, pv_clipped_kwh, dc_kw, ac_kw
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            model_id,
                            ts_local,
                            pv_east_kwh,
                            pv_south_kwh,
                            pv_total_kwh,
                            pv_unclipped_kwh,
                            pv_clipped_kwh,
                            dc_kw,
                            ac_kw,
                        )
                        for model_id, ts_local, pv_east_kwh, pv_south_kwh, pv_total_kwh, pv_unclipped_kwh, pv_clipped_kwh, dc_kw, ac_kw in pv_model_rows
                    ],
                )

            conn.execute("DELETE FROM run_ensemble_hourly WHERE run_id = ?", (run_id,))
            if ensemble_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO run_ensemble_hourly (
                        run_id, ts_local, pv_kwh_p50, pv_kwh_p10, pv_kwh_p90, weather_code_model_id, weather_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (run_id, ts_local, pv_kwh_p50, pv_kwh_p10, pv_kwh_p90, weather_code_model_id, weather_code)
                        for ts_local, pv_kwh_p50, pv_kwh_p10, pv_kwh_p90, weather_code_model_id, weather_code in ensemble_rows
                    ],
                )

            conn.execute("DELETE FROM provider_payloads WHERE run_id = ?", (run_id,))
            if provider_payload_rows:
                try:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO provider_payloads (
                            run_id, model_id, fetched_at_utc, endpoint, params_json,
                            response_headers_json, response_json, http_status, latency_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        provider_payload_rows,
                    )
                except Exception:
                    logger.exception(
                        "db_sqlite provider_payloads_persist_failed run_id=%s rows=%d",
                        run_id,
                        len(provider_payload_rows),
                    )
            conn.commit()
    except Exception:
        logger.exception(
            "db_sqlite insert_forecast_run_failed run_id=%s target_date=%s status=%s",
            run_id,
            target_date,
            payload.get("status"),
        )
        raise


def fetch_effective_daily_kwh(
    db_path: str,
    *,
    lookback_runs: int = 14,
    prefer_same_day_type: bool = True,
) -> tuple[float | None, dict]:
    """Return effective daily kWh from recent successful run history."""
    limit = max(1, int(lookback_runs))
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT run_at_utc, yesterday_kwh_used
                FROM forecast_runs
                WHERE status = 'ok' AND yesterday_kwh_used IS NOT NULL
                ORDER BY run_at_utc DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.OperationalError:
        logger.warning(
            "db_sqlite effective_daily_kwh_query_failed db_path=%s lookback_runs=%d",
            db_path,
            limit,
            exc_info=True,
        )
        return None, {"n_samples": 0, "method": "none", "day_type": "mixed"}

    meta = {"n_samples": 0, "method": "none", "day_type": "mixed"}
    if not rows:
        return None, meta

    tz = ZoneInfo(DEFAULT_TIMEZONE)
    today_local = dt.datetime.now(tz).date()
    target_is_weekend = today_local.weekday() >= 5

    same_type_values: list[float] = []
    mixed_values: list[float] = []
    for row in rows:
        try:
            val = float(row["yesterday_kwh_used"])
        except (TypeError, ValueError):
            continue
        mixed_values.append(val)
        run_at_raw = row["run_at_utc"]
        try:
            run_at = pd.to_datetime(run_at_raw, utc=True).tz_convert(tz)
        except Exception:
            continue
        row_is_weekend = run_at.weekday() >= 5
        if row_is_weekend == target_is_weekend:
            same_type_values.append(val)

    values = same_type_values if (prefer_same_day_type and len(same_type_values) >= 2) else mixed_values
    if not values:
        return None, meta

    meta["n_samples"] = len(values)
    if prefer_same_day_type and len(same_type_values) >= 2:
        meta["day_type"] = "weekend" if target_is_weekend else "weekday"
    else:
        meta["day_type"] = "mixed"

    if len(values) >= 5:
        meta["method"] = "median"
        return float(pd.Series(values).median()), meta
    if len(values) >= 2:
        meta["method"] = "mean"
        return float(pd.Series(values).mean()), meta
    return None, meta


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
                    "pv_forecast_kwh": _safe_float(row["pv_forecast_kwh"]),
                    "cons_forecast_kwh": _safe_float(row["cons_forecast_kwh"]),
                },
                "run_at": row["run_at_utc"],
                "run_type": row["run_type"] or "manual",
            }
        )
    items.reverse()
    return items


def _summary_from_row(row: sqlite3.Row) -> dict:
    models_summary: dict[str, Any] | None = None
    models_ok_count = 0
    models_failed_count = 0
    primary_model_id: str | None = None
    raw_weather_ensemble = row["weather_ensemble_json"] if "weather_ensemble_json" in row.keys() else None
    if raw_weather_ensemble:
        try:
            weather_ensemble = json.loads(raw_weather_ensemble)
        except (TypeError, ValueError):
            weather_ensemble = {}
        if isinstance(weather_ensemble, dict):
            selected_models = weather_ensemble.get("selected_models") or []
            if not isinstance(selected_models, list):
                selected_models = []
            failed_models = weather_ensemble.get("failed_models") or []
            if not isinstance(failed_models, list):
                failed_models = []
            failed_set = {str(model_id) for model_id in failed_models}
            models_failed_count = len(failed_set)
            models_ok_count = max(0, len(selected_models) - models_failed_count)
            primary_model_id = weather_ensemble.get("primary_model_id") or weather_ensemble.get("weather_primary_model_id")
            models_summary = {
                "selected_models": selected_models,
                "weights_used": weather_ensemble.get("weights_used") or {},
                "failed_models": failed_models,
            }

    pv_quality: dict[str, Any] = {}
    raw_pv_quality = row["pv_quality"] if "pv_quality" in row.keys() else None
    if isinstance(raw_pv_quality, str) and raw_pv_quality:
        try:
            parsed_pv_quality = json.loads(raw_pv_quality)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed_pv_quality = {}
        if isinstance(parsed_pv_quality, dict):
            pv_quality = {
                "label": parsed_pv_quality.get("label"),
                "score": parsed_pv_quality.get("score"),
            }
            pv_quality = {k: v for k, v in pv_quality.items() if v is not None}

    return {
        "run_id": row["run_id"],
        "target_date": row["target_date"],
        "metrics": {
            "charge_kw": float(row["charge_kw"] or 0.0),
            "cutoff_soc": float(row["cutoff_soc"] or 0.0) / 100.0,
            "pv_forecast_kwh": _safe_float(row["pv_forecast_kwh"]),
            "cons_forecast_kwh": _safe_float(row["cons_forecast_kwh"]),
        },
        "status": row["status"],
        "warnings_count": int(row["warnings_count"] or 0),
        "run_duration_ms": row["run_duration_ms"],
        "models_ok_count": models_ok_count,
        "models_failed_count": models_failed_count,
        "primary_model_id": primary_model_id,
        "pv_p10_kwh": _safe_float(row["pv_p10_kwh"]),
        "pv_p50_kwh": _safe_float(row["pv_p50_kwh"]),
        "pv_p90_kwh": _safe_float(row["pv_p90_kwh"]),
        "pv_quality": pv_quality,
        "models_summary": models_summary,
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
                    status,
                    warnings_count,
                    run_duration_ms,
                    weather_ensemble_json,
                    pv_quality,
                    pv_p10_kwh,
                    pv_p50_kwh,
                    pv_p90_kwh,
                    ROW_NUMBER() OVER (
                        PARTITION BY target_date
                        ORDER BY run_at_utc DESC, COALESCE(created_at_utc, run_at_utc) DESC, run_id DESC
                    ) AS rn
                FROM forecast_runs
            )
            SELECT
                run_id,
                target_date,
                run_at_utc,
                run_type,
                charge_kw,
                cutoff_soc,
                pv_forecast_kwh,
                cons_forecast_kwh,
                status,
                warnings_count,
                run_duration_ms,
                weather_ensemble_json,
                pv_quality,
                pv_p10_kwh,
                pv_p50_kwh,
                pv_p90_kwh
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
            SELECT
                run_id,
                target_date,
                run_at_utc,
                run_type,
                charge_kw,
                cutoff_soc,
                pv_forecast_kwh,
                cons_forecast_kwh,
                status,
                warnings_count,
                run_duration_ms,
                weather_ensemble_json,
                pv_quality,
                pv_p10_kwh,
                pv_p50_kwh,
                pv_p90_kwh
            FROM forecast_runs
            {date_filter_sql}
            ORDER BY target_date ASC, run_at_utc ASC, COALESCE(created_at_utc, run_at_utc) ASC, run_id ASC
            """,
            params,
        ).fetchall()
    return [_summary_from_row(row) for row in rows]


def _decode_json_payload(raw: Any, default: Any, *, field_name: str = "unknown") -> Any:
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("db_sqlite decode_json_payload_failed field=%s", field_name, exc_info=True)
            return default
    return default


def _full_run_shared_payload(row: sqlite3.Row) -> dict[str, Any]:
    warnings = _decode_json_payload(row["warnings_json"], [], field_name="warnings_json")
    inputs_used = _decode_json_payload(row["inputs_used_json"], {}, field_name="inputs_used_json")
    config_json = _decode_json_payload(row["config_json"], {}, field_name="config_json")
    weather_ensemble = _decode_json_payload(row["weather_ensemble_json"], {}, field_name="weather_ensemble_json")
    return {
        "run_id": row["run_id"],
        "target_date": row["target_date"],
        "run_at_utc": row["run_at_utc"],
        "run_type": row["run_type"] or "manual",
        "status": row["status"],
        "warnings_count": int(row["warnings_count"] or len(warnings)),
        "warnings": warnings,
        "inputs_used": inputs_used,
        "config_hash": row["config_hash"] or "",
        "config_json": config_json,
        "weather_ensemble": weather_ensemble,
        "pv_totals_kwh": {
            "p10": _safe_float(row["pv_p10_kwh"]),
            "p50": _safe_float(row["pv_p50_kwh"]),
            "p90": _safe_float(row["pv_p90_kwh"]),
        },
        "metrics": {
            "charge_kw": float(row["charge_kw"] or 0.0),
            "cutoff_soc": float(row["cutoff_soc"] or 0.0) / 100.0,
            "pv_forecast_kwh": _safe_float(row["pv_forecast_kwh"]),
            "cons_forecast_kwh": _safe_float(row["cons_forecast_kwh"]),
        },
        "run_at": row["run_at_utc"],
    }


def _materialize_full_run_hourly_sections(hourly_rows: list[sqlite3.Row]) -> dict[str, Any]:
    hourly = _hourly_rows_to_frame(hourly_rows)
    if hourly.empty:
        return {}

    idx = _hourly_materialization_index(hourly)
    materialized = _hourly_numeric_materialization_frame(hourly)
    pv_df = _build_hourly_pv_section(materialized, idx)
    flows_df = _build_hourly_flows_section(materialized, idx)
    soc_df = _build_hourly_soc_section(materialized, idx)
    return {
        "pv": _to_split_orient_payload(pv_df),
        "flows": _to_split_orient_payload(flows_df),
        "soc": _to_split_orient_payload(soc_df),
    }


def _hourly_rows_to_frame(hourly_rows: list[sqlite3.Row]) -> pd.DataFrame:
    if not hourly_rows:
        return pd.DataFrame()
    first_row = hourly_rows[0]
    if hasattr(first_row, "keys"):
        return pd.DataFrame.from_records(hourly_rows, columns=list(first_row.keys()))
    return pd.DataFrame.from_records(hourly_rows)


def _hourly_materialization_index(hourly: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(hourly["ts_local"], errors="coerce")


def _hourly_numeric_materialization_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    pv_total = pd.to_numeric(hourly["pv_kwh"], errors="coerce")
    soc_pct = pd.to_numeric(hourly["soc_pct"], errors="coerce").fillna(0.0)
    materialized = pd.DataFrame(
        {
            "pv_total_kwh": pv_total,
            "pv_total_unclipped_kwh": pd.to_numeric(hourly.get("pv_total_unclipped_kwh"), errors="coerce").fillna(pv_total),
            "pv_east_kwh": pd.to_numeric(hourly.get("pv_east_kwh"), errors="coerce"),
            "pv_south_kwh": pd.to_numeric(hourly.get("pv_south_kwh"), errors="coerce").fillna(pv_total),
            "pv_clipped_kwh": pd.to_numeric(hourly.get("pv_clipped_kwh"), errors="coerce"),
            "load_kwh": pd.to_numeric(hourly["load_kwh"], errors="coerce").fillna(0.0),
            "grid_import_kwh": pd.to_numeric(hourly["grid_import_kwh"], errors="coerce").fillna(0.0),
            "grid_export_kwh": pd.to_numeric(hourly["grid_export_kwh"], errors="coerce").fillna(0.0),
            "batt_charge_kwh": pd.to_numeric(hourly["batt_charge_kwh"], errors="coerce").fillna(0.0),
            "batt_discharge_kwh": pd.to_numeric(hourly["batt_discharge_kwh"], errors="coerce").fillna(0.0),
            "soc_end_pct": soc_pct,
            "value": soc_pct / 100.0,
        }
    )
    return materialized


def _build_hourly_pv_section(materialized: pd.DataFrame, idx: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        materialized.loc[
            :,
            [
                "pv_total_kwh",
                "pv_total_unclipped_kwh",
                "pv_east_kwh",
                "pv_south_kwh",
                "pv_clipped_kwh",
                "load_kwh",
            ],
        ],
        index=idx,
    )


def _build_hourly_flows_section(materialized: pd.DataFrame, idx: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        materialized.loc[
            :,
            ["grid_import_kwh", "grid_export_kwh", "batt_charge_kwh", "batt_discharge_kwh", "soc_end_pct"],
        ],
        index=idx,
    )


def _build_hourly_soc_section(materialized: pd.DataFrame, idx: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(materialized.loc[:, ["value"]], index=idx)


def _to_split_orient_payload(frame: pd.DataFrame) -> dict[str, Any]:
    return json.loads(frame.to_json(date_format="iso", orient="split"))


def _fetch_full_run_row(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT run_id, target_date, run_at_utc, run_type, timezone,
               charge_kw, cutoff_soc, pv_forecast_kwh, cons_forecast_kwh,
               status, warnings_count, warnings_json,
               inputs_used_json, weather_ensemble_json,
               pv_p10_kwh, pv_p50_kwh, pv_p90_kwh,
               config_hash, config_json
        FROM forecast_runs
        WHERE run_id = ?
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()


def _fetch_full_run_hourly_rows(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT ts_local, pv_kwh, pv_total_unclipped_kwh, pv_east_kwh, pv_south_kwh, pv_clipped_kwh,
               load_kwh, grid_import_kwh, grid_export_kwh,
               batt_charge_kwh, batt_discharge_kwh, soc_pct
        FROM forecast_hourly
        WHERE run_id = ?
        ORDER BY ts_local ASC
        """,
        (run_id,),
    ).fetchall()


def _fetch_latest_run_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        """
        SELECT run_id
        FROM forecast_runs
        ORDER BY run_at_utc DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    return str(row["run_id"])


def _build_full_run_payload(row: sqlite3.Row, hourly_rows: list[sqlite3.Row]) -> dict:
    payload = _full_run_shared_payload(row)
    payload.update(_materialize_full_run_hourly_sections(hourly_rows))
    return payload


def fetch_full_run_by_id(db_path: str, run_id: str) -> dict | None:
    try:
        with _connect(db_path) as conn:
            row = _fetch_full_run_row(conn, run_id)
            if row is None:
                return None
            hourly_rows = _fetch_full_run_hourly_rows(conn, row["run_id"])
        return _build_full_run_payload(row, hourly_rows)
    except Exception:
        logger.exception("db_sqlite fetch_full_run_by_id_failed run_id=%s", run_id)
        raise


def fetch_latest_full_run(db_path: str) -> dict | None:
    try:
        with _connect(db_path) as conn:
            run_id = _fetch_latest_run_id(conn)
    except Exception:
        logger.exception("db_sqlite fetch_latest_full_run_failed stage=lookup_latest_run_id")
        raise
    if run_id is None:
        return None
    return fetch_full_run_by_id(db_path, run_id)


def fetch_latest_run_id_for_date(db_path: str, target_date: str) -> str | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT run_id
            FROM forecast_runs
            WHERE target_date = ?
            ORDER BY run_at_utc DESC
            LIMIT 1
            """,
            (target_date,),
        ).fetchone()
    if row is None:
        return None
    return str(row["run_id"])


def fetch_forecast_pv_hourly(db_path: str, run_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT ts_local, pv_kwh
            FROM forecast_hourly
            WHERE run_id = ?
            ORDER BY ts_local ASC
            """,
            (run_id,),
        ).fetchall()
    return [{"ts_local": str(r["ts_local"]), "pv_kwh": _safe_float(r["pv_kwh"])} for r in rows]


def fetch_forecast_pv_hourly_by_model(db_path: str, run_id: str) -> dict[str, list[dict[str, Any]]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT model_id, ts_local, pv_total_kwh
            FROM pv_hourly_by_model
            WHERE run_id = ?
            ORDER BY model_id ASC, ts_local ASC
            """,
            (run_id,),
        ).fetchall()

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        model_id = str(row["model_id"])
        by_model.setdefault(model_id, []).append(
            {
                "ts_local": str(row["ts_local"]),
                "pv_kwh": _safe_float(row["pv_total_kwh"]),
            }
        )
    return by_model


def fetch_actual_pv_hourly_for_date(db_path: str, score_date: str, source: str = "manual_csv") -> list[dict[str, Any]]:
    prefix = f"{score_date}%"
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT ts_local, pv_kwh
            FROM actual_hourly
            WHERE source = ? AND ts_local LIKE ?
            ORDER BY ts_local ASC
            """,
            (source, prefix),
        ).fetchall()
    return [{"ts_local": str(r["ts_local"]), "pv_kwh": _safe_float(r["pv_kwh"])} for r in rows]


def upsert_daily_score(db_path: str, payload: dict[str, Any]) -> None:
    model_scores_json = payload.get("model_scores")
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_scores (
                run_id,
                score_date,
                pv_mae_kwh,
                pv_mape,
                load_mae_kwh,
                load_mape,
                soc_mae_pct,
                import_error_kwh,
                export_error_kwh,
                created_at_utc,
                pv_rmse_kwh,
                pv_bias_kwh,
                pv_daily_forecast_kwh,
                pv_daily_actual_kwh,
                pv_daily_error_kwh,
                pv_hourly_points,
                source,
                model_scores_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("run_id"),
                payload.get("score_date"),
                payload.get("pv_mae_kwh"),
                None,
                None,
                None,
                None,
                None,
                None,
                payload.get("created_at_utc") or _iso_utc_now(),
                payload.get("pv_rmse_kwh"),
                payload.get("pv_bias_kwh"),
                payload.get("pv_daily_forecast_kwh"),
                payload.get("pv_daily_actual_kwh"),
                payload.get("pv_daily_error_kwh"),
                payload.get("pv_hourly_points"),
                payload.get("source") or "manual_csv",
                json.dumps(model_scores_json or {}, sort_keys=True),
            ),
        )
        conn.commit()




def upsert_backtest_daily_score(db_path: str, payload: dict[str, Any]) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO backtest_daily_scores (
                score_date, model_id, source,
                pv_forecast_kwh, pv_actual_kwh,
                pv_mae_kwh, pv_rmse_kwh, pv_bias_kwh, pv_daily_error_kwh,
                pv_hourly_points, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("score_date"),
                payload.get("model_id"),
                payload.get("source") or "backtest",
                payload.get("pv_forecast_kwh"),
                payload.get("pv_actual_kwh"),
                payload.get("pv_mae_kwh"),
                payload.get("pv_rmse_kwh"),
                payload.get("pv_bias_kwh"),
                payload.get("pv_daily_error_kwh"),
                payload.get("pv_hourly_points"),
                payload.get("created_at_utc") or _iso_utc_now(),
            ),
        )
        conn.commit()


def fetch_backtest_daily_scores(
    db_path: str,
    *,
    start_date: str,
    end_date: str,
    source: str = "backtest",
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT score_date, model_id, source,
                   pv_forecast_kwh, pv_actual_kwh,
                   pv_mae_kwh, pv_rmse_kwh, pv_bias_kwh, pv_daily_error_kwh,
                   pv_hourly_points, created_at_utc
            FROM backtest_daily_scores
            WHERE score_date >= ? AND score_date <= ? AND source = ?
            ORDER BY score_date ASC, model_id ASC
            """,
            (start_date, end_date, source),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_daily_score_for_date(db_path: str, score_date: str, source: str = "manual_csv") -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM daily_scores
            WHERE score_date = ? AND source = ?
            ORDER BY created_at_utc DESC
            LIMIT 1
            """,
            (score_date, source),
        ).fetchone()
    return dict(row) if row is not None else None

def fetch_recent_model_mae_scores(
    db_path: str,
    *,
    lookback_days: int = 30,
    source: str | None = None,
) -> dict[str, dict[str, float]]:
    lookback = max(int(lookback_days), 1)
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=lookback - 1)

    query = """
        SELECT score_date, model_scores_json
        FROM daily_scores
        WHERE score_date >= ? AND score_date <= ?
    """
    params: list[Any] = [start_date.isoformat(), end_date.isoformat()]
    if source:
        query += " AND source = ?"
        params.append(source)

    with _connect(db_path) as conn:
        rows = conn.execute(query, tuple(params)).fetchall()

    mae_values: dict[str, list[float]] = {}
    rmse_values: dict[str, list[float]] = {}
    day_values: dict[str, set[str]] = {}

    for row in rows:
        score_date = str(row["score_date"])
        raw = row["model_scores_json"]
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        for model_id, metrics in parsed.items():
            if not isinstance(metrics, dict):
                continue
            mae = _safe_float(metrics.get("pv_mae_kwh"))
            rmse = _safe_float(metrics.get("pv_rmse_kwh"))
            if mae is not None and mae >= 0.0:
                mae_values.setdefault(str(model_id), []).append(float(mae))
                day_values.setdefault(str(model_id), set()).add(score_date)
            if rmse is not None and rmse >= 0.0:
                rmse_values.setdefault(str(model_id), []).append(float(rmse))

    result: dict[str, dict[str, float]] = {}
    for model_id in sorted(set(mae_values) | set(rmse_values)):
        maes = mae_values.get(model_id, [])
        rmses = rmse_values.get(model_id, [])
        result[model_id] = {
            "pv_mae_kwh": float(sum(maes) / len(maes)) if maes else float("nan"),
            "pv_rmse_kwh": float(sum(rmses) / len(rmses)) if rmses else float("nan"),
            "days": float(len(day_values.get(model_id, set()))),
        }
    return result
