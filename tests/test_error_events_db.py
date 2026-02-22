from pathlib import Path

from db_sqlite import (
    delete_all_error_events,
    delete_error_event,
    fetch_error_event_by_id,
    fetch_error_events,
    init_db,
    insert_error_event,
    set_error_fixed,
)


def test_init_db_creates_error_events(tmp_path: Path) -> None:
    db_path = tmp_path / "planner.sqlite"
    init_db(str(db_path))
    items = fetch_error_events(str(db_path), limit=10, include_fixed=True)
    assert items == []


def test_insert_and_fetch_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "planner.sqlite"
    init_db(str(db_path))
    error_id = insert_error_event(
        str(db_path),
        source="frontend",
        severity="error",
        error_type="exception",
        where="app.py:test",
        title="Test error",
        body="Something happened",
        context={"x": 1},
    )
    rows = fetch_error_events(str(db_path), include_fixed=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["error_id"] == error_id
    assert row["source"] == "frontend"
    assert row["error_type"] == "exception"
    detail = fetch_error_event_by_id(str(db_path), error_id)
    assert detail is not None
    assert detail["body"] == "Something happened"


def test_dedupe_reuses_error_id(tmp_path: Path) -> None:
    db_path = tmp_path / "planner.sqlite"
    init_db(str(db_path))
    args = dict(
        source="backend",
        severity="error",
        error_type="exception",
        where="backend_api:/v1/x",
        title="Same",
        body="Same body",
        dedupe_key="abc",
    )
    first = insert_error_event(str(db_path), **args)
    second = insert_error_event(str(db_path), **args)
    assert first == second


def test_fixed_toggle_sets_and_clears_fixed_at(tmp_path: Path) -> None:
    db_path = tmp_path / "planner.sqlite"
    init_db(str(db_path))
    error_id = insert_error_event(
        str(db_path),
        source="backend",
        severity="error",
        error_type="exception",
        where="backend_api:test",
        title="Fix me",
        body="Body",
    )
    set_error_fixed(str(db_path), error_id=error_id, fixed=True)
    detail = fetch_error_event_by_id(str(db_path), error_id)
    assert detail is not None and detail["fixed"] == 1 and detail["fixed_at_utc"]
    set_error_fixed(str(db_path), error_id=error_id, fixed=False)
    detail2 = fetch_error_event_by_id(str(db_path), error_id)
    assert detail2 is not None and detail2["fixed"] == 0 and detail2["fixed_at_utc"] is None


def test_delete_single_and_all(tmp_path: Path) -> None:
    db_path = tmp_path / "planner.sqlite"
    init_db(str(db_path))
    a = insert_error_event(str(db_path), source="frontend", severity="error", error_type="exception", where="a", title="a", body="a")
    b = insert_error_event(str(db_path), source="frontend", severity="error", error_type="exception", where="b", title="b", body="b")
    set_error_fixed(str(db_path), error_id=b, fixed=True)
    delete_error_event(str(db_path), error_id=a)
    rows = fetch_error_events(str(db_path), include_fixed=True)
    assert len(rows) == 1 and rows[0]["error_id"] == b
    deleted_fixed = delete_all_error_events(str(db_path), only_fixed=True)
    assert deleted_fixed == 1
    c = insert_error_event(str(db_path), source="frontend", severity="error", error_type="exception", where="c", title="c", body="c")
    deleted_all = delete_all_error_events(str(db_path), only_fixed=False)
    assert deleted_all == 1
    assert fetch_error_event_by_id(str(db_path), c) is None
