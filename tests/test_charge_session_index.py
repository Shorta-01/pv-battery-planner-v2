import planner_core as core


def test_charge_session_index_from_window_exact_hours():
    start = core.pd.Timestamp("2026-02-23 22:00", tz=core.TIMEZONE)
    end = core.pd.Timestamp("2026-02-24 07:00", tz=core.TIMEZONE)

    idx = core.get_charge_session_index_from_window(start, end)

    assert len(idx) == 9
    assert idx[0] == start
    assert idx[-1] == core.pd.Timestamp("2026-02-24 06:00", tz=core.TIMEZONE)


def test_charge_session_index_from_window_counts_only_full_hours():
    start = core.pd.Timestamp("2026-02-23 23:30", tz=core.TIMEZONE)
    end = core.pd.Timestamp("2026-02-24 06:15", tz=core.TIMEZONE)

    idx = core.get_charge_session_index_from_window(start, end)

    expected = core.pd.date_range(
        core.pd.Timestamp("2026-02-24 00:00", tz=core.TIMEZONE),
        core.pd.Timestamp("2026-02-24 06:00", tz=core.TIMEZONE),
        freq="h",
        inclusive="left",
    )
    assert idx.equals(expected)
    assert len(idx) == 6
