from __future__ import annotations

import re
from typing import Optional

_HHMM_RE = re.compile(r"^(\d{2}):(\d{2})$")


def parse_hhmm(s: str, *, allow_24_end: bool = False) -> int:
    match = _HHMM_RE.fullmatch(s)
    if not match:
        raise ValueError(f"Invalid time format: {s!r}. Expected HH:MM.")

    hour = int(match.group(1))
    minute = int(match.group(2))

    if hour == 24 and minute == 0:
        if allow_24_end:
            return 1440
        raise ValueError("24:00 is only valid for end times.")

    if hour < 0 or hour > 23:
        raise ValueError(f"Hour out of range in {s!r}.")
    if minute < 0 or minute > 59:
        raise ValueError(f"Minute out of range in {s!r}.")

    return hour * 60 + minute


def format_hhmm(minutes: int) -> str:
    if minutes < 0 or minutes > 1440:
        raise ValueError("Minutes must be in [0, 1440].")
    if minutes == 1440:
        return "24:00"

    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def is_all_day(start_min: int, end_min: int) -> bool:
    return start_min == 0 and end_min == 1440


def compute_offpeak_segments(start_min: int, end_min: int) -> list[tuple[int, int]]:
    if is_all_day(start_min, end_min):
        return [(0, 1440)]
    if start_min == end_min:
        raise ValueError("Start and end cannot be equal unless using 00:00–24:00 for all day.")
    if start_min < end_min:
        return [(start_min, end_min)]
    return [(start_min, 1440), (0, end_min)]


def make_summary_lines(start_str: str, end_str: str) -> tuple[str, Optional[str]]:
    start_min = parse_hhmm(start_str, allow_24_end=False)
    end_min = parse_hhmm(end_str, allow_24_end=True)

    _ = compute_offpeak_segments(start_min, end_min)

    if is_all_day(start_min, end_min):
        return ("Off-peak: All day", None)

    start_display = format_hhmm(start_min)
    end_display = format_hhmm(end_min)
    return (
        f"Off-peak: {start_display}–{end_display}",
        f"Peak: {end_display}–{start_display}",
    )
