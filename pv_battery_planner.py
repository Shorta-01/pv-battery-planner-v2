#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
import sys
import planner_core as core

if __name__ != "__main__":
    sys.modules[__name__] = core
else:
    soc_at_22_percent = core.ask_required_soc("Battery SOC at 22:00 from FusionSolar (%): ") * 100.0

    while True:
        yesterday_consumption_kwh = core.ask_required_float("Total consumption yesterday from FusionSolar (kWh): ")
        if yesterday_consumption_kwh > 0:
            break
        print("Consumption must be > 0.")

    target_date = dt.date.today() + dt.timedelta(days=1)
    cfg = core.get_effective_config()
    out = core.run_forecast_pipeline(
        cfg=cfg,
        target_date=target_date,
        soc_at_22_percent=soc_at_22_percent,
        yesterday_kwh=yesterday_consumption_kwh,
        buffer_percent=0.0,
        user_max_ac_kw=float(cfg["battery"].get("max_ac_charge_kw_hard_limit", core.MAX_AC_CHARGE_KW_HARD_LIMIT)),
    )
    core.print_hourly_pv(out.hourly_df, out.weather.sunrise, out.weather.sunset)
    core.print_expensive_hourly_flow(out.expensive_detail_df, out.tomorrow_date)
    core.print_fusionsolar_actions(out.cutoff_soc, out.charge_kw, out.cutoff_note)
