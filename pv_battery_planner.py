#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import planner_core as core

if __name__ != "__main__":
    sys.modules[__name__] = core
else:
    inputs = core.PlannerInputs(
        soc_at_22=core.ask_required_soc("Battery SOC at 22:00 from FusionSolar (%): "),
        yesterday_consumption_kwh=0.0,
    )

    while True:
        yesterday_consumption_kwh = core.ask_required_float("Total consumption yesterday from FusionSolar (kWh): ")
        if yesterday_consumption_kwh > 0:
            inputs.yesterday_consumption_kwh = yesterday_consumption_kwh
            break
        print("Consumption must be > 0.")

    out = core.run_planner(inputs)
    core.print_hourly_pv(out.hourly_df, out.weather.sunrise, out.weather.sunset)
    core.print_expensive_hourly_flow(out.expensive_detail_df, out.tomorrow_date)
    core.print_fusionsolar_actions(out.cutoff_soc, out.charge_kw, out.cutoff_note)
