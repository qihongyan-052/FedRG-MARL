from __future__ import annotations

from typing import List

from env.three_park_charging_env import PARK_TYPES


def realtime_projection_log_fields() -> List[str]:
    fields = ["episode", "step", "time", "weather"]
    for park_type in PARK_TYPES:
        fields.extend(
            [
                f"{park_type}_cs_projected_demand_kwh",
                f"{park_type}_cs_limit_kwh",
            ]
        )
    fields.extend(
        [
            "tr_net_demand_kwh",
            "tr_limit_kwh",
            "tr_triggered",
            "tr_overload_direction",
            "tr_overload_kwh",
            "tr_total_capacity_kwh",
            "tr_total_preference_capacity_kwh",
            "tr_total_responsibility",
            "tr_safety_base_ratio",
            "tr_blended_capacity_kwh",
            "tr_scaling_coefficient",
            "tr_penalty_coefficient",
            "tr_infeasible_residual_kwh",
        ]
    )
    return fields
