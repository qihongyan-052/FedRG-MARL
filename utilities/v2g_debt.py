from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


ParkType = Literal["residential", "office", "commercial"]
STEP_MINUTES = 15
EPISODE_STEPS = 96


@dataclass
class ParkDebtConfig:
    park_type: ParkType
    alpha: float = 0.20
    comp_min: float = 0.05
    comp_max: float = 0.18
    min_soc_margin_for_v2g: float = 0.05
    departure_debt_penalty_per_kwh: float = 5.0
    departure_soc_shortfall_penalty_per_kwh: float = 8.0


def build_default_park_configs() -> Dict[ParkType, ParkDebtConfig]:
    return {
        "residential": ParkDebtConfig(
            park_type="residential",
            alpha=0.20,
            comp_min=0.06,
            comp_max=0.16,
            min_soc_margin_for_v2g=0.08,
        ),
        "office": ParkDebtConfig(
            park_type="office",
            alpha=0.20,
            comp_min=0.05,
            comp_max=0.15,
            min_soc_margin_for_v2g=0.06,
        ),
        "commercial": ParkDebtConfig(
            park_type="commercial",
            alpha=0.20,
            comp_min=0.06,
            comp_max=0.17,
            min_soc_margin_for_v2g=0.10,
            departure_debt_penalty_per_kwh=5.5,
            departure_soc_shortfall_penalty_per_kwh=8.5,
        ),
    }


@dataclass
class EVDebtAccount:
    ev_id: str
    park_type: ParkType
    battery_capacity_kwh: float
    current_soc: float
    target_departure_soc: float
    min_soc: float
    departure_step: int
    debt_kwh: float = 0.0
    total_v2g_battery_discharge_kwh: float = 0.0
    total_v2g_grid_discharge_kwh: float = 0.0
    total_debt_repaid_kwh: float = 0.0
    total_billable_charge_kwh: float = 0.0
    total_free_charge_kwh: float = 0.0
    total_cash_compensation: float = 0.0
    is_departed: bool = False

    def available_energy_above_min_soc_kwh(self, extra_margin: float) -> float:
        safe_soc = min(max(self.min_soc + extra_margin, 0.0), 1.0)
        return max(0.0, (self.current_soc - safe_soc) * self.battery_capacity_kwh)


@dataclass
class DebtActionResult:
    success: bool
    action_type: str
    ev_id: str
    park_type: ParkType
    step: int
    requested_battery_kwh: float
    executed_battery_kwh: float
    executed_grid_kwh: float
    debt_before_kwh: float
    debt_after_kwh: float
    soc_before: float
    soc_after: float
    compensation_price: float = 0.0
    cash_compensation: float = 0.0
    free_charge_to_debt_kwh: float = 0.0
    billable_charge_kwh: float = 0.0
    charge_revenue: float = 0.0
    message: str = ""


@dataclass
class DepartureSettlementResult:
    success: bool
    ev_id: str
    park_type: ParkType
    departure_step: int
    debt_remaining_kwh: float
    soc_at_departure: float
    target_departure_soc: float
    debt_penalty: float
    soc_shortfall_kwh: float
    soc_shortfall_penalty: float
    total_penalty: float


class MultiParkDebtManager:
    """V2G 补偿与债务机制。补偿按网侧放电能量 * 0.2 * 园区当前售电价结算。"""

    def __init__(self, sell_price_table: Dict[ParkType, List[float]], park_configs: Optional[Dict[ParkType, ParkDebtConfig]] = None) -> None:
        self.sell_price_table = sell_price_table
        self.park_configs = park_configs or build_default_park_configs()
        self.accounts: Dict[str, EVDebtAccount] = {}
        self._validate_price_table()

    def reset(self) -> None:
        self.accounts.clear()

    def register_ev(
        self,
        ev_id: str,
        park_type: ParkType,
        battery_capacity_kwh: float,
        current_soc: float,
        target_departure_soc: float,
        min_soc: float,
        departure_step: int,
    ) -> None:
        self.accounts[ev_id] = EVDebtAccount(
            ev_id=ev_id,
            park_type=park_type,
            battery_capacity_kwh=battery_capacity_kwh,
            current_soc=current_soc,
            target_departure_soc=target_departure_soc,
            min_soc=min_soc,
            departure_step=departure_step,
        )

    def remove_ev(self, ev_id: str) -> None:
        self.accounts.pop(ev_id, None)

    def get_account(self, ev_id: str) -> EVDebtAccount:
        return self.accounts[ev_id]

    def get_sell_price(self, park_type: ParkType, step: int) -> float:
        return float(self.sell_price_table[park_type][step])

    def get_compensation_price(self, park_type: ParkType, step: int) -> float:
        cfg = self.park_configs[park_type]
        price = cfg.alpha * self.get_sell_price(park_type, step)
        return float(max(cfg.comp_min, min(cfg.comp_max, price)))

    def process_v2g_discharge(
        self,
        ev_id: str,
        step: int,
        battery_discharge_kwh: float,
        grid_discharge_kwh: float,
    ) -> DebtActionResult:
        account = self.accounts[ev_id]
        cfg = self.park_configs[account.park_type]
        debt_before = account.debt_kwh
        soc_before = account.current_soc
        if account.is_departed:
            return self._empty_result(account, step, "v2g_discharge", battery_discharge_kwh, "already departed")
        if battery_discharge_kwh <= 0.0 or grid_discharge_kwh <= 0.0:
            return self._empty_result(account, step, "v2g_discharge", battery_discharge_kwh, "zero discharge")
        feasible_battery_kwh = min(battery_discharge_kwh, account.available_energy_above_min_soc_kwh(cfg.min_soc_margin_for_v2g))
        feasible_ratio = feasible_battery_kwh / max(battery_discharge_kwh, 1e-9)
        executed_battery_kwh = feasible_battery_kwh
        executed_grid_kwh = grid_discharge_kwh * feasible_ratio
        if executed_battery_kwh <= 1e-9:
            return self._empty_result(account, step, "v2g_discharge", battery_discharge_kwh, "soc floor reached")
        account.current_soc = max(0.0, account.current_soc - executed_battery_kwh / account.battery_capacity_kwh)
        account.debt_kwh += executed_battery_kwh
        account.total_v2g_battery_discharge_kwh += executed_battery_kwh
        account.total_v2g_grid_discharge_kwh += executed_grid_kwh
        compensation_price = self.get_compensation_price(account.park_type, step)
        cash_compensation = compensation_price * executed_grid_kwh
        account.total_cash_compensation += cash_compensation
        return DebtActionResult(
            success=True,
            action_type="v2g_discharge",
            ev_id=account.ev_id,
            park_type=account.park_type,
            step=step,
            requested_battery_kwh=battery_discharge_kwh,
            executed_battery_kwh=executed_battery_kwh,
            executed_grid_kwh=executed_grid_kwh,
            debt_before_kwh=debt_before,
            debt_after_kwh=account.debt_kwh,
            soc_before=soc_before,
            soc_after=account.current_soc,
            compensation_price=compensation_price,
            cash_compensation=cash_compensation,
            message="v2g discharge processed",
        )

    def process_charge(self, ev_id: str, step: int, battery_charge_kwh: float) -> DebtActionResult:
        account = self.accounts[ev_id]
        debt_before = account.debt_kwh
        soc_before = account.current_soc
        if account.is_departed:
            return self._empty_result(account, step, "charge", battery_charge_kwh, "already departed")
        if battery_charge_kwh <= 0.0:
            return self._empty_result(account, step, "charge", battery_charge_kwh, "zero charge")
        battery_room = max(0.0, account.battery_capacity_kwh * (1.0 - account.current_soc))
        executed_battery_kwh = min(battery_charge_kwh, battery_room)
        if executed_battery_kwh <= 1e-9:
            return self._empty_result(account, step, "charge", battery_charge_kwh, "battery full")
        account.current_soc = min(1.0, account.current_soc + executed_battery_kwh / account.battery_capacity_kwh)
        free_charge_to_debt_kwh = min(executed_battery_kwh, account.debt_kwh)
        billable_charge_kwh = max(0.0, executed_battery_kwh - free_charge_to_debt_kwh)
        account.debt_kwh = max(0.0, account.debt_kwh - free_charge_to_debt_kwh)
        account.total_debt_repaid_kwh += free_charge_to_debt_kwh
        account.total_free_charge_kwh += free_charge_to_debt_kwh
        account.total_billable_charge_kwh += billable_charge_kwh
        charge_revenue = billable_charge_kwh * self.get_sell_price(account.park_type, step)
        return DebtActionResult(
            success=True,
            action_type="charge",
            ev_id=account.ev_id,
            park_type=account.park_type,
            step=step,
            requested_battery_kwh=battery_charge_kwh,
            executed_battery_kwh=executed_battery_kwh,
            executed_grid_kwh=executed_battery_kwh,
            debt_before_kwh=debt_before,
            debt_after_kwh=account.debt_kwh,
            soc_before=soc_before,
            soc_after=account.current_soc,
            free_charge_to_debt_kwh=free_charge_to_debt_kwh,
            billable_charge_kwh=billable_charge_kwh,
            charge_revenue=charge_revenue,
            message="charge processed",
        )

    def settle_departure(self, ev_id: str, departure_step: int) -> DepartureSettlementResult:
        account = self.accounts[ev_id]
        cfg = self.park_configs[account.park_type]
        debt_remaining = max(0.0, account.debt_kwh)
        target_energy = account.target_departure_soc * account.battery_capacity_kwh
        current_energy = account.current_soc * account.battery_capacity_kwh
        soc_shortfall_kwh = max(0.0, target_energy - current_energy)
        debt_penalty = debt_remaining * cfg.departure_debt_penalty_per_kwh
        soc_penalty = soc_shortfall_kwh * cfg.departure_soc_shortfall_penalty_per_kwh
        account.is_departed = True
        return DepartureSettlementResult(
            success=debt_remaining <= 1e-9 and soc_shortfall_kwh <= 1e-9,
            ev_id=account.ev_id,
            park_type=account.park_type,
            departure_step=departure_step,
            debt_remaining_kwh=debt_remaining,
            soc_at_departure=account.current_soc,
            target_departure_soc=account.target_departure_soc,
            debt_penalty=debt_penalty,
            soc_shortfall_kwh=soc_shortfall_kwh,
            soc_shortfall_penalty=soc_penalty,
            total_penalty=debt_penalty + soc_penalty,
        )

    def build_observation_features(self, ev_id: str, current_step: int) -> Dict[str, float]:
        account = self.accounts[ev_id]
        return {
            "current_soc": account.current_soc,
            "target_departure_soc": account.target_departure_soc,
            "debt_kwh": account.debt_kwh,
            "time_to_departure_steps": float(max(0, account.departure_step - current_step)),
            "sell_price": self.get_sell_price(account.park_type, current_step),
            "compensation_price": self.get_compensation_price(account.park_type, current_step),
        }

    def _validate_price_table(self) -> None:
        for park in ("residential", "office", "commercial"):
            if park not in self.sell_price_table:
                raise ValueError(f"missing park price table: {park}")
            if len(self.sell_price_table[park]) != EPISODE_STEPS:
                raise ValueError(f"price table for {park} must have {EPISODE_STEPS} steps")

    @staticmethod
    def _empty_result(account: EVDebtAccount, step: int, action_type: str, requested_kwh: float, message: str) -> DebtActionResult:
        return DebtActionResult(
            success=False,
            action_type=action_type,
            ev_id=account.ev_id,
            park_type=account.park_type,
            step=step,
            requested_battery_kwh=requested_kwh,
            executed_battery_kwh=0.0,
            executed_grid_kwh=0.0,
            debt_before_kwh=account.debt_kwh,
            debt_after_kwh=account.debt_kwh,
            soc_before=account.current_soc,
            soc_after=account.current_soc,
            message=message,
        )
