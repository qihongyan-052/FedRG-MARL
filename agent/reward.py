from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# Training-side weights: these directly shape the reward/cost stored into replay buffer.
TRAIN_PROFIT_WEIGHT = 1.0
TRAIN_USER_SATISFACTION_PENALTY_WEIGHT = 1.0
TRAIN_CS_PENALTY_WEIGHT = 0.2
TRAIN_TR_PENALTY_WEIGHT = 0.2
TRAIN_BES_TERMINAL_PENALTY_WEIGHT = 1.0
TRAIN_DEBT_PENALTY_WEIGHT = 1.0

# Logging uses the same coefficients as training so CSV output, plots, checkpoint
# selection, and replay-buffer rewards all share one reward definition.
LOG_PROFIT_WEIGHT = TRAIN_PROFIT_WEIGHT
LOG_USER_SATISFACTION_PENALTY_WEIGHT = TRAIN_USER_SATISFACTION_PENALTY_WEIGHT
LOG_CS_PENALTY_WEIGHT = TRAIN_CS_PENALTY_WEIGHT
LOG_TR_PENALTY_WEIGHT = TRAIN_TR_PENALTY_WEIGHT
LOG_BES_TERMINAL_PENALTY_WEIGHT = TRAIN_BES_TERMINAL_PENALTY_WEIGHT
LOG_DEBT_PENALTY_WEIGHT = TRAIN_DEBT_PENALTY_WEIGHT


@dataclass(frozen=True)
class TrainingRewardWeights:
    profit_weight: float = TRAIN_PROFIT_WEIGHT
    user_satisfaction_penalty_weight: float = TRAIN_USER_SATISFACTION_PENALTY_WEIGHT
    cs_penalty_weight: float = TRAIN_CS_PENALTY_WEIGHT
    tr_penalty_weight: float = TRAIN_TR_PENALTY_WEIGHT
    bes_terminal_penalty_weight: float = TRAIN_BES_TERMINAL_PENALTY_WEIGHT
    debt_penalty_weight: float = TRAIN_DEBT_PENALTY_WEIGHT


@dataclass(frozen=True)
class LoggingRewardWeights:
    profit_weight: float = LOG_PROFIT_WEIGHT
    user_satisfaction_penalty_weight: float = LOG_USER_SATISFACTION_PENALTY_WEIGHT
    cs_penalty_weight: float = LOG_CS_PENALTY_WEIGHT
    tr_penalty_weight: float = LOG_TR_PENALTY_WEIGHT
    bes_terminal_penalty_weight: float = LOG_BES_TERMINAL_PENALTY_WEIGHT
    debt_penalty_weight: float = LOG_DEBT_PENALTY_WEIGHT


@dataclass(frozen=True)
class EVDepartureRecord:
    ev_id: str
    park_type: str
    soc_at_departure: float
    target_soc: float
    debt_remaining_kwh: float
    soc_shortfall_kwh: float = 0.0


@dataclass(frozen=True)
class StepRewardInput:
    ev_charge_revenue: float
    grid_sale_revenue: float
    grid_purchase_cost: float
    v2g_compensation_cost: float
    cs_projection_penalty_abs: float
    tr_projection_penalty_abs: float
    departure_records: List[EVDepartureRecord] = field(default_factory=list)
    is_terminal_step: bool = False
    bes_terminal_energy_penalty_abs: float = 0.0


@dataclass(frozen=True)
class RewardBreakdown:
    profit_term: float
    user_satisfaction_penalty: float
    cs_projection_penalty: float
    tr_projection_penalty: float
    bes_terminal_penalty: float
    debt_penalty: float
    training_profit_reward: float
    training_user_satisfaction_penalty: float
    training_cs_projection_penalty: float
    training_tr_projection_penalty: float
    training_bes_terminal_penalty: float
    training_debt_penalty: float
    training_constraint_cost: float
    training_total_reward: float
    logging_profit_reward: float
    logging_constraint_cost: float
    total_reward: float

    @property
    def profit_reward(self) -> float:
        return self.training_profit_reward

    @property
    def constraint_cost(self) -> float:
        return self.training_constraint_cost

    @property
    def logging_total_reward(self) -> float:
        return self.total_reward


class RewardBuilder:
    """
    Centralized reward bookkeeping with two coefficient sets:
    1. training weights for replay-buffer reward/cost
    2. logging weights for printed/logged total reward
    """

    def __init__(
        self,
        training_weights: TrainingRewardWeights | None = None,
        logging_weights: LoggingRewardWeights | None = None,
    ) -> None:
        self.training_weights = training_weights or TrainingRewardWeights()
        self.logging_weights = logging_weights or LoggingRewardWeights()

    def compute(self, reward_input: StepRewardInput) -> RewardBreakdown:
        profit_term = self._compute_profit_term(reward_input)
        user_satisfaction_penalty = self._compute_user_satisfaction_penalty(reward_input.departure_records)
        cs_projection_penalty = self._compute_cs_projection_penalty(reward_input)
        tr_projection_penalty = self._compute_tr_projection_penalty(reward_input)
        bes_terminal_penalty = self._compute_bes_terminal_penalty(reward_input)
        debt_penalty = self._compute_debt_penalty(reward_input.departure_records)

        training_profit_reward = self.training_weights.profit_weight * profit_term
        training_constraint_cost = (
            self.training_weights.user_satisfaction_penalty_weight * user_satisfaction_penalty
            + self.training_weights.cs_penalty_weight * cs_projection_penalty
            + self.training_weights.tr_penalty_weight * tr_projection_penalty
            + self.training_weights.bes_terminal_penalty_weight * bes_terminal_penalty
            + self.training_weights.debt_penalty_weight * debt_penalty
        )
        training_user_satisfaction_penalty = (
            self.training_weights.user_satisfaction_penalty_weight * user_satisfaction_penalty
        )
        training_cs_projection_penalty = self.training_weights.cs_penalty_weight * cs_projection_penalty
        training_tr_projection_penalty = self.training_weights.tr_penalty_weight * tr_projection_penalty
        training_bes_terminal_penalty = (
            self.training_weights.bes_terminal_penalty_weight * bes_terminal_penalty
        )
        training_debt_penalty = self.training_weights.debt_penalty_weight * debt_penalty
        training_total_reward = training_profit_reward - training_constraint_cost

        logging_profit_reward = self.logging_weights.profit_weight * profit_term
        logging_constraint_cost = (
            self.logging_weights.user_satisfaction_penalty_weight * user_satisfaction_penalty
            + self.logging_weights.cs_penalty_weight * cs_projection_penalty
            + self.logging_weights.tr_penalty_weight * tr_projection_penalty
            + self.logging_weights.bes_terminal_penalty_weight * bes_terminal_penalty
            + self.logging_weights.debt_penalty_weight * debt_penalty
        )

        total_reward = logging_profit_reward - logging_constraint_cost

        return RewardBreakdown(
            profit_term=profit_term,
            user_satisfaction_penalty=user_satisfaction_penalty,
            cs_projection_penalty=cs_projection_penalty,
            tr_projection_penalty=tr_projection_penalty,
            bes_terminal_penalty=bes_terminal_penalty,
            debt_penalty=debt_penalty,
            training_profit_reward=training_profit_reward,
            training_user_satisfaction_penalty=training_user_satisfaction_penalty,
            training_cs_projection_penalty=training_cs_projection_penalty,
            training_tr_projection_penalty=training_tr_projection_penalty,
            training_bes_terminal_penalty=training_bes_terminal_penalty,
            training_debt_penalty=training_debt_penalty,
            training_constraint_cost=training_constraint_cost,
            training_total_reward=training_total_reward,
            logging_profit_reward=logging_profit_reward,
            logging_constraint_cost=logging_constraint_cost,
            total_reward=total_reward,
        )

    @staticmethod
    def _compute_profit_term(reward_input: StepRewardInput) -> float:
        return (
            reward_input.ev_charge_revenue
            + reward_input.grid_sale_revenue
            - reward_input.grid_purchase_cost
            - reward_input.v2g_compensation_cost
        )

    @staticmethod
    def _compute_user_satisfaction_penalty(departure_records: List[EVDepartureRecord]) -> float:
        return sum(max(0.0, record.soc_shortfall_kwh) for record in departure_records)

    @staticmethod
    def _compute_cs_projection_penalty(reward_input: StepRewardInput) -> float:
        reduction_kwh = abs(reward_input.cs_projection_penalty_abs)
        return reduction_kwh * reduction_kwh

    @staticmethod
    def _compute_tr_projection_penalty(reward_input: StepRewardInput) -> float:
        return abs(reward_input.tr_projection_penalty_abs)

    @staticmethod
    def _compute_bes_terminal_penalty(reward_input: StepRewardInput) -> float:
        if not reward_input.is_terminal_step:
            return 0.0
        return abs(reward_input.bes_terminal_energy_penalty_abs)

    @staticmethod
    def _compute_debt_penalty(departure_records: List[EVDepartureRecord]) -> float:
        return sum(max(0.0, record.debt_remaining_kwh) for record in departure_records)
