from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List
import random


@dataclass
class Transition:
    obs: Any
    action: Any
    reward: float
    cost: float
    next_obs: Any
    done: float
    local_cost: float | None = None
    regional_cost: float | None = None


class ReplayBuffer:
    """Simple replay buffer for graph transitions."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        obs: Any,
        action: Any,
        reward: float,
        cost: float,
        next_obs: Any,
        done: bool,
        local_cost: float | None = None,
        regional_cost: float | None = None,
    ) -> None:
        self.buffer.append(
            Transition(
                obs=obs,
                action=action,
                reward=float(reward),
                cost=float(cost),
                next_obs=next_obs,
                done=1.0 if done else 0.0,
                local_cost=float(local_cost) if local_cost is not None else None,
                regional_cost=float(regional_cost) if regional_cost is not None else None,
            )
        )

    def sample(self, batch_size: int) -> List[Transition]:
        return self.rng.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "buffer": list(self.buffer),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.capacity = int(state_dict["capacity"])
        self.buffer = deque(state_dict["buffer"], maxlen=self.capacity)
        self.rng.setstate(state_dict["rng_state"])
