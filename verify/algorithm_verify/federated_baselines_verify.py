from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Fed_average.fed_controller import FederatedAveragingCoordinator
from evaluate_three_park_agent import (
    EvaluationConfig,
    _infer_saved_experiment_signature,
    _validate_saved_route_signature,
)
from train_three_park_agent import (
    TrainingConfig,
    _serialize_agent_for_non_interrupt_model_save,
    build_local_agents,
    resolve_federation_scheme,
    validate_training_config,
)


class _DummyAgent:
    def __init__(self, value: float) -> None:
        self.state = {"actor_backbone": {"weight": torch.tensor([value])}}
        self.reference = None

    def get_shared_state(self):
        return self.state

    def get_actor_relation_fed_mask(self):
        return {}

    def load_shared_state(self, state):
        self.state = state

    def set_global_actor_reference(self, state):
        self.reference = state


def _base_config(**overrides) -> TrainingConfig:
    config = TrainingConfig(
        run_name="federated-baseline-verify",
        algorithm_variant="sp_rgnn_csac",
        enable_federation=True,
        federation_scheme="personalized",
        actor_proximal_weight=2e-4,
        update_device="cpu",
        act_device="cpu",
    )
    return replace(config, **overrides)


def verify_scheme_resolution_and_validation() -> None:
    personalized = _base_config()
    validate_training_config(personalized)
    assert resolve_federation_scheme(personalized) == "personalized"

    fedavg = _base_config(federation_scheme="fedavg", actor_proximal_weight=0.0)
    validate_training_config(fedavg)
    assert resolve_federation_scheme(fedavg) == "fedavg"

    fedprox = _base_config(federation_scheme="fedprox", actor_proximal_weight=2e-4)
    validate_training_config(fedprox)
    assert resolve_federation_scheme(fedprox) == "fedprox"

    try:
        validate_training_config(_base_config(federation_scheme="fedavg", actor_proximal_weight=2e-4))
    except RuntimeError as exc:
        assert "actor_proximal_weight=0.0" in str(exc)
    else:
        raise AssertionError("FedAvg accepted a non-zero proximal weight")

    for config, expected in ((fedavg, "fedavg"), (fedprox, "fedprox")):
        agents = build_local_agents(config)
        assert all(agent.config.federation_scheme == expected for agent in agents.values())
        checkpoint = _serialize_agent_for_non_interrupt_model_save(
            next(iter(agents.values())),
            park_type="residential",
            episode=0,
        )
        saved_config = checkpoint["agent_config"]
        assert saved_config["federation_scheme"] == expected
        assert saved_config["actor_proximal_weight"] == config.actor_proximal_weight
        saved_signature = _infer_saved_experiment_signature(checkpoint)
        assert saved_signature["federation_scheme"] == expected
        _validate_saved_route_signature(
            saved_signature,
            EvaluationConfig(
                algorithm_variant="sp_rgnn_csac",
                enable_federation=True,
                federation_scheme=expected,
                privacy_mode=config.privacy_mode,
                decouple_actor_output_heads=config.decouple_actor_output_heads,
            ),
        )


def verify_uniform_fedavg_and_resume_state() -> None:
    agents = {
        "residential": _DummyAgent(1.0),
        "office": _DummyAgent(2.0),
        "commercial": _DummyAgent(3.0),
    }
    coordinator = FederatedAveragingCoordinator()
    coordinator.aggregate(
        agents,
        normalized_weights={park_id: 1.0 / 3.0 for park_id in agents},
        selected_blocks=["actor_backbone"],
    )
    for agent in agents.values():
        assert torch.allclose(agent.state["actor_backbone"]["weight"], torch.tensor([2.0]))
        assert agent.reference is not None
    assert coordinator.aggregate_count == 1
    assert coordinator.last_metrics["fed_weights"] == [[1.0 / 3.0] * 3] * 3

    restored = FederatedAveragingCoordinator()
    restored.load_state(coordinator.export_state())
    assert restored.aggregate_count == coordinator.aggregate_count
    assert restored.last_metrics == coordinator.last_metrics


if __name__ == "__main__":
    verify_scheme_resolution_and_validation()
    verify_uniform_fedavg_and_resume_state()
    print("federated baseline verification passed")
