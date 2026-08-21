# FedRG-MARL: Privacy-Aware Personalized Federated Multi-Agent RL for Multi-Park EV Charging

This repository contains the implementation of a three-park electric-vehicle (EV) charging control simulator and the associated reinforcement-learning baselines.  The main method, **FedRG-MARL**, combines a shared-private relation-aware graph constrained SAC agent with personalized actor federation.

The simulated parks are residential, office, and commercial. Each park controls EV charging/discharging and battery energy storage (BES), while charging-station and transformer constraints are enforced by the environment.

> This public code repository intentionally excludes experiment outputs and manuscript materials. See [Data and artifact availability](#data-and-artifact-availability).

## Main features

- Three-park EV charging environment with a 96-step, 15-minute daily horizon.
- Local SAC, constrained SAC, TD3, GNN, HGT, MLP, SP-RGNN-CSAC, and rule-based / optimization baselines.
- Shared-private relation-aware graph encoder:
  - relation-specific shared transformations;
  - local low-rank relation adapters;
  - optional relation-gated fusion;
  - optional typed critic pooling.
- Personalized actor federation with target-specific source compatibility scoring, learned source weights, personalized candidates, and gated soft loading.
- Transformer (TR) coordination and a fixed-point pairwise additive-mask simulation for regional scalar aggregation.
- Training, checkpoint evaluation, baseline evaluation, verification, plotting, and profiling utilities.

## Repository layout

```text
.
+-- agent/                 # State construction, rewards, replay buffer
+-- algorithm/             # RL agents and non-learning baselines
+-- config_files/          # Topology, prices, vehicle, and sampling configuration
+-- env/                   # Three-park charging environment
+-- Fed_average/           # FedAvg/FedProx coordinator and personalized federation
+-- safety_design/         # EV, BES, charging-station safety projections
+-- sample/                # Stochastic EV, SOC, arrival, stay-time, and PV samplers
+-- tr_coordination/       # Regional transformer coordination and additive masking
+-- utilities/             # Supporting utilities
+-- verify/                # Environment and algorithm verification scripts
+-- visualize/             # Training and architecture visualization utilities
+-- train_three_park_agent.py
+-- evaluate_three_park_agent.py
+-- evaluate_greedy_max_charge_baseline.py
+-- evaluate_lp_mpc_baseline.py
+-- evaluate_rule_based_baseline.py
`-- profile_r2_8_overhead.py
```

## Installation

The codebase is written for Python 3.9+ and PyTorch. Install a PyTorch build appropriate for your platform first, then install the remaining dependencies:

```bash
pip install numpy matplotlib psutil
pip install torch-geometric
```

The project also uses standard-library modules only for CSV, JSON, timing, and configuration handling. GPU execution is optional; the device is controlled through `TrainingConfig.act_device` and `TrainingConfig.update_device`.

## Configuration

The main run settings are defined by the `TrainingConfig` dataclass in `train_three_park_agent.py`:

- `algorithm_variant`: selects an RL architecture or ablation.
- `enable_federation`: enables parameter federation.
- `federation_scheme`: `personalized`, `fedavg`, `fedprox`, `none`, or `auto`.
- `privacy_mode`: `strong` or `none` for the regional coordination path.
- `seed`: training seed.
- `total_episodes`: number of training episodes.

The three-park topology and device limits are configured in `config_files/three_parks_topology_config.json`.

### Main FedRG-MARL configuration

Use the following settings for the personalized federated SP-RGNN-CSAC route:

```python
TrainingConfig(
    algorithm_variant="sp_rgnn_csac",
    enable_federation=True,
    federation_scheme="personalized",
    federate_critic_backbone=False,
    privacy_mode="strong",
)
```

The actual training entry point is:

```bash
python train_three_park_agent.py
```

Before launching a run, set `run_name` and the desired fields in `TrainingConfig` (or invoke `run_training()` from a separate driver script with an explicit `TrainingConfig`).

## Evaluation

Checkpoint evaluation is implemented in `evaluate_three_park_agent.py`. It restores the model checkpoint, validates the saved experiment signature, and evaluates deterministic policies over stochastic daily scenarios.

```bash
python evaluate_three_park_agent.py
```

Key evaluation fields are in `EvaluationConfig`:

- `run_name`
- `algorithm_variant`
- `enable_federation`
- `federation_scheme`
- `checkpoint_kind` (`best` or `final`)
- `seed` (evaluation seed)
- `eval_episodes`

`seed` in `EvaluationConfig` is an **evaluation seed**, not a training seed. For each evaluation episode, the environment is reset with `episode_seed = seed + episode`.

Evaluation step records use CSV fields including:

- `total_profit_reward`
- `total_user_payment_cost`
- `total_v2g_compensation_cost`
- `total_constraint_cost`
- park-level profit, energy, and state fields.

The greedy maximum-charge, LP-MPC, and rule-based baselines have separate evaluation entry points.

## Federation implementation note

The personalized federation code is located in `Fed_average/learnable_personalized_fed_actor.py`.

For each round, each local agent exports the shared relation parameters of its actor backbone. The coordinator evaluates each source backbone using each target agent's local scoring function, constructs one personalized candidate per target, evaluates the candidate, and soft-loads the accepted candidate into the target actor.

This repository implements federation as a centralized single-process simulation. It does **not** implement encrypted model parameters, secure aggregation of neural-network parameters, MPC, homomorphic encryption, or a trusted execution environment. The pairwise additive masking code in `tr_coordination/strong_privacy_coordinator.py` applies only to selected regional scalar aggregations, not to federated neural-network parameters.

## Reproducibility utilities

- `verify/`: environment and algorithm verification scripts.
- `visualize/train/`: training reward and component plots.
- `plot_tr_coordination_from_eval_steps.py`: transformer-coordination plots from evaluation step CSVs.
- `profile_r2_8_overhead.py`: checkpoint-based communication, parameter-inventory, federation-runtime, additive-mask, and online-latency profiling.

Profiling is single-process CPU profiling and does not measure network transfer latency or throughput.

## Data and artifact availability

The following directories are intentionally **not included** in the GitHub repository:

```text
saved/       # Checkpoints, training logs, evaluation CSVs, generated figures, profiling artifacts
返修意见/     # Revision materials
论文/          # Manuscript materials
```

As a result, a fresh clone contains the source code and configuration files, but not pretrained checkpoints or the experiment artifacts required to reproduce reported numerical results directly. Training and evaluation scripts expect experiment artifacts under `saved/<run_name>/` when checkpoint evaluation is requested.

## License

No license has been specified yet. Add a license file before distributing or reusing this code outside the intended project scope.
