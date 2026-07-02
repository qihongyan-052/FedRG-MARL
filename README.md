# FedRG-MARL

This repository provides the source code for FedRG-MARL, a privacy-preserving federated relational-graph multi-agent reinforcement learning framework for electric vehicle scheduling in transformer-coupled multi-park microgrids.

## Overview

FedRG-MARL is designed for coordinated electric vehicle scheduling in regional multi-park microgrids under shared-transformer coupling, heterogeneous resources, stochastic charging demand, and privacy constraints. The framework integrates relation-aware graph representation, constrained actor-critic learning, personalized federated aggregation, and privacy-preserving regional coordination.

## Repository structure

- `agent/`: agent-related modules
- `algorithm/`: reinforcement learning and optimization algorithms
- `Fed_average/`: federated aggregation modules
- `env/`: multi-park microgrid simulation environment
- `safety_design/`: safety constraint and feasibility handling modules
- `tr_coordination/`: transformer-level regional coordination modules
- `utilities/`: utility functions
- `config_files/`: configuration files
- `sample/`: sample input or scenario files
- `visualize/`: visualization scripts
- `verify/env_verify/`: environment verification scripts

Main scripts include:

- `train_three_park_agent.py`: training script for the three-park scheduling case
- `evaluate_three_park_agent.py`: evaluation script for the trained FedRG-MARL agent
- `evaluate_rule_based_baseline.py`: rule-based baseline evaluation
- `evaluate_greedy_max_charge_baseline.py`: greedy charging baseline evaluation
- `evaluate_lp_mpc_baseline.py`: LP-MPC baseline evaluation
- `plot_tr_coordination_from_eval_steps.py`: visualization of transformer coordination results

## Environment

The code was developed using Python and PyTorch.

Please install the required packages according to your local Python environment. If a `requirements.txt` file is provided, install dependencies with:

```bash
pip install -r requirements.txt
