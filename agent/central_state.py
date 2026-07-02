from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math

from agent.state import get_node_sizes, normalize_privacy_mode


PARK_TYPES: tuple[str, ...] = ("residential", "office", "commercial")

CENTRAL_NODE_SIZES: Dict[str, int] = {
    "cs": 7,
    "bes": 4,
    "pv": 3,
    "external": 7,
    "ev": 8,
}


def get_central_node_sizes(privacy_mode: str) -> Dict[str, int]:
    resolved_privacy_mode = normalize_privacy_mode(privacy_mode)
    if resolved_privacy_mode != "none":
        return get_node_sizes(resolved_privacy_mode)
    return dict(CENTRAL_NODE_SIZES)


@dataclass
class CentralGraphBuilder:
    """
    Build a central-agent observation graph without changing the original local StateBuilder.

    Design goals:
    - Reuse the existing per-park node features as the physical state backbone.
    - Remove local-agent-specific cross-park summary features from park CS nodes.
    - Keep per-park external nodes for park-level prices and local exogenous context.
    - Add one global transformer-facing external node that only carries system-level
      coordination signals for the central controller.
    """

    privacy_mode: str

    def __post_init__(self) -> None:
        self.node_sizes = get_central_node_sizes(self.privacy_mode)

    def build(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        park_graphs = obs["park_graphs"]
        node_types: List[str] = []
        node_names: List[str] = []
        edge_index: List[Tuple[int, int]] = []
        combined_features: Dict[str, List[List[float]]] = {
            "cs": [],
            "bes": [],
            "pv": [],
            "external": [],
            "ev": [],
        }
        combined_indexes: Dict[str, List[int]] = {
            "cs": [],
            "bes": [],
            "pv": [],
            "external": [],
            "ev": [],
        }
        park_action_slices: Dict[str, Dict[str, Any]] = {}
        bes_node_to_park: Dict[int, str] = {}
        ev_node_to_id_and_park: Dict[int, Tuple[str, str]] = {}

        for park_type in PARK_TYPES:
            graph = park_graphs[park_type]
            offset = len(node_types)
            node_types.extend(graph["node_types"])
            node_names.extend(graph["node_names"])
            edge_index.extend(
                (int(src) + offset, int(dst) + offset)
                for src, dst in graph["edge_index"]
            )

            cs_features = [self._project_central_cs_features(feature_row) for feature_row in graph["cs_features"]]
            combined_features["cs"].extend(cs_features)
            combined_features["bes"].extend(graph["bes_features"])
            combined_features["pv"].extend(graph["pv_features"])
            external_features = [
                self._project_central_external_features(feature_row)
                for feature_row in graph["external_features"]
            ]
            combined_features["external"].extend(external_features)
            combined_features["ev"].extend(graph["ev_features"])

            for node_type in ("cs", "bes", "pv", "external", "ev"):
                indexes = [int(index) + offset for index in graph[f"{node_type}_indexes"]]
                combined_indexes[node_type].extend(indexes)

            bes_index = (int(graph["bes_indexes"][0]) + offset) if graph["bes_indexes"] else None
            ev_indexes = [int(index) + offset for index in graph["ev_indexes"]]
            for ev_global_index, ev_local_index in zip(ev_indexes, graph["ev_indexes"]):
                ev_id = str(graph["node_names"][int(ev_local_index)])
                ev_node_to_id_and_park[ev_global_index] = (park_type, ev_id)
            if bes_index is not None:
                bes_node_to_park[bes_index] = park_type

            park_action_slices[park_type] = {
                "bes_index": bes_index,
                "ev_indexes": ev_indexes,
                "cs_index": (int(graph["cs_indexes"][0]) + offset) if graph["cs_indexes"] else None,
                "local_node_count": len(graph["node_types"]),
                "local_bes_index": int(graph["bes_indexes"][0]) if graph["bes_indexes"] else None,
                "local_ev_indexes": [int(index) for index in graph["ev_indexes"]],
            }

        global_tr_external_index = len(node_types)
        node_types.append("external")
        node_names.append("global_tr_external")
        combined_indexes["external"].append(global_tr_external_index)
        combined_features["external"].append(
            self._build_global_tr_external_features(park_graphs)
        )

        for park_type in PARK_TYPES:
            cs_index = park_action_slices[park_type]["cs_index"]
            if cs_index is None:
                continue
            edge_index.append((global_tr_external_index, cs_index))
            edge_index.append((cs_index, global_tr_external_index))

        return {
            "privacy_mode": obs.get("privacy_mode", self.privacy_mode),
            "node_types": node_types,
            "node_names": node_names,
            "edge_index": edge_index,
            "cs_indexes": combined_indexes["cs"],
            "bes_indexes": combined_indexes["bes"],
            "pv_indexes": combined_indexes["pv"],
            "external_indexes": combined_indexes["external"],
            "ev_indexes": combined_indexes["ev"],
            "cs_features": combined_features["cs"],
            "bes_features": combined_features["bes"],
            "pv_features": combined_features["pv"],
            "external_features": combined_features["external"],
            "ev_features": combined_features["ev"],
            "park_action_slices": park_action_slices,
            "park_order": list(PARK_TYPES),
            "global_tr_external_index": global_tr_external_index,
            "bes_node_to_park": bes_node_to_park,
            "ev_node_to_id_and_park": ev_node_to_id_and_park,
        }

    @staticmethod
    def _project_central_cs_features(feature_row: List[float]) -> List[float]:
        # Central agent already sees all three park subgraphs, so it does not need
        # the local-agent-oriented "other parks" summary tail on the CS node.
        if len(feature_row) <= 7:
            return list(feature_row)
        return list(feature_row[:7])

    @staticmethod
    def _project_central_external_features(feature_row: List[float]) -> List[float]:
        # Keep park-level external context unchanged for the central agent:
        # time, weather, park buy/sell prices, and per-park TR feedback.
        return list(feature_row[:7])

    @staticmethod
    def _build_global_tr_external_features(park_graphs: Dict[str, Dict[str, Any]]) -> List[float]:
        cs_rows = [park_graphs[park_type]["cs_features"][0] for park_type in PARK_TYPES]
        external_rows = [park_graphs[park_type]["external_features"][0] for park_type in PARK_TYPES]

        total_net_pressure = math.tanh(sum(float(row[3]) for row in cs_rows))
        total_same_direction_controllable = math.tanh(sum(float(row[4]) for row in cs_rows))
        max_overload_feedback = max(float(row[5]) for row in external_rows)
        mean_signed_control_signal = sum(float(row[6]) for row in external_rows) / float(len(external_rows))
        mean_local_flexibility_ratio = sum(float(row[2]) for row in cs_rows) / float(len(cs_rows))
        max_prev_reduction_ratio = max(float(row[5]) for row in cs_rows)
        mean_prev_local_penalty = sum(float(row[6]) for row in cs_rows) / float(len(cs_rows))

        return [
            total_net_pressure,
            total_same_direction_controllable,
            max_overload_feedback,
            mean_signed_control_signal,
            mean_local_flexibility_ratio,
            max_prev_reduction_ratio,
            mean_prev_local_penalty,
        ]


def build_central_tr_graph(obs: Dict[str, Any], privacy_mode: str | None = None) -> Dict[str, Any]:
    resolved_privacy_mode = privacy_mode if privacy_mode is not None else str(obs.get("privacy_mode", "strong"))
    return CentralGraphBuilder(privacy_mode=resolved_privacy_mode).build(obs)
