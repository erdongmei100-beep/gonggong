"""Core MILP model for the Bus Synchronization Timetabling with Dwelling Times (BST-DT).

The formulation follows the project instructions: generate all trip-level
arrival variables, enforce synchronization windows with Big-M constraints,
and optionally enforce stop capacity constraints.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

from src.data_models import ModelData

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BSTDTModel:
    """Build and return a Gurobi model for BST-DT."""

    def __init__(self, data: ModelData, enable_capacity_constraints: bool = False) -> None:
        self.data = data
        self.enable_capacity_constraints = enable_capacity_constraints
        self.model: gp.Model | None = None
        self.X: Dict[str, gp.Var] = {}
        self.Z: Dict[Tuple[str, str], gp.Var] = {}
        self.T: Dict[Tuple[str, int, str], gp.Var] = {}
        self.Y: Dict[Tuple[str, str, int, int, str], gp.Var] = {}
        self.O: Dict[Tuple[str, int, str, int], gp.Var] = {}

    def build_model(self) -> gp.Model:
        m = gp.Model("BST-DT")
        self.model = m
        self._create_variables()
        self._add_trip_window_constraints()
        self._add_arrival_constraints()
        self._add_synchronization_constraints()
        self._add_cycle_time_constraints()
        if self.enable_capacity_constraints:
            self._add_capacity_constraints()
        self._set_objective()
        return m

    def _create_variables(self) -> None:
        assert self.model is not None
        m = self.model
        travel_lookup: Dict[str, set[str]]
        if isinstance(self.data.travel_times, list):
            travel_lookup = {}
            for tt in self.data.travel_times:
                line_id = getattr(tt, "line_id", None)
                stop_id = getattr(tt, "to_stop_id", None)
                zone_id = None
                if stop_id and stop_id in self.data.bus_stops:
                    zone_id = self.data.bus_stops[stop_id].zone_id
                if line_id and zone_id:
                    travel_lookup.setdefault(line_id, set()).add(zone_id)
        else:
            travel_lookup = self.data.travel_times
        for line_id, line in self.data.lines.items():
            constraints = self.data.service_constraints.get(line_id)
            lb = constraints.first_trip_min_time if constraints else 0.0
            ub = constraints.first_trip_max_time if constraints else line.headway
            self.X[line_id] = m.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"X[{line_id}]")

            min_dwell = constraints.min_dwell_time if constraints else 0.0
            max_dwell_line = constraints.max_dwell_time if constraints else self.data.model_parameters.max_dwelling_time_global
            max_dwell = min(max_dwell_line, self.data.model_parameters.max_dwelling_time_global)
            valid_zones = travel_lookup.get(line_id, set())
            for zone_id in valid_zones:
                self.Z[(line_id, zone_id)] = m.addVar(
                    lb=min_dwell,
                    ub=max_dwell,
                    vtype=GRB.CONTINUOUS,
                    name=f"Z[{line_id},{zone_id}]",
                )

            for trip in range(1, self.data.num_trips_by_line.get(line_id, 0) + 1):
                for zone_id in valid_zones:
                    self.T[(line_id, trip, zone_id)] = m.addVar(
                        lb=0.0,
                        ub=self.data.model_parameters.planning_horizon,
                        vtype=GRB.CONTINUOUS,
                        name=f"T[{line_id},{trip},{zone_id}]",
                    )

        for pair in self.data.synchronization_pairs:
            trips_i = self.data.num_trips_by_line.get(pair.line_i, 0)
            trips_j = self.data.num_trips_by_line.get(pair.line_j, 0)
            for p in range(1, trips_i + 1):
                for q in range(1, trips_j + 1):
                    self.Y[(pair.line_i, pair.line_j, p, q, pair.zone_id)] = m.addVar(
                        vtype=GRB.BINARY,
                        name=f"Y[{pair.line_i},{pair.line_j},{p},{q},{pair.zone_id}]",
                    )

    def _add_trip_window_constraints(self) -> None:
        assert self.model is not None
        m = self.model
        for line_id, line in self.data.lines.items():
            constraints = self.data.service_constraints.get(line_id)
            trips = self.data.num_trips_by_line.get(line_id, 0)
            if constraints is None:
                continue
            last_departure = self.X[line_id] + (trips - 1) * line.headway
            m.addConstr(
                last_departure >= constraints.last_trip_min_time,
                name=f"last_trip_lb[{line_id}]",
            )
            m.addConstr(
                last_departure <= constraints.last_trip_max_time,
                name=f"last_trip_ub[{line_id}]",
            )

    def _add_arrival_constraints(self) -> None:
        assert self.model is not None
        m = self.model
        for line_id, line in self.data.lines.items():
            headway = line.headway
            zone_sequence = self.data.get_zone_sequence(line_id)
            for trip in range(1, self.data.num_trips_by_line.get(line_id, 0) + 1):
                dwell_prefix = []
                for zone_id in zone_sequence:
                    travel_time = self.data.travel_time_map.get((line_id, zone_id))
                    if travel_time is None:
                        logger.warning("Travel time missing for line %s zone %s", line_id, zone_id)
                        continue
                    arrival_var = self.T.get((line_id, trip, zone_id))
                    dwell_var = self.Z.get((line_id, zone_id))
                    if arrival_var is None or dwell_var is None:
                        logger.debug(
                            "Skipping arrival constraint for line %s trip %s zone %s due to missing variables",
                            line_id,
                            trip,
                            zone_id,
                        )
                        continue
                    prefix_expr = gp.quicksum(dwell_prefix) if dwell_prefix else 0.0
                    m.addConstr(
                        arrival_var == self.X[line_id] + travel_time + (trip - 1) * headway + prefix_expr,
                        name=f"arrival[{line_id},{trip},{zone_id}]",
                    )
                    dwell_prefix.append(dwell_var)

    def _add_synchronization_constraints(self) -> None:
        assert self.model is not None
        m = self.model
        horizon = self.data.model_parameters.planning_horizon
        for pair in self.data.synchronization_pairs:
            trips_i = self.data.num_trips_by_line.get(pair.line_i, 0)
            trips_j = self.data.num_trips_by_line.get(pair.line_j, 0)
            for p in range(1, trips_i + 1):
                for q in range(1, trips_j + 1):
                    y_var = self.Y[(pair.line_i, pair.line_j, p, q, pair.zone_id)]
                    t_i = self.T.get((pair.line_i, p, pair.zone_id))
                    t_j = self.T.get((pair.line_j, q, pair.zone_id))
                    if t_i is None or t_j is None:
                        logger.warning(
                            "Skipping sync constraint for %s-%s at %s due to missing arrival variables",
                            pair.line_i,
                            pair.line_j,
                            pair.zone_id,
                        )
                        continue
                    m.addConstr(
                        t_j - t_i >= pair.min_sync_window - horizon * (1 - y_var),
                        name=f"sync_lb[{pair.line_i},{pair.line_j},{p},{q},{pair.zone_id}]",
                    )
                    m.addConstr(
                        t_j - t_i <= pair.max_sync_window + horizon * (1 - y_var),
                        name=f"sync_ub[{pair.line_i},{pair.line_j},{p},{q},{pair.zone_id}]",
                    )

    def _add_cycle_time_constraints(self) -> None:
        assert self.model is not None
        m = self.model
        for line_id in self.data.lines:
            constraints = self.data.service_constraints.get(line_id)
            base_time = self.data.base_travel_time.get(line_id)
            if constraints is None or base_time is None:
                continue
            dwell_vars = [self.Z[(line_id, z)] for z in self.data.get_zone_sequence(line_id) if (line_id, z) in self.Z]
            if dwell_vars:
                m.addConstr(
                    gp.quicksum(dwell_vars) <= constraints.max_cycle_time_increase * base_time,
                    name=f"cycle_increase[{line_id}]",
                )

    def _add_capacity_constraints(self) -> None:
        assert self.model is not None
        m = self.model
        horizon = int(self.data.model_parameters.planning_horizon)
        big_m = horizon + 10
        occupancy_by_stop: Dict[Tuple[str, int], list[gp.Var]] = {}
        capacity_map: Dict[str, float] = {}

        for (line_id, zone_id, stop_id), assignment in self.data.line_stop_assignments.items():
            stop_cap = self.data.bus_stops.get(stop_id).capacity if stop_id in self.data.bus_stops else self.data.model_parameters.station_capacity_default
            capacity_map[stop_id] = stop_cap
            trips = self.data.num_trips_by_line.get(line_id, 0)
            for trip in range(1, trips + 1):
                t_arr = self.T.get((line_id, trip, zone_id))
                dwell = self.Z.get((line_id, zone_id))
                if t_arr is None or dwell is None:
                    continue
                for t_step in range(horizon + 1):
                    occ_var = m.addVar(vtype=GRB.BINARY, name=f"O[{line_id},{trip},{stop_id},{t_step}]")
                    self.O[(line_id, trip, stop_id, t_step)] = occ_var
                    m.addConstr(t_arr <= t_step + big_m * (1 - occ_var))
                    m.addConstr(t_arr + dwell >= t_step - big_m * (1 - occ_var))
                    occupancy_by_stop.setdefault((stop_id, t_step), []).append(occ_var)

        for (stop_id, t_step), occ_vars in occupancy_by_stop.items():
            cap = capacity_map.get(stop_id, self.data.model_parameters.station_capacity_default)
            m.addConstr(gp.quicksum(occ_vars) <= cap, name=f"cap[{stop_id},{t_step}]")

    def _set_objective(self) -> None:
        assert self.model is not None
        obj = gp.quicksum(weighted_var for weighted_var in self._weighted_sync_vars())
        self.model.setObjective(obj, GRB.MAXIMIZE)

    def _weighted_sync_vars(self):
        for pair in self.data.synchronization_pairs:
            trips_i = self.data.num_trips_by_line.get(pair.line_i, 0)
            trips_j = self.data.num_trips_by_line.get(pair.line_j, 0)
            for p in range(1, trips_i + 1):
                for q in range(1, trips_j + 1):
                    y_var = self.Y.get((pair.line_i, pair.line_j, p, q, pair.zone_id))
                    if y_var is not None:
                        yield pair.sync_weight * y_var


# Backward compatible name used by the scripts
BSTDT_Model = BSTDTModel
