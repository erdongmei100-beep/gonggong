"""Load BST-DT input data from CSV files.

This module reads the provided CSV files, converts them into typed data
classes, and builds the helper dictionaries required by the optimization
model (trip counts, zone sequences, travel times to zones).
"""
from __future__ import annotations

import logging
from math import floor
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.data_models import (
    BusLine,
    BusStop,
    LineStopAssignment,
    ModelData,
    ModelParameters,
    ServiceConstraint,
    SynchronizationPair,
    TransferZone,
    TravelTimeEntry,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BSTDTDataLoader:
    """Loader for the BST-DT dataset."""

    def __init__(self, data_dir: str = "./data_complete") -> None:
        self.data_dir = Path(data_dir)

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        return pd.read_csv(path)

    def load_all_data(self) -> ModelData:
        lines_df = self._load_csv("lines.csv")
        zones_df = self._load_csv("transfer_zones.csv")
        stops_df = self._load_csv("bus_stops.csv")
        assignments_df = self._load_csv("line_stop_assignments.csv")
        travel_times_df = self._load_csv("travel_times.csv")
        sync_pairs_df = self._load_csv("synchronization_pairs.csv")
        model_params_df = self._load_csv("model_parameters.csv")
        service_constraints_df = self._load_csv("service_constraints.csv")

        lines = self._parse_lines(lines_df)
        zones = self._parse_transfer_zones(zones_df)
        stops = self._parse_bus_stops(stops_df)
        assignments = self._parse_assignments(assignments_df)
        travel_times = self._parse_travel_times(travel_times_df)
        sync_pairs = self._parse_sync_pairs(sync_pairs_df)
        service_constraints = self._parse_service_constraints(service_constraints_df)
        model_parameters = self._parse_model_parameters(model_params_df)

        stop_zone_map: Dict[str, str] = {
            stop_id.strip(): stop.zone_id.strip() for stop_id, stop in stops.items()
        }
        for zone_id in zones.keys():
            stop_zone_map.setdefault(zone_id.strip(), zone_id.strip())
        line_zone_sequence = self._build_zone_sequences(assignments)
        travel_time_map, base_travel = self._build_travel_time_map(
            travel_times, stop_zone_map, set(lines.keys()), zones
        )
        num_trips_by_line = self._compute_trip_counts(lines, model_parameters.planning_horizon)

        return ModelData(
            lines=lines,
            transfer_zones=zones,
            bus_stops=stops,
            line_stop_assignments=assignments,
            travel_times=travel_times,
            synchronization_pairs=sync_pairs,
            service_constraints=service_constraints,
            model_parameters=model_parameters,
            travel_time_map=travel_time_map,
            line_zone_sequence=line_zone_sequence,
            num_trips_by_line=num_trips_by_line,
            base_travel_time=base_travel,
        )

    def _parse_lines(self, df: pd.DataFrame) -> Dict[str, BusLine]:
        lines: Dict[str, BusLine] = {}
        for _, row in df.iterrows():
            line_id = str(row["line_id"])
            lines[line_id] = BusLine(
                line_id=line_id,
                headway=float(row["headway"]),
                frequency=int(row["frequency"]),
                name=str(row.get("name", "")),
                direction=str(row.get("direction", "")),
                depot_location_x=float(row.get("depot_location_x", 0.0)),
                depot_location_y=float(row.get("depot_location_y", 0.0)),
            )
        return lines

    def _parse_transfer_zones(self, df: pd.DataFrame) -> Dict[str, TransferZone]:
        zones: Dict[str, TransferZone] = {}
        for _, row in df.iterrows():
            zone_id = str(row["zone_id"])
            zones[zone_id] = TransferZone(
                zone_id=zone_id,
                name=str(row.get("name", "")),
                dwelling_allowed=bool(row.get("dwelling_allowed", True)),
                max_capacity=int(row.get("max_capacity", 0)),
            )
        return zones

    def _parse_bus_stops(self, df: pd.DataFrame) -> Dict[str, BusStop]:
        stops: Dict[str, BusStop] = {}
        for _, row in df.iterrows():
            stop_id = str(row["stop_id"]).strip()
            stops[stop_id] = BusStop(
                stop_id=stop_id,
                zone_id=str(row["zone_id"]).strip(),
                name=str(row.get("name", "")),
                capacity=int(row.get("capacity", 0)),
                boarding_position=int(row.get("boarding_position", 1)),
            )
        return stops

    def _parse_assignments(self, df: pd.DataFrame) -> Dict[Tuple[str, str, str], LineStopAssignment]:
        assignments: Dict[Tuple[str, str, str], LineStopAssignment] = {}
        for _, row in df.iterrows():
            assignment = LineStopAssignment(
                line_id=str(row["line_id"]).strip(),
                zone_id=str(row["zone_id"]).strip(),
                stop_id=str(row["stop_id"]).strip(),
                stop_sequence=int(row["stop_sequence"]),
                dwell_time_allowed=bool(row.get("dwell_time_allowed", True)),
                max_dwelling_time=float(row.get("max_dwelling_time", 0.0)),
            )
            assignments[(assignment.line_id, assignment.zone_id, assignment.stop_id)] = assignment
        return assignments

    def _parse_travel_times(self, df: pd.DataFrame) -> List[TravelTimeEntry]:
        working_df = df.copy()
        if "travel_time_min" in working_df.columns and "travel_time" not in working_df.columns:
            working_df = working_df.rename(columns={"travel_time_min": "travel_time"})
        if "from_location" in working_df.columns and "from_stop_id" not in working_df.columns:
            working_df = working_df.rename(columns={"from_location": "from_stop_id"})

        travel_times: List[TravelTimeEntry] = []
        for _, row in working_df.iterrows():
            travel_times.append(
                TravelTimeEntry(
                    line_id=str(row["line_id"]).strip(),
                    from_stop_id=str(row.get("from_stop_id", "DEPOT") or "DEPOT").strip(),
                    to_stop_id=str(row.get("to_stop_id", "")).strip(),
                    travel_time=float(row["travel_time"]),
                )
            )
        return travel_times

    def _parse_sync_pairs(self, df: pd.DataFrame) -> List[SynchronizationPair]:
        pairs: List[SynchronizationPair] = []
        for _, row in df.iterrows():
            pairs.append(
                SynchronizationPair(
                    line_i=str(row["line_i"]),
                    line_j=str(row["line_j"]),
                    zone_id=str(row["zone_id"]),
                    min_sync_window=float(row["min_sync_window"]),
                    max_sync_window=float(row["max_sync_window"]),
                    sync_weight=float(row.get("sync_weight", 1.0)),
                    walking_time_between=float(row.get("walking_time_between", 0.0)),
                    sync_priority=str(row.get("sync_priority", "")),
                )
            )
        return pairs

    def _parse_service_constraints(self, df: pd.DataFrame) -> Dict[str, ServiceConstraint]:
        constraints: Dict[str, ServiceConstraint] = {}
        for _, row in df.iterrows():
            line_id = str(row["line_id"])
            constraints[line_id] = ServiceConstraint(
                line_id=line_id,
                first_trip_min_time=float(row.get("first_trip_min_time", 0.0)),
                first_trip_max_time=float(row.get("first_trip_max_time", 0.0)),
                last_trip_min_time=float(row.get("last_trip_min_time", 0.0)),
                last_trip_max_time=float(row.get("last_trip_max_time", 0.0)),
                min_dwell_time=float(row.get("min_dwell_time", 0.0)),
                max_dwell_time=float(row.get("max_dwell_time", 0.0)),
                max_cycle_time_increase=float(row.get("max_cycle_time_increase", 0.0)),
            )
        return constraints

    def _parse_model_parameters(self, df: pd.DataFrame) -> ModelParameters:
        params = {str(row["parameter"]): row["value"] for _, row in df.iterrows()}
        return ModelParameters(
            planning_horizon=float(params.get("planning_horizon", 0.0)),
            max_dwelling_time_global=float(params.get("max_dwelling_time_global", 0.0)),
            default_sync_window=float(params.get("default_sync_window", 0.0)),
            station_capacity_default=float(params.get("station_capacity_default", 0.0)),
        )

    def _build_zone_sequences(
        self, assignments: Dict[Tuple[str, str, str], LineStopAssignment]
    ) -> Dict[str, List[str]]:
        sequences: Dict[str, List[str]] = {}
        for (_, zone_id, _), assignment in assignments.items():
            seq = sequences.setdefault(assignment.line_id, [])
            seq.append((assignment.stop_sequence, zone_id))
        ordered: Dict[str, List[str]] = {}
        for line_id, values in sequences.items():
            sorted_zones = [z for _, z in sorted(values, key=lambda x: x[0])]
            seen = set()
            ordered[line_id] = []
            for zone in sorted_zones:
                if zone not in seen:
                    seen.add(zone)
                    ordered[line_id].append(zone)
        return ordered

    def _build_travel_time_map(
        self,
        travel_times: List[TravelTimeEntry],
        stop_zone_map: Dict[str, str],
        transfer_zones: Dict[str, TransferZone],
        known_lines: set[str],
        transfer_zone_lookup: Dict[str, TransferZone],
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
        travel_time_map: Dict[Tuple[str, str], float] = {}
        base_travel: Dict[str, float] = {}
        by_line: Dict[str, List[TravelTimeEntry]] = {}
        for entry in travel_times:
            by_line.setdefault(entry.line_id, []).append(entry)

        first_zone_lookup_debugged = False

        for line_id, entries in by_line.items():
            if line_id not in known_lines:
                logger.warning("Travel time provided for unknown line %s", line_id)
            cumulative: Dict[str, float] = {}
            # initialize with depot departures
            for entry in entries:
                if entry.from_stop_id.upper() == "DEPOT":
                    cumulative[entry.to_stop_id] = min(
                        cumulative.get(entry.to_stop_id, float("inf")), entry.travel_time
                    )

            progressed = True
            while progressed:
                progressed = False
                for entry in entries:
                    if entry.to_stop_id in cumulative:
                        continue
                    if entry.from_stop_id in cumulative:
                        cumulative[entry.to_stop_id] = cumulative[entry.from_stop_id] + entry.travel_time
                        progressed = True

            if not cumulative:
                logger.warning("No travel times could be derived for line %s", line_id)
                continue

            base_travel[line_id] = max(cumulative.values())
            for stop_id, time_val in cumulative.items():
                stop_id_clean = str(stop_id).strip()
                zone_id = stop_zone_map.get(stop_id_clean)
                if zone_id is None and stop_id_clean in transfer_zone_lookup:
                    zone_id = stop_id_clean
                if zone_id is None:
                    if not first_zone_lookup_debugged:
                        logger.debug(
                            "Stop '%s' in travel_times for line %s not found in bus_stops map (available sample: %s)",
                            stop_id_clean,
                            line_id,
                            list(stop_zone_map.keys())[:5],
                        )
                        first_zone_lookup_debugged = True
                    logger.warning(
                        "Stop %s missing zone mapping; skipping travel time", stop_id_clean
                    )
                    continue
                key = (line_id, zone_id)
                if key not in travel_time_map or time_val < travel_time_map[key]:
                    travel_time_map[key] = time_val

        return travel_time_map, base_travel

    def _compute_trip_counts(self, lines: Dict[str, BusLine], planning_horizon: float) -> Dict[str, int]:
        trip_counts: Dict[str, int] = {}
        for line_id, line in lines.items():
            headway = max(line.headway, 1e-6)
            trip_counts[line_id] = max(1, floor(planning_horizon / headway))
        return trip_counts


def load_bstdt_data(data_dir: str = "./data_complete") -> ModelData:
    loader = BSTDTDataLoader(data_dir)
    return loader.load_all_data()
