"""Data structures for the BST-DT model.

These classes map the CSV inputs to typed Python objects and also carry
pre-computed helper dictionaries (travel times, zone sequences, trip
counts) used by the optimization model.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class BusLine:
    """Basic information for a bus line."""

    line_id: str
    headway: float
    frequency: int
    name: str = ""
    direction: str = ""
    depot_location_x: float = 0.0
    depot_location_y: float = 0.0


@dataclass
class BusStop:
    """A physical bus stop assigned to a transfer zone."""

    stop_id: str
    zone_id: str
    name: str
    capacity: int
    boarding_position: int


@dataclass
class TransferZone:
    """Transfer zone description."""

    zone_id: str
    name: str
    dwelling_allowed: bool
    max_capacity: int


@dataclass
class LineStopAssignment:
    """Mapping between lines and stops, including ordering along the line."""

    line_id: str
    zone_id: str
    stop_id: str
    stop_sequence: int
    dwell_time_allowed: bool
    max_dwelling_time: float


@dataclass
class TravelTimeEntry:
    """Travel time between successive stops (or depot to first stop)."""

    line_id: str
    from_stop_id: str
    to_stop_id: str
    travel_time: float


@dataclass
class SynchronizationPair:
    """Synchronization parameters for a pair of lines at a transfer zone."""

    line_i: str
    line_j: str
    zone_id: str
    min_sync_window: float
    max_sync_window: float
    sync_weight: float
    walking_time_between: float
    sync_priority: str


@dataclass
class ServiceConstraint:
    """Service time windows and dwell bounds for a single line."""

    line_id: str
    first_trip_min_time: float
    first_trip_max_time: float
    last_trip_min_time: float
    last_trip_max_time: float
    min_dwell_time: float
    max_dwell_time: float
    max_cycle_time_increase: float


@dataclass
class ModelParameters:
    """Global model-level parameters."""

    planning_horizon: float
    max_dwelling_time_global: float
    default_sync_window: float
    station_capacity_default: float


@dataclass
class ModelData:
    """Container for all model inputs and derived structures."""

    lines: Dict[str, BusLine]
    transfer_zones: Dict[str, TransferZone]
    bus_stops: Dict[str, BusStop]
    line_stop_assignments: Dict[Tuple[str, str, str], LineStopAssignment]
    travel_times: List[TravelTimeEntry]
    synchronization_pairs: List[SynchronizationPair]
    service_constraints: Dict[str, ServiceConstraint]
    model_parameters: ModelParameters

    # Derived structures populated by the loader
    travel_time_map: Dict[Tuple[str, str], float] = field(default_factory=dict)
    line_zone_sequence: Dict[str, List[str]] = field(default_factory=dict)
    num_trips_by_line: Dict[str, int] = field(default_factory=dict)
    base_travel_time: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure zone orderings keep unique entries and are consistent with assignments
        for line_id, seq in list(self.line_zone_sequence.items()):
            seen = set()
            unique_seq: List[str] = []
            for zone_id in seq:
                if zone_id not in seen:
                    seen.add(zone_id)
                    unique_seq.append(zone_id)
            self.line_zone_sequence[line_id] = unique_seq

    @property
    def planning_horizon(self) -> float:
        return self.model_parameters.planning_horizon

    def get_zone_sequence(self, line_id: str) -> List[str]:
        return self.line_zone_sequence.get(line_id, [])

    def get_travel_time_to_zone(self, line_id: str, zone_id: str) -> float:
        return self.travel_time_map.get((line_id, zone_id), 0.0)
