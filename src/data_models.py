from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

# --- 原始数据结构 (保持不变) ---
@dataclass
class Line:
    line_id: str
    headway: int
    frequency: int
    direction: str
    depot_x: float
    depot_y: float

@dataclass
class Stop:
    stop_id: str
    name: str
    lat: float
    lon: float
    zone_id: str  # 确保 Stop 知道自己属于哪个 Zone

@dataclass
class TravelTime:
    line_id: str
    from_stop: str
    to_stop: str
    time_min: float
    is_transfer: bool = False

@dataclass
class ServiceConstraint:
    line_id: str
    first_trip_min: int
    first_trip_max: int
    last_trip_min: int
    last_trip_max: int

@dataclass
class SyncPair:
    line_i: str
    line_j: str
    zone_id: str
    min_window: int
    max_window: int
    weight: float

@dataclass
class LineStopAssignment:
    line_id: str
    zone_id: str
    stop_id: str
    sequence: int
    max_dwell: int
    dwell_allowed: bool

@dataclass
class TransferZone:
    from_zone: str
    to_zone: str
    walking_time: float

# --- 核心升级：数据容器 ---
@dataclass
class BSTDTData:
    """
    不仅包含原始数据列表，还包含预计算的高速查找表 (Lookup Maps)
    """
    lines: List[Line] = field(default_factory=list)
    stops: List[Stop] = field(default_factory=list)
    travel_times: List[TravelTime] = field(default_factory=list)
    service_constraints: List[ServiceConstraint] = field(default_factory=list)
    sync_pairs: List[SyncPair] = field(default_factory=list)
    line_stop_assignments: List[LineStopAssignment] = field(default_factory=list)
    transfer_zones: List[TransferZone] = field(default_factory=list)

    # === 高速查找表 (O(1) Access) ===
    # 映射: (LineID, ZoneID) -> StopID
    # 解决: "L1 在 Z2 停在这个站，而 L2 在 Z2 停在那个站" 的问题
    map_line_zone_to_stop: Dict[Tuple[str, str], str] = field(default_factory=dict)

    # 映射: (FromZone, ToZone) -> WalkingTime
    # 解决: 换乘步行时间查找
    map_zone_transfer_time: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # 映射: LineID -> Line对象 (方便查 headway)
    map_lines: Dict[str, Line] = field(default_factory=dict)