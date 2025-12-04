"""
BST-DT模型数据模型定义
包含所有核心数据类和数据结构
"""
import pandas as pd  # 确保有这行导入
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from enum import Enum


class Direction(Enum):
    """线路方向枚举"""
    OUTBOUND = "outbound"
    INBOUND = "inbound"
    CIRCULAR = "circular"


class DwellTimePolicy(Enum):
    """停留时间策略枚举"""
    NO_DWELL = "no_dwell"  # 不允许停留
    FIXED_DWELL = "fixed_dwell"  # 固定停留时间
    OPTIMIZABLE_DWELL = "optimizable_dwell"  # 可优化停留时间


@dataclass
class BusLine:
    """
    公交线路类
    对应lines.csv中的每条记录
    """
    line_id: str  # 线路唯一标识
    name: str = ""  # 线路名称（可选）
    headway: int = 30  # 发车间隔（分钟）
    frequency: int = 0  # 计划时段内班次数（根据T/h_i计算）
    direction: Direction = Direction.OUTBOUND  # 方向
    depot_location: Tuple[float, float] = (0.0, 0.0)  # 车场位置坐标
    is_night_service: bool = True  # 是否为夜间服务
    depot_location_x: float = 0.0  # ← 这里改了，匹配CSV列名
    depot_location_y: float = 0.0  # ← 这里改了，匹配CSV列名
    color: str = "#000000"  # 线路颜色（用于可视化）

    # 运行时计算的属性
    first_departure_time: Optional[float] = None  # X^i变量
    departure_times: List[float] = field(default_factory=list)  # 所有班次出发时间
    arrival_times: Dict[Tuple[str, int], float] = field(default_factory=dict)  # (zone_id, trip) -> 到达时间
    dwell_times: Dict[str, float] = field(default_factory=dict)  # zone_id -> 停留时间
    travel_time_to_zones: Dict[str, float] = field(default_factory=dict)  # 到各换乘区的旅行时间

    def __post_init__(self):
        """初始化后处理"""
        # 确保direction是Direction枚举
        if isinstance(self.direction, str):
            try:
                self.direction = Direction(self.direction.lower())
            except ValueError:
                self.direction = Direction.OUTBOUND

        # 初始化数据结构
        self.arrival_times = {}
        self.dwell_times = {}
        self.travel_time_to_zones = {}
        self.departure_times = []

    def calculate_departure_times(self, first_departure: float, planning_horizon: int) -> List[float]:
        """计算所有班次的出发时间"""
        self.departure_times = []
        current_time = first_departure
        trip_num = 0

        while current_time <= planning_horizon and trip_num < self.frequency:
            self.departure_times.append(current_time)
            current_time += self.headway
            trip_num += 1

        return self.departure_times

    def get_arrival_time(self, zone_id: str, trip_num: int) -> float:
        """获取指定班次在指定换乘区的到达时间"""
        return self.arrival_times.get((zone_id, trip_num), -1)

    def get_dwell_time(self, zone_id: str) -> float:
        """获取在指定换乘区的停留时间"""
        return self.dwell_times.get(zone_id, 0.0)

    @property
    def num_trips(self) -> int:
        """班次数别名，与frequency相同"""
        return self.frequency


@dataclass
class TransferZone:
    """换乘区"""
    zone_id: str
    name: str  # ← 新增字段
    dwelling_allowed: bool
    location_x: float = 0.0
    location_y: float = 0.0
    description: str = ""  # ← 新增字段
    is_major_transfer: bool = False
    max_capacity: int = 0  # ← 新增字段
    has_security: bool = False  # ← 新增字段
    walking_radius_m: float = 200.0  # ← 新增字段

    # 保持以下字段不变
    bus_stops: List['BusStop'] = field(default_factory=list)
    lines_serving: Set[str] = field(default_factory=set)


@dataclass
class BusStop:
    """
    公交站点类
    对应bus_stops.csv中的每条记录
    """
    stop_id: str  # 站点唯一标识
    zone_id: str  # 所属换乘区
    name: str = ""  # 站点名称
    capacity: int = 2  # 同时停靠最大车辆数（车位）
    boarding_position: int = 1  # 停靠位置序号
    location: Tuple[float, float] = (0.0, 0.0)  # 位置坐标
    has_shelter: bool = False  # 是否有候车亭
    is_accessible: bool = True  # 是否无障碍
    description: str = ""  # 描述信息

    # 运行时属性
    occupancy_schedule: List[Tuple[str, int, float, float]] = field(
        default_factory=list)  # (line_id, trip, arrival, departure)

    def __post_init__(self):
        """初始化后处理"""
        self.occupancy_schedule = []

    def add_occupancy(self, line_id: str, trip_num: int, arrival_time: float, departure_time: float):
        """添加车辆占用记录"""
        self.occupancy_schedule.append((line_id, trip_num, arrival_time, departure_time))
        # 按到达时间排序
        self.occupancy_schedule.sort(key=lambda x: x[2])

    def get_occupancy_at_time(self, time: float) -> List[Tuple[str, int]]:
        """获取指定时间点的占用车辆"""
        occupancy = []
        for line_id, trip_num, arrival, departure in self.occupancy_schedule:
            if arrival <= time <= departure:
                occupancy.append((line_id, trip_num))
        return occupancy

    def is_available_at_time(self, time: float, duration: float = 0) -> bool:
        """检查指定时间段是否有空闲车位"""
        current_occupancy = len(self.get_occupancy_at_time(time))

        if duration > 0:
            # 检查时间段内是否一直有空闲
            check_times = np.linspace(time, time + duration, num=10)
            for t in check_times:
                if len(self.get_occupancy_at_time(t)) >= self.capacity:
                    return False
            return current_occupancy < self.capacity
        else:
            return current_occupancy < self.capacity

    def clear_occupancy(self):
        """清空占用记录"""
        self.occupancy_schedule = []


@dataclass
class LineStopAssignment:
    """
    线路站点分配类
    对应line_stop_assignments.csv中的每条记录
    """
    line_id: str  # 线路ID
    zone_id: str  # 换乘区ID
    stop_id: str  # 站点ID
    stop_sequence: int  # 停靠顺序（在换乘区内的顺序）
    max_dwelling_time: float = 3.0  # 最大停留时间 L_b^i
    dwell_time_allowed: bool = True  # 是否允许停留
    dwell_time_policy: DwellTimePolicy = DwellTimePolicy.OPTIMIZABLE_DWELL  # 停留时间策略

    # 旅行时间信息
    travel_time_from_previous: float = 0.0  # 从前一站点到本站的旅行时间
    travel_time_to_next: float = 0.0  # 从本站到下一站点的旅行时间
    travel_time_from_depot: float = 0.0  # 从车场到本站的累计旅行时间

    # 优化变量（将在模型中创建）
    dwell_time_var: Optional[Any] = None  # 停留时间变量
    arrival_time_var: Optional[Any] = None  # 到达时间变量
    departure_time_var: Optional[Any] = None  # 出发时间变量

    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.dwell_time_policy, str):
            try:
                self.dwell_time_policy = DwellTimePolicy(self.dwell_time_policy.lower())
            except ValueError:
                self.dwell_time_policy = DwellTimePolicy.OPTIMIZABLE_DWELL

    @property
    def key(self) -> Tuple[str, str, str]:
        """唯一标识键"""
        return (self.line_id, self.zone_id, self.stop_id)

    @property
    def min_dwelling_time(self) -> float:
        """最小停留时间"""
        if self.dwell_time_policy == DwellTimePolicy.NO_DWELL:
            return 0.0
        elif self.dwell_time_policy == DwellTimePolicy.FIXED_DWELL:
            return self.max_dwelling_time
        else:  # OPTIMIZABLE_DWELL
            return 0.0

    @property
    def is_dwell_optimizable(self) -> bool:
        """停留时间是否可优化"""
        return self.dwell_time_allowed and self.dwell_time_policy == DwellTimePolicy.OPTIMIZABLE_DWELL


@dataclass
class TravelTimeSegment:
    """
    旅行时间段类
    对应travel_times.csv中的每条记录
    """
    line_id: str  # 线路ID
    from_stop_id: str  # 起点站ID（或"DEPOT"）
    to_stop_id: str  # 终点站ID
    travel_time: float  # 旅行时间（分钟）
    distance: float = 0.0  # 距离（公里，可选）
    is_transfer_segment: bool = False  # 是否为换乘段
    reliability_factor: float = 1.0  # 可靠性因子（1.0表示完全可靠）

    def __post_init__(self):
        """初始化后处理"""
        # 确保travel_time为正数
        self.travel_time = max(0.0, self.travel_time)

    @property
    def key(self) -> Tuple[str, str, str]:
        """唯一标识键"""
        return (self.line_id, self.from_stop_id, self.to_stop_id)

    @property
    def is_depot_segment(self) -> bool:
        """是否从车场出发"""
        return self.from_stop_id.upper() == "DEPOT"

    @property
    def effective_travel_time(self) -> float:
        """有效旅行时间（考虑可靠性）"""
        return self.travel_time * self.reliability_factor


@dataclass
class SynchronizationPair:
    """
    同步参数类
    对应synchronization_pairs.csv中的每条记录
    """
    line_i: str  # 同步线路i
    line_j: str  # 同步线路j
    zone_id: str  # 同步换乘区
    min_sync_window: float = 0.0  # 最小同步时间窗口 W_ij^b
    max_sync_window: float = 5.0  # 最大同步时间窗口 W̄_ij^b
    sync_weight: float = 1.0  # 同步权重 IMP_ij^b
    walking_time: float = 2.0  # 步行时间（分钟）
    is_bidirectional: bool = True  # 是否双向同步
    priority: int = 1  # 优先级（1-10，越高越优先）
    walking_time_between: float = 0  # 新增字段
    sync_priority: str = "normal"  # 新增字段

    # 计算属性
    sync_variables: Dict[Tuple[int, int], Any] = field(default_factory=dict)  # (trip_i, trip_j) -> 同步变量
    sync_status: Dict[Tuple[int, int], bool] = field(default_factory=dict)  # (trip_i, trip_j) -> 是否同步

    def __post_init__(self):
        """初始化后处理"""
        self.sync_variables = {}
        self.sync_status = {}

    @property
    def key(self) -> Tuple[str, str, str]:
        """唯一标识键"""
        if self.line_i <= self.line_j:
            return (self.line_i, self.line_j, self.zone_id)
        else:
            return (self.line_j, self.line_i, self.zone_id)

    @property
    def sync_window_width(self) -> float:
        """同步窗口宽度"""
        return self.max_sync_window - self.min_sync_window

    @property
    def is_valid(self) -> bool:
        """同步对是否有效"""
        return (self.line_i != self.line_j and
                self.min_sync_window <= self.max_sync_window and
                self.sync_weight > 0)

    def set_sync_status(self, trip_i: int, trip_j: int, status: bool):
        """设置同步状态"""
        self.sync_status[(trip_i, trip_j)] = status

    def get_sync_status(self, trip_i: int, trip_j: int) -> bool:
        """获取同步状态"""
        return self.sync_status.get((trip_i, trip_j), False)

    def get_all_sync_pairs(self) -> List[Tuple[int, int]]:
        """获取所有同步班次对"""
        return list(self.sync_status.keys())


@dataclass
class ServiceConstraint:
    """
    服务约束类
    对应service_constraints.csv中的每条记录
    """
    line_id: str  # 线路ID
    first_trip_min_time: float = 0.0  # 首班车最早出发时间
    first_trip_max_time: float = 30.0  # 首班车最晚出发时间
    last_trip_min_time: float = 200.0  # 末班车最早出发时间
    last_trip_max_time: float = 239.0  # 末班车最晚出发时间
    min_headway: float = 10.0  # 最小发车间隔
    max_headway: float = 60.0  # 最大发车间隔
    max_total_dwell_time: float = 10.0  # 最大总停留时间

    def __post_init__(self):
        """初始化后处理"""
        # 确保时间范围有效
        self.first_trip_min_time = max(0.0, self.first_trip_min_time)
        self.first_trip_max_time = max(self.first_trip_min_time, self.first_trip_max_time)
        self.last_trip_min_time = max(self.first_trip_max_time, self.last_trip_min_time)
        self.last_trip_max_time = max(self.last_trip_min_time, self.last_trip_max_time)

    @property
    def first_trip_time_window(self) -> Tuple[float, float]:
        """首班车时间窗口"""
        return (self.first_trip_min_time, self.first_trip_max_time)

    @property
    def last_trip_time_window(self) -> Tuple[float, float]:
        """末班车时间窗口"""
        return (self.last_trip_min_time, self.last_trip_max_time)

    def is_valid_first_time(self, time: float) -> bool:
        """检查首班车时间是否有效"""
        return self.first_trip_min_time <= time <= self.first_trip_max_time

    def is_valid_last_time(self, time: float) -> bool:
        """检查末班车时间是否有效"""
        return self.last_trip_min_time <= time <= self.last_trip_max_time


@dataclass
class ModelParameters:
    """
    模型参数类
    对应model_parameters.csv中的参数
    """
    planning_horizon: int = 239  # 计划时段长度 T（分钟）
    max_dwelling_time: float = 3.0  # 最大停留时间 L
    min_dwelling_time: float = 0.0  # 最小停留时间
    big_m_value: float = 10000.0  # Big-M参数
    time_step: float = 1.0  # 时间步长（分钟）
    sync_window_default: Tuple[float, float] = (0.0, 5.0)  # 默认同步窗口

    # 求解器参数
    time_limit: int = 28800  # 时间限制（秒）
    mip_gap: float = 0.01  # MIP gap
    thread_count: int = 0  # 线程数（0=自动）

    # 模型选项
    use_bus_capacity_constraints: bool = True  # 是否使用站点容量约束
    use_dwelling_time_optimization: bool = True  # 是否优化停留时间
    use_valid_inequalities: bool = True  # 是否使用有效不等式
    use_periodic_constraints: bool = True  # 是否使用周期性约束

    def __post_init__(self):
        """初始化后处理"""
        # 确保参数合理
        self.planning_horizon = max(1, self.planning_horizon)
        self.max_dwelling_time = max(0.0, self.max_dwelling_time)
        self.min_dwelling_time = max(0.0, min(self.min_dwelling_time, self.max_dwelling_time))
        self.big_m_value = max(100.0, self.big_m_value)
        self.time_step = max(0.1, self.time_step)

        # 确保同步窗口有效
        if self.sync_window_default[0] > self.sync_window_default[1]:
            self.sync_window_default = (0.0, 5.0)

    @classmethod
    def from_dataframe(cls, df):
        """从DataFrame创建参数对象"""
        if df is None or df.empty:
            return cls()

        # 获取类的构造函数参数
        import inspect
        # 使用 cls.__init__ 而不是 cls
        signature = inspect.signature(cls.__init__)
        valid_params = list(signature.parameters.keys())
        # 移除'self'参数
        if 'self' in valid_params:
            valid_params.remove('self')

        params_dict = {}
        ignored_params = []

        for _, row in df.iterrows():
            try:
                param_name = str(row['parameter']).strip()
                value = row['value']

                # 检查参数名是否是类的有效参数
                if param_name in valid_params:
                    # 尝试转换为适当类型
                    value_str = str(value).strip()

                    try:
                        # 尝试转换为int
                        params_dict[param_name] = int(value_str)
                    except ValueError:
                        try:
                            # 尝试转换为float
                            params_dict[param_name] = float(value_str)
                        except ValueError:
                            # 如果转换失败，保持原样（可能是字符串或布尔值）
                            if value_str.lower() == 'true':
                                params_dict[param_name] = True
                            elif value_str.lower() == 'false':
                                params_dict[param_name] = False
                            else:
                                params_dict[param_name] = value_str
                else:
                    # 记录忽略的参数
                    ignored_params.append(param_name)

            except KeyError as e:
                print(f"数据行缺少必要字段: {e}")
                continue

        if ignored_params:
            print(f"注意: 忽略了 {len(ignored_params)} 个未知参数: {ignored_params}")

        # 创建对象
        try:
            return cls(**params_dict)
        except TypeError as e:
            print(f"创建ModelParameters对象失败: {e}")
            print(f"有效参数: {valid_params}")
            print(f"提供的参数: {list(params_dict.keys())}")
            # 返回默认对象
            return cls()


@dataclass
class ModelData:
    """
    统一的数据容器类
    包含所有BST-DT模型需要的数据
    """
    # 基础数据
    lines: Dict[str, BusLine]
    transfer_zones: Dict[str, TransferZone]
    bus_stops: Dict[str, BusStop]
    line_stop_assignments: Dict[Tuple[str, str, str], LineStopAssignment]
    travel_time_segments: List[TravelTimeSegment]
    sync_pairs: Dict[Tuple[str, str, str], SynchronizationPair]
    service_constraints: Dict[str, ServiceConstraint]
    model_parameters: ModelParameters

    # 计算得到的属性
    line_travel_times: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (line_id, zone_id) -> 旅行时间
    zone_line_map: Dict[str, List[str]] = field(default_factory=dict)  # zone_id -> 经过的线路列表
    line_zone_map: Dict[str, List[str]] = field(default_factory=dict)  # line_id -> 经过的换乘区列表

    # 运行时数据
    optimization_variables: Dict[str, Any] = field(default_factory=dict)  # 优化变量
    optimization_constraints: Dict[str, Any] = field(default_factory=dict)  # 优化约束
    solution: Optional[Dict] = None  # 求解结果

    def __post_init__(self):
        """初始化后处理"""
        # 初始化字典
        self.line_travel_times = {}
        self.zone_line_map = {}
        self.line_zone_map = {}
        self.optimization_variables = {}
        self.optimization_constraints = {}

        # 计算线路和换乘区的映射关系
        self._build_mappings()
        self._calculate_travel_times()

        # 重要：验证并过滤无效的同步对
        self._validate_and_filter_sync_pairs()

    def _build_mappings(self):
        """构建线路和换乘区的映射关系"""
        # 初始化映射字典
        self.zone_line_map = {zone_id: [] for zone_id in self.transfer_zones.keys()}
        self.line_zone_map = {line_id: [] for line_id in self.lines.keys()}

        # 通过line_stop_assignments构建映射
        for assignment in self.line_stop_assignments.values():
            line_id = assignment.line_id
            zone_id = assignment.zone_id

            # 添加到line_zone_map（确保不重复）
            if zone_id not in self.line_zone_map[line_id]:
                self.line_zone_map[line_id].append(zone_id)

            # 添加到zone_line_map（确保不重复）
            if line_id not in self.zone_line_map[zone_id]:
                self.zone_line_map[zone_id].append(line_id)

        # 更新TransferZone对象中的lines_serving
        for zone_id, lines in self.zone_line_map.items():
            if zone_id in self.transfer_zones:
                self.transfer_zones[zone_id].lines_serving = set(lines)

        # 排序以便调试
        for line_id in self.line_zone_map:
            self.line_zone_map[line_id].sort()

    def _validate_and_filter_sync_pairs(self):
        """验证并过滤无效的同步对"""
        import logging
        logger = logging.getLogger(__name__)

        logger.info("验证同步对有效性...")

        invalid_pairs = []
        valid_pairs = {}

        # 先确保映射已经建立
        if not self.line_zone_map:
            self._build_mappings()

        for (line_i, line_j, zone_id), pair in self.sync_pairs.items():
            # 检查线路是否存在
            if line_i not in self.lines:
                logger.warning(f"线路 {line_i} 不存在于线路数据中，跳过同步对 {line_i}-{line_j}@{zone_id}")
                invalid_pairs.append((line_i, line_j, zone_id))
                continue

            if line_j not in self.lines:
                logger.warning(f"线路 {line_j} 不存在于线路数据中，跳过同步对 {line_i}-{line_j}@{zone_id}")
                invalid_pairs.append((line_i, line_j, zone_id))
                continue

            # 检查换乘区是否存在
            if zone_id not in self.transfer_zones:
                logger.warning(f"换乘区 {zone_id} 不存在于换乘区数据中，跳过同步对 {line_i}-{line_j}@{zone_id}")
                invalid_pairs.append((line_i, line_j, zone_id))
                continue

            # 检查两条线路是否都经过该换乘区
            line_i_zones = self.get_zones_for_line(line_i)
            line_j_zones = self.get_zones_for_line(line_j)

            if zone_id not in line_i_zones:
                logger.warning(f"线路 {line_i} 不经过换乘区 {zone_id}，其经过的换乘区: {line_i_zones}")
                invalid_pairs.append((line_i, line_j, zone_id))
                continue

            if zone_id not in line_j_zones:
                logger.warning(f"线路 {line_j} 不经过换乘区 {zone_id}，其经过的换乘区: {line_j_zones}")
                invalid_pairs.append((line_i, line_j, zone_id))
                continue

            # 所有检查通过，保留同步对
            valid_pairs[(line_i, line_j, zone_id)] = pair

        # 更新同步对字典
        self.sync_pairs = valid_pairs

        if invalid_pairs:
            logger.warning(f"过滤了 {len(invalid_pairs)} 个无效的同步对")
            logger.info(f"保留了 {len(valid_pairs)} 个有效的同步对")

        # 重新构建映射关系（因为可能过滤了同步对）
        self._build_mappings()

    def _calculate_travel_times(self):
        """计算线路到各换乘区的旅行时间"""
        # 按线路组织旅行时间段
        line_segments = {}
        for segment in self.travel_time_segments:
            if segment.line_id not in line_segments:
                line_segments[segment.line_id] = []
            line_segments[segment.line_id].append(segment)

        # 为每条线路计算到换乘区的旅行时间
        for line_id, segments in line_segments.items():
            # 按起点排序（假设有顺序）
            # 从DEPOT开始排序
            depot_segments = [s for s in segments if s.is_depot_segment]
            other_segments = [s for s in segments if not s.is_depot_segment]

            # 计算累计旅行时间
            cumulative_time = 0.0

            # 处理从车场出发的段
            for segment in depot_segments:
                cumulative_time += segment.travel_time

                # 如果到达的是换乘区的站点，记录旅行时间
                to_stop_id = segment.to_stop_id
                if to_stop_id in self.bus_stops:
                    zone_id = self.bus_stops[to_stop_id].zone_id
                    self.line_travel_times[(line_id, zone_id)] = cumulative_time

            # 处理其他段
            for segment in other_segments:
                cumulative_time += segment.travel_time

                # 如果到达的是换乘区的站点，记录旅行时间
                to_stop_id = segment.to_stop_id
                if to_stop_id in self.bus_stops:
                    zone_id = self.bus_stops[to_stop_id].zone_id
                    self.line_travel_times[(line_id, zone_id)] = cumulative_time

    def get_lines_for_zone(self, zone_id: str) -> List[str]:
        """获取经过指定换乘区的所有线路"""
        return self.zone_line_map.get(zone_id, [])

    def get_zones_for_line(self, line_id: str) -> List[str]:
        """获取指定线路经过的所有换乘区"""
        return self.line_zone_map.get(line_id, [])

    def get_stops_for_line_zone(self, line_id: str, zone_id: str) -> List[BusStop]:
        """获取线路在指定换乘区停靠的所有站点"""
        stops = []
        for assignment in self.line_stop_assignments.values():
            if assignment.line_id == line_id and assignment.zone_id == zone_id:
                stop_id = assignment.stop_id
                if stop_id in self.bus_stops:
                    stops.append(self.bus_stops[stop_id])
        return stops

    def get_sync_pairs_for_zone(self, zone_id: str) -> List[SynchronizationPair]:
        """获取指定换乘区的所有同步对"""
        pairs = []
        for pair in self.sync_pairs.values():
            if pair.zone_id == zone_id:
                pairs.append(pair)
        return pairs

    def get_line_frequency(self, line_id: str) -> int:
        """获取线路的班次数"""
        if line_id in self.lines:
            return self.lines[line_id].frequency
        return 0

    def get_total_possible_syncs(self) -> int:
        """计算理论上最大的同步次数"""
        total = 0
        for pair in self.sync_pairs.values():
            line_i_freq = self.get_line_frequency(pair.line_i)
            line_j_freq = self.get_line_frequency(pair.line_j)
            total += line_i_freq * line_j_freq
        return total

    @property
    def num_lines(self) -> int:
        """线路数量"""
        return len(self.lines)

    @property
    def num_zones(self) -> int:
        """换乘区数量"""
        return len(self.transfer_zones)

    @property
    def num_stops(self) -> int:
        """站点数量"""
        return len(self.bus_stops)

    @property
    def num_sync_pairs(self) -> int:
        """同步对数量"""
        return len(self.sync_pairs)

    @property
    def planning_horizon(self) -> int:
        """计划时段长度（分钟）"""
        return self.model_parameters.planning_horizon

    def validate_data(self) -> List[str]:
        """验证数据完整性，返回错误消息列表"""
        errors = []

        # 检查线路频率
        for line_id, line in self.lines.items():
            if line.frequency <= 0:
                errors.append(f"线路 {line_id} 的班次数必须为正数")

        # 检查换乘区站点
        for zone_id, zone in self.transfer_zones.items():
            if len(zone.bus_stops) == 0:
                errors.append(f"换乘区 {zone_id} 没有站点")

        # 检查同步对
        for pair_key, pair in self.sync_pairs.items():
            if pair.line_i not in self.lines:
                errors.append(f"同步对 {pair_key} 中的线路 {pair.line_i} 不存在")
            if pair.line_j not in self.lines:
                errors.append(f"同步对 {pair_key} 中的线路 {pair.line_j} 不存在")
            if pair.zone_id not in self.transfer_zones:
                errors.append(f"同步对 {pair_key} 中的换乘区 {pair.zone_id} 不存在")

            # 检查线路是否经过换乘区
            line_i_zones = self.get_zones_for_line(pair.line_i)
            line_j_zones = self.get_zones_for_line(pair.line_j)

            if pair.zone_id not in line_i_zones:
                errors.append(f"同步对 {pair_key} 中的线路 {pair.line_i} 不经过换乘区 {pair.zone_id}")
            if pair.zone_id not in line_j_zones:
                errors.append(f"同步对 {pair_key} 中的线路 {pair.line_j} 不经过换乘区 {pair.zone_id}")

        return errors

    def get_line_zone_sequence(self, line_id: str) -> List[str]:
        """获取线路经过换乘区的顺序"""
        zones = self.get_zones_for_line(line_id)

        # 如果没有顺序信息，按默认顺序
        if not zones:
            return []

        # 尝试从line_stop_assignments中获取顺序
        zone_assignments = []
        for (l_id, z_id, s_id), assignment in self.line_stop_assignments.items():
            if l_id == line_id:
                zone_assignments.append((z_id, assignment.stop_sequence))

        # 按stop_sequence排序
        zone_assignments.sort(key=lambda x: x[1])

        # 提取zone_id并按顺序返回
        ordered_zones = [zone_id for zone_id, _ in zone_assignments]

        # 去重（保持顺序）
        seen = set()
        unique_zones = []
        for zone in ordered_zones:
            if zone not in seen:
                seen.add(zone)
                unique_zones.append(zone)

        return unique_zones

    def print_summary(self):
        """打印数据摘要"""
        print("=" * 60)
        print("BST-DT模型数据摘要")
        print("=" * 60)
        print(f"线路数量: {self.num_lines}")
        print(f"换乘区数量: {self.num_zones}")
        print(f"站点数量: {self.num_stops}")
        print(f"有效同步对数量: {self.num_sync_pairs}")
        print(f"计划时段长度: {self.planning_horizon} 分钟")

        # 线路信息
        print(f"\n线路详细信息:")
        for line_id, line in self.lines.items():
            zones = self.get_zones_for_line(line_id)
            print(f"  {line_id}: 发车间隔={line.headway}分钟, "
                  f"班次数={line.frequency}, 方向={line.direction}, "
                  f"经过的换乘区={zones}")

        # 换乘区信息
        print(f"\n换乘区详细信息:")
        for zone_id, zone in self.transfer_zones.items():
            lines = self.get_lines_for_zone(zone_id)
            print(f"  {zone_id}: 允许停留={zone.dwelling_allowed}, "
                  f"站点数={len(zone.bus_stops)}, "
                  f"经过的线路数={len(lines)}")

        # 同步对信息
        print(f"\n同步对详细信息:")
        for (line_i, line_j, zone_id), pair in list(self.sync_pairs.items())[:10]:  # 只显示前10个
            print(f"  {line_i}-{line_j}@{zone_id}: "
                  f"窗口=[{pair.min_sync_window}, {pair.max_sync_window}], "
                  f"权重={pair.sync_weight}")

        if len(self.sync_pairs) > 10:
            print(f"  ... 还有 {len(self.sync_pairs) - 10} 个同步对")

        print("=" * 60)


@dataclass
class Solution:
    """求解结果类"""
    objective_value: float = 0.0  # 目标函数值
    total_synchronizations: int = 0  # 总同步次数
    weighted_synchronizations: float = 0.0  # 加权同步次数
    solve_time: float = 0.0  # 求解时间（秒）
    mip_gap: float = 0.0  # MIP gap
    status: str = "UNKNOWN"  # 求解状态

    # 详细结果
    timetables: Dict[str, Dict] = field(default_factory=dict)  # 时刻表
    dwell_times: Dict[str, Dict[str, float]] = field(default_factory=dict)  # 停留时间
    sync_results: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = field(default_factory=dict)  # 同步结果

    # 性能指标
    baseline_syncs: int = 0  # 基准同步次数
    improvement_percentage: float = 0.0  # 改进百分比

    def calculate_metrics(self):
        """计算性能指标"""
        if self.baseline_syncs > 0:
            self.improvement_percentage = (
                    (self.total_synchronizations - self.baseline_syncs) /
                    self.baseline_syncs * 100
            )

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'objective_value': self.objective_value,
            'total_synchronizations': self.total_synchronizations,
            'weighted_synchronizations': self.weighted_synchronizations,
            'solve_time': self.solve_time,
            'mip_gap': self.mip_gap,
            'status': self.status,
            'improvement_percentage': self.improvement_percentage
        }

    def print_summary(self):
        """打印结果摘要"""
        print("=" * 60)
        print("BST-DT模型求解结果摘要")
        print("=" * 60)
        print(f"求解状态: {self.status}")
        print(f"目标函数值: {self.objective_value:.2f}")
        print(f"总同步次数: {self.total_synchronizations}")
        print(f"加权同步次数: {self.weighted_synchronizations:.2f}")
        print(f"求解时间: {self.solve_time:.2f}秒")
        print(f"MIP Gap: {self.mip_gap:.4%}")
        if self.baseline_syncs > 0:
            print(f"基准同步次数: {self.baseline_syncs}")
            print(f"改进百分比: {self.improvement_percentage:.1f}%")
        print("=" * 60)


# 为了方便使用，定义一个数据容器创建函数
def create_model_data() -> ModelData:
    """创建空的ModelData实例"""
    return ModelData(
        lines={},
        transfer_zones={},
        bus_stops={},
        line_stop_assignments={},
        travel_time_segments=[],
        sync_pairs={},
        service_constraints={},
        model_parameters=ModelParameters()
    )