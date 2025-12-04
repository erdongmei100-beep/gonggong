"""
BST-DT模型配置模块
负责管理所有配置参数，包括求解器参数、模型参数和运行参数
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class SolverConfig:
    """求解器配置"""

    # 基本求解参数
    time_limit: int = 28800  # 8小时（秒）
    mip_gap: float = 0.01  # 1%的gap
    thread_count: int = 4  # 使用线程数
    output_flag: bool = True  # 是否显示求解过程

    # 高级求解参数
    presolve: int = 2  # 预处理级别（0=关闭，1=保守，2=积极）
    heuristics: float = 0.05  # 启发式搜索比例
    cuts: int = 2  # 切割生成级别（0=关闭，1=保守，2=积极，3=非常积极）
    solution_limit: int = 20000000  # 最大解数量

    # 数值参数
    numeric_focus: int = 0  # 数值稳定性（0=自动，1=中等，2=高，3=非常高）
    feasibility_tol: float = 1e-6  # 可行性容差
    optimality_tol: float = 1e-6  # 最优性容差

    # 日志设置
    log_file: str = ""  # 日志文件路径，空字符串表示不保存
    log_to_console: bool = True  # 是否输出到控制台

    def to_gurobi_params(self) -> Dict[str, Any]:
        """转换为Gurobi参数字典"""
        return {
            'TimeLimit': self.time_limit,
            'MIPGap': self.mip_gap,
            'Threads': self.thread_count,
            'OutputFlag': 1 if self.output_flag else 0,
            'Presolve': self.presolve,
            'Heuristics': self.heuristics,
            'Cuts': self.cuts,
            'SolutionLimit': self.solution_limit,
            'NumericFocus': self.numeric_focus,
            'FeasibilityTol': self.feasibility_tol,
            'OptimalityTol': self.optimality_tol,
            'LogFile': self.log_file,
            'LogToConsole': 1 if self.log_to_console else 0
        }


@dataclass
class ModelConfig:
    """模型配置"""

    # 核心模型参数
    planning_horizon: int = 239  # 计划时段长度（分钟）
    max_dwelling_time: int = 3  # 最大停留时间（分钟）
    big_m_value: float = 10000.0  # Big-M参数

    # 同步窗口参数
    default_min_sync_window: float = 0.0  # 默认最小同步窗口（分钟）
    default_max_sync_window: float = 5.0  # 默认最大同步窗口（分钟）
    default_sync_weight: float = 1.0  # 默认同步权重

    # 约束参数
    max_cycle_time_increase: float = 0.1  # 最大周期时间增加比例（10%）
    min_headway: int = 10  # 最小发车间隔（分钟）
    max_headway: int = 60  # 最大发车间隔（分钟）

    # 边境条件参数
    border_transition_period: int = 60  # 日间夜间过渡时间（分钟）
    min_departure_gap: int = 5  # 最小发车间隔（分钟）

    # 站点参数
    default_bus_stop_capacity: int = 2  # 默认站点容量
    min_dwelling_time: float = 0.0  # 最小停留时间
    max_dwelling_time_per_trip: float = 10.0  # 每个班次最大总停留时间

    def validate(self):
        """验证配置参数的有效性"""
        assert self.planning_horizon > 0, "计划时段必须大于0"
        assert self.max_dwelling_time >= 0, "最大停留时间不能为负数"
        assert self.big_m_value > 0, "Big-M参数必须大于0"
        assert 0 <= self.default_min_sync_window < self.default_max_sync_window, "同步窗口参数无效"
        assert self.default_sync_weight >= 0, "同步权重不能为负数"
        assert 0 <= self.max_cycle_time_increase <= 1, "周期时间增加比例必须在0-1之间"
        assert self.min_headway > 0 and self.max_headway >= self.min_headway, "发车间隔参数无效"


@dataclass
class ConstraintConfig:
    """约束配置"""

    # 约束开关
    use_bus_capacity_constraints: bool = True  # 是否使用站点容量约束
    use_max_dwell_time_constraints: bool = True  # 是否使用最大停留时间约束
    use_border_condition_constraints: bool = True  # 是否使用边境条件约束
    use_valid_inequalities: bool = True  # 是否使用有效不等式
    use_periodic_constraints: bool = True  # 是否使用周期性约束

    # 有效不等式配置
    use_sync_inequalities: bool = True  # 是否使用同步不等式
    use_headway_inequalities: bool = True  # 是否使用发车间隔不等式
    use_left_shift_constraints: bool = True  # 是否使用左移约束

    # 约束参数
    enforce_exact_dwell_times: bool = False  # 是否强制精确停留时间
    allow_zero_dwell_time: bool = True  # 是否允许零停留时间
    strict_sync_window: bool = False  # 是否严格执行同步窗口

    def get_active_constraints(self) -> Dict[str, bool]:
        """获取活动的约束列表"""
        return {
            'bus_capacity': self.use_bus_capacity_constraints,
            'max_dwell_time': self.use_max_dwell_time_constraints,
            'border_condition': self.use_border_condition_constraints,
            'valid_inequalities': self.use_valid_inequalities,
            'periodic_constraints': self.use_periodic_constraints,
            'sync_inequalities': self.use_sync_inequalities,
            'headway_inequalities': self.use_headway_inequalities,
            'left_shift_constraints': self.use_left_shift_constraints
        }


@dataclass
class OutputConfig:
    """输出配置"""

    # 输出开关
    save_results: bool = True  # 是否保存结果
    save_timetables: bool = True  # 是否保存时刻表
    save_dwell_times: bool = True  # 是否保存停留时间
    save_synchronizations: bool = True  # 是否保存同步信息
    save_model_stats: bool = True  # 是否保存模型统计信息

    # 输出格式
    output_format: str = 'json'  # 输出格式：json, csv, both
    compress_output: bool = False  # 是否压缩输出文件
    pretty_print: bool = True  # 是否美化输出

    # 输出路径
    results_dir: str = "./results"
    results_prefix: str = "bstdt_results"
    create_timestamp_dir: bool = True  # 是否创建时间戳目录

    # 日志输出
    log_level: str = 'INFO'  # 日志级别：DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True  # 是否记录到文件
    log_file_path: str = "./logs/bstdt.log"  # 日志文件路径

    # 可视化输出
    generate_plots: bool = False  # 是否生成图表
    plot_format: str = 'png'  # 图表格式：png, pdf, svg
    plot_dpi: int = 300  # 图表分辨率

    def get_output_path(self, instance_name: str) -> Path:
        """获取输出路径"""
        base_path = Path(self.results_dir)

        if self.create_timestamp_dir:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = base_path / f"{instance_name}_{timestamp}"
        else:
            output_path = base_path / instance_name

        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


@dataclass
class DataConfig:
    """数据配置"""

    # 数据文件路径
    data_dir: str = "./data_complete"

    # 数据文件名称
    lines_file: str = "lines.csv"
    transfer_zones_file: str = "transfer_zones.csv"
    bus_stops_file: str = "bus_stops.csv"
    line_stop_assignments_file: str = "line_stop_assignments.csv"
    travel_times_file: str = "travel_times.csv"
    synchronization_pairs_file: str = "synchronization_pairs.csv"
    model_parameters_file: str = "model_parameters.csv"
    service_constraints_file: str = "service_constraints.csv"

    # 数据验证
    validate_data: bool = True  # 是否验证数据
    strict_validation: bool = False  # 是否严格验证
    fix_data_errors: bool = True  # 是否自动修复数据错误

    # 数据预处理
    preprocess_travel_times: bool = True  # 是否预处理旅行时间
    calculate_missing_times: bool = True  # 是否计算缺失的旅行时间
    fill_missing_values: bool = True  # 是否填充缺失值

    # 缓存设置
    use_cache: bool = True  # 是否使用缓存
    cache_dir: str = "./cache"
    cache_expiry: int = 3600  # 缓存过期时间（秒）

    def get_file_paths(self) -> Dict[str, Path]:
        """获取所有数据文件路径"""
        data_path = Path(self.data_dir)
        return {
            'lines': data_path / self.lines_file,
            'transfer_zones': data_path / self.transfer_zones_file,
            'bus_stops': data_path / self.bus_stops_file,
            'line_stop_assignments': data_path / self.line_stop_assignments_file,
            'travel_times': data_path / self.travel_times_file,
            'synchronization_pairs': data_path / self.synchronization_pairs_file,
            'model_parameters': data_path / self.model_parameters_file,
            'service_constraints': data_path / self.service_constraints_file,
        }


@dataclass
class BSTDTConfig:
    """BST-DT模型总配置"""

    # 配置版本
    version: str = "1.0.0"

    # 实例配置
    instance_name: str = "default_instance"
    instance_type: str = "small"  # small, medium, large
    description: str = "BST-DT模型配置"

    # 子配置
    solver: SolverConfig = field(default_factory=SolverConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # 运行时配置
    verbose: bool = True  # 是否详细输出
    debug_mode: bool = False  # 是否调试模式
    save_config: bool = True  # 是否保存配置
    random_seed: int = 42  # 随机种子

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BSTDTConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> 'BSTDTConfig':
        """从JSON文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BSTDTConfig':
        """从字典创建配置"""
        # 创建基本配置
        config = cls()

        # 更新顶层配置
        for key, value in config_dict.items():
            if hasattr(config, key):
                # 如果值是字典且对应的是数据类，递归处理
                if isinstance(value, dict) and key in ['solver', 'model', 'constraints', 'output', 'data']:
                    sub_config_class = getattr(config, key).__class__
                    setattr(config, key, sub_config_class(**value))
                else:
                    setattr(config, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        config_dict = {}

        # 基本属性
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue

            if isinstance(value, (SolverConfig, ModelConfig, ConstraintConfig, OutputConfig, DataConfig)):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value

        return config_dict

    def to_yaml(self, yaml_path: str):
        """保存为YAML文件"""
        config_dict = self.to_dict()

        # 确保目录存在
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def to_json(self, json_path: str):
        """保存为JSON文件"""
        config_dict = self.to_dict()

        # 确保目录存在
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def validate(self):
        """验证完整配置"""
        # 验证子配置
        self.model.validate()

        # 验证数据文件是否存在
        file_paths = self.data.get_file_paths()
        for name, path in file_paths.items():
            if not path.exists():
                if self.data.strict_validation:
                    raise FileNotFoundError(f"数据文件不存在: {path}")
                elif self.verbose:
                    print(f"警告: 数据文件不存在: {path}")

        # 验证输出目录
        if self.output.save_results:
            output_path = Path(self.output.results_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # 验证日志目录
        if self.output.log_to_file:
            log_path = Path(self.output.log_file_path).parent
            log_path.mkdir(parents=True, exist_ok=True)

        # 验证缓存目录
        if self.data.use_cache:
            cache_path = Path(self.data.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

    def get_instance_config(self, instance_type: str) -> 'BSTDTConfig':
        """获取特定实例类型的配置"""
        # 创建副本
        instance_config = BSTDTConfig.from_dict(self.to_dict())
        instance_config.instance_type = instance_type

        # 根据实例类型调整配置
        if instance_type == "small":
            instance_config.solver.time_limit = 3600  # 1小时
            instance_config.solver.thread_count = 2
            instance_config.instance_name = "small_instance"

        elif instance_type == "medium":
            instance_config.solver.time_limit = 14400  # 4小时
            instance_config.solver.thread_count = 4
            instance_config.solver.mip_gap = 0.02
            instance_config.instance_name = "medium_instance"

        elif instance_type == "large":
            instance_config.solver.time_limit = 28800  # 8小时
            instance_config.solver.thread_count = 8
            instance_config.solver.mip_gap = 0.05
            instance_config.instance_name = "large_instance"
            instance_config.constraints.use_bus_capacity_constraints = False  # 大实例关闭复杂约束

        return instance_config

    def __str__(self) -> str:
        """友好的字符串表示"""
        lines = [
            f"BST-DT 配置",
            f"版本: {self.version}",
            f"实例: {self.instance_name} ({self.instance_type})",
            f"描述: {self.description}",
            "",
            "求解器配置:",
            f"  时间限制: {self.solver.time_limit}秒",
            f"  MIP Gap: {self.solver.mip_gap:.1%}",
            f"  线程数: {self.solver.thread_count}",
            "",
            "模型配置:",
            f"  计划时段: {self.model.planning_horizon}分钟",
            f"  最大停留时间: {self.model.max_dwelling_time}分钟",
            f"  默认同步窗口: {self.model.default_min_sync_window}-{self.model.default_max_sync_window}分钟",
            "",
            "约束配置:",
            f"  站点容量约束: {'启用' if self.constraints.use_bus_capacity_constraints else '禁用'}",
            f"  有效不等式: {'启用' if self.constraints.use_valid_inequalities else '禁用'}",
            f"  周期性约束: {'启用' if self.constraints.use_periodic_constraints else '禁用'}",
            "",
            "数据配置:",
            f"  数据目录: {self.data.data_dir}",
            f"  数据验证: {'启用' if self.data.validate_data else '禁用'}",
            "",
            "输出配置:",
            f"  结果目录: {self.output.results_dir}",
            f"  日志级别: {self.output.log_level}",
        ]

        return "\n".join(lines)


# 预定义配置
def get_small_instance_config() -> BSTDTConfig:
    """获取小规模实例配置"""
    config = BSTDTConfig()
    config.instance_type = "small"
    config.instance_name = "small_instance"
    config.solver.time_limit = 3600  # 1小时
    config.solver.thread_count = 2
    config.description = "小规模实例（10条线路）"
    return config


def get_medium_instance_config() -> BSTDTConfig:
    """获取中等规模实例配置"""
    config = BSTDTConfig()
    config.instance_type = "medium"
    config.instance_name = "medium_instance"
    config.solver.time_limit = 14400  # 4小时
    config.solver.thread_count = 4
    config.solver.mip_gap = 0.02
    config.description = "中等规模实例（40条线路）"
    return config


def get_default_config() -> BSTDTConfig:
    """获取默认配置"""
    return BSTDTConfig()


def load_config(config_path: Optional[str] = None) -> BSTDTConfig:
    """加载配置（支持YAML和JSON）"""
    if config_path is None:
        return get_default_config()

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"警告: 配置文件不存在: {config_path}，使用默认配置")
        return get_default_config()

    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
        return BSTDTConfig.from_yaml(str(config_path))
    elif config_path.suffix.lower() == '.json':
        return BSTDTConfig.from_json(str(config_path))
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")