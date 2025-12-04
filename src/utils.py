# src/utils.py
"""
BST-DT模型工具函数
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
from enum import Enum


class ModelStatus(Enum):
    """模型状态枚举"""
    OPTIMAL = 2
    TIME_LIMIT = 9
    INF_OR_UNBD = 3
    INFEASIBLE = 4
    UNBOUNDED = 5
    OTHER = 1


def format_time(minutes: float) -> str:
    """将分钟转换为HH:MM格式"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


def calculate_travel_time_stats(travel_times_df: pd.DataFrame) -> Dict[str, Any]:
    """计算旅行时间统计信息"""
    stats = {
        'mean': travel_times_df['travel_time_min'].mean(),
        'std': travel_times_df['travel_time_min'].std(),
        'min': travel_times_df['travel_time_min'].min(),
        'max': travel_times_df['travel_time_min'].max(),
        'count': len(travel_times_df)
    }
    return stats


def create_directory_structure(base_dir: str = ".") -> Dict[str, Path]:
    """创建项目目录结构"""
    paths = {
        'src': Path(base_dir) / 'src',
        'scripts': Path(base_dir) / 'scripts',
        'results': Path(base_dir) / 'results',
        'results_small': Path(base_dir) / 'results' / 'small_instance',
        'results_medium': Path(base_dir) / 'results' / 'medium_instance',
        'notebooks': Path(base_dir) / 'notebooks'
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def validate_data_files(data_dir: Path) -> Dict[str, bool]:
    """验证数据文件是否存在"""
    required_files = [
        'lines.csv',
        'transfer_zones.csv',
        'bus_stops.csv',
        'line_stop_assignments.csv',
        'travel_times.csv',
        'synchronization_pairs.csv',
        'model_parameters.csv',
        'service_constraints.csv'
    ]

    status = {}
    for file in required_files:
        file_path = data_dir / file
        status[file] = file_path.exists()

    return status


def calculate_big_M_values(data: 'ModelData', scaling_factor: float = 1.2) -> Tuple[float, float]:
    """计算Big-M值的合理范围"""
    # 根据论文中的计算方法
    max_travel_time = data.travel_times['travel_time_min'].max()
    max_headway = max(line.headway for line in data.lines.values())

    # M_lower 和 M_upper 的计算
    M_lower = -data.planning_horizon + max_travel_time - max_headway
    M_upper = data.planning_horizon - max_travel_time + max_headway

    # 应用缩放因子
    M_lower = M_lower * scaling_factor
    M_upper = M_upper * scaling_factor

    return abs(M_lower), abs(M_upper)


def save_model_summary(model: 'BSTDT_Model', filename: str):
    """保存模型摘要"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'variables': {
            'total': model.model.NumVars,
            'binary': sum(1 for v in model.model.getVars() if v.VType == 'B'),
            'integer': sum(1 for v in model.model.getVars() if v.VType == 'I'),
            'continuous': sum(1 for v in model.model.getVars() if v.VType == 'C')
        },
        'constraints': {
            'total': model.model.NumConstrs,
            'linear': model.model.NumConstrs,  # 假设都是线性约束
        },
        'nonzeros': model.model.NumNZs,
        'parameters': {
            'time_limit': model.config.solver_time_limit,
            'mip_gap': model.config.mip_gap,
            'threads': model.config.thread_count
        }
    }

    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)


def load_config_from_yaml(config_file: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    try:
        import yaml
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        print("警告: PyYAML未安装，使用默认配置")
        return {}
    except FileNotFoundError:
        print(f"警告: 配置文件 {config_file} 不存在，使用默认配置")
        return {}


def calculate_performance_metrics(results: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    """计算性能指标"""
    metrics = {}

    if 'objective_value' in results and 'objective_value' in baseline:
        current_obj = results['objective_value']
        baseline_obj = baseline['objective_value']

        if baseline_obj != 0:
            metrics['improvement_percentage'] = (current_obj - baseline_obj) / baseline_obj * 100
        else:
            metrics['improvement_percentage'] = float('inf') if current_obj > 0 else 0

    if 'synchronizations' in results and 'synchronizations' in baseline:
        current_sync = results['synchronizations']
        baseline_sync = baseline['synchronizations']

        if baseline_sync != 0:
            metrics['sync_improvement'] = (current_sync - baseline_sync) / baseline_sync * 100
        else:
            metrics['sync_improvement'] = float('inf') if current_sync > 0 else 0

    return metrics