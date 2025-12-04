# src/data_loader.py 的完整修正版本
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.data_models import *


class BSTDTDataLoader:
    """BST-DT数据加载器"""

    def __init__(self, data_dir: str = "./data_complete"):
        self.data_dir = Path(data_dir)
        logger.info(f"初始化数据加载器，数据目录: {self.data_dir}")

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """加载CSV文件"""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        logger.debug(f"加载文件: {filename}")
        return pd.read_csv(file_path)

    def load_all_data(self) -> ModelData:
        """加载所有数据"""
        logger.info("开始加载BST-DT模型数据...")

        try:
            # 加载CSV文件
            lines_df = self._load_csv("lines.csv")
            zones_df = self._load_csv("transfer_zones.csv")
            stops_df = self._load_csv("bus_stops.csv")
            assignments_df = self._load_csv("line_stop_assignments.csv")
            travel_times_df = self._load_csv("travel_times.csv")
            sync_pairs_df = self._load_csv("synchronization_pairs.csv")
            model_params_df = self._load_csv("model_parameters.csv")
            service_constr_df = self._load_csv("service_constraints.csv")

            logger.info(f"加载了 {len(model_params_df)} 个模型参数")

        except FileNotFoundError as e:
            logger.error(f"文件加载失败: {e}")
            raise
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

        # 处理基础数据
        lines = self._process_lines(lines_df)
        zones = self._process_zones(zones_df)
        stops = self._process_stops(stops_df, zones)
        assignments = self._process_assignments(assignments_df)
        sync_pairs = self._process_sync_pairs(sync_pairs_df)

        # 处理约束和参数
        service_constraints = self._process_service_constraints(service_constr_df)
        model_parameters = self._process_model_parameters(model_params_df)

        # 处理旅行时间数据 - 转换为TravelTimeSegment对象列表
        travel_time_segments = self._convert_travel_times_df_to_segments(travel_times_df, lines)

        # 创建统一数据容器
        planning_horizon = int(model_parameters.planning_horizon) if hasattr(model_parameters, 'planning_horizon') else 239
        data = ModelData(
            lines=lines,
            transfer_zones=zones,
            bus_stops=stops,
            line_stop_assignments=assignments,
            travel_time_segments=travel_time_segments,  # 使用正确的参数名
            sync_pairs=sync_pairs,
            service_constraints=service_constraints,
            model_parameters=model_parameters,
        )

        # 更新线路与换乘区的关系
        self._update_line_zone_relations(data)

        logger.info(f"数据加载完成: {len(lines)}条线路, {len(zones)}个换乘区, {len(stops)}个站点")
        return data

    # ===================== 数据处理方法 =====================

    def _process_lines(self, df: pd.DataFrame) -> Dict[str, BusLine]:
        """处理线路数据 - 根据实际CSV列名调整"""
        logger.info(f"处理线路数据，共 {len(df)} 条线路")
        lines = {}

        for _, row in df.iterrows():
            try:
                line = BusLine(
                    line_id=str(row['line_id']),
                    headway=int(row['headway']),
                    frequency=int(row['frequency']),
                    direction=str(row['direction']),
                    depot_location_x=float(row.get('depot_location_x', 0.0)),
                    depot_location_y=float(row.get('depot_location_y', 0.0))
                )
                lines[line.line_id] = line
            except KeyError as e:
                logger.warning(f"线路数据缺少字段 {e}，使用默认值")
                line = BusLine(
                    line_id=str(row.get('line_id', f'L{len(lines) + 1}')),
                    headway=int(row.get('headway', 30)),
                    frequency=int(row.get('frequency', 4)),
                    direction=str(row.get('direction', 'outbound')),
                    depot_location_x=0.0,
                    depot_location_y=0.0
                )
                lines[line.line_id] = line

        return lines

    def _process_zones(self, df: pd.DataFrame) -> Dict[str, TransferZone]:
        """处理换乘区数据 - 根据实际CSV列名调整"""
        logger.info(f"处理换乘区数据，共 {len(df)} 个换乘区")
        zones = {}

        for _, row in df.iterrows():
            try:
                zone = TransferZone(
                    zone_id=row['zone_id'],
                    name=str(row.get('name', '')),
                    dwelling_allowed=bool(row['dwelling_allowed']),
                    location_x=float(row.get('location_x', 0.0)),
                    location_y=float(row.get('location_y', 0.0)),
                    description=str(row.get('description', '')),
                    is_major_transfer=bool(row.get('is_major_transfer', False)),
                    max_capacity=int(row.get('max_capacity', 0)),
                    has_security=bool(row.get('has_security', False)),
                    walking_radius_m=float(row.get('walking_radius_m', 200.0))
                )
                zones[zone.zone_id] = zone
            except KeyError as e:
                logger.warning(f"换乘区数据缺少字段 {e}，使用默认值")
                zone = TransferZone(
                    zone_id=row.get('zone_id', f'Z{len(zones) + 1}'),
                    name=str(row.get('name', '')),
                    dwelling_allowed=bool(row.get('dwelling_allowed', True)),
                    location_x=0.0,
                    location_y=0.0,
                    is_major_transfer=False
                )
                zones[zone.zone_id] = zone

        return zones

    def _process_stops(self, df: pd.DataFrame, zones: Dict[str, TransferZone]) -> Dict[str, BusStop]:
        """处理站点数据 - 根据实际CSV列名调整"""
        logger.info(f"处理站点数据，共 {len(df)} 个站点")
        stops = {}

        for _, row in df.iterrows():
            try:
                stop = BusStop(
                    stop_id=row['stop_id'],
                    zone_id=row['zone_id'],
                    capacity=int(row['capacity']),
                    boarding_position=int(row.get('boarding_position', 1))
                )
                stops[stop.stop_id] = stop

                # 将站点添加到对应的换乘区
                zone = zones.get(stop.zone_id)
                if zone:
                    zone.bus_stops.append(stop)
            except KeyError as e:
                logger.warning(f"站点数据缺少字段 {e}，跳过此站点")
                continue

        return stops

    def _process_assignments(self, df: pd.DataFrame) -> Dict[Tuple[str, str, str], LineStopAssignment]:
        """处理线路站点分配数据 - 根据实际CSV列名调整"""
        logger.info(f"处理线路站点分配数据，共 {len(df)} 条记录")
        assignments = {}

        for _, row in df.iterrows():
            try:
                assignment = LineStopAssignment(
                    line_id=row['line_id'],
                    zone_id=row['zone_id'],
                    stop_id=row['stop_id'],
                    stop_sequence=int(row['stop_sequence']),
                    max_dwelling_time=float(row['max_dwelling_time']),
                    dwell_time_allowed=bool(row['dwell_time_allowed'])
                )
                assignments[(assignment.line_id, assignment.zone_id, assignment.stop_id)] = assignment
            except KeyError as e:
                logger.warning(f"线路站点分配数据缺少字段 {e}，跳过此记录")
                continue

        return assignments

    def _convert_travel_times_df_to_segments(self, df: pd.DataFrame, lines: Dict[str, BusLine]) -> List[TravelTimeSegment]:
        """将旅行时间DataFrame转换为TravelTimeSegment对象列表"""
        logger.info(f"转换旅行时间数据，共 {len(df)} 条记录")
        segments = []

        # 重命名列，使列名与模型期望的一致
        if 'travel_time_min' in df.columns and 'travel_time' not in df.columns:
            df = df.rename(columns={'travel_time_min': 'travel_time'})
        if 'from_location' in df.columns and 'from_stop_id' not in df.columns:
            df = df.rename(columns={'from_location': 'from_stop_id'})

        for _, row in df.iterrows():
            try:
                segment = TravelTimeSegment(
                    line_id=str(row['line_id']),
                    from_stop_id=str(row.get('from_stop_id', 'DEPOT')),
                    to_stop_id=str(row.get('to_stop_id', '')),
                    travel_time=float(row['travel_time']),
                    distance=float(row.get('distance', 0.0)),
                    is_transfer_segment=bool(row.get('is_transfer_segment', False)),
                    reliability_factor=float(row.get('reliability_factor', 1.0))
                )
                segments.append(segment)
            except KeyError as e:
                logger.warning(f"旅行时间数据缺少字段 {e}，跳过此记录: {row.to_dict()}")
                continue
            except Exception as e:
                logger.warning(f"旅行时间数据转换失败: {e}，数据: {row.to_dict()}")
                continue

        logger.info(f"成功转换 {len(segments)} 个旅行时间段")
        return segments

    def _process_sync_pairs(self, df: pd.DataFrame) -> Dict[Tuple[str, str, str], SynchronizationPair]:
        """处理同步对数据 - 根据实际CSV列名调整"""
        logger.info(f"处理同步对数据，共 {len(df)} 条记录")
        sync_pairs = {}

        for _, row in df.iterrows():
            try:
                pair = SynchronizationPair(
                    line_i=row['line_i'],
                    line_j=row['line_j'],
                    zone_id=row['zone_id'],
                    min_sync_window=float(row['min_sync_window']),
                    max_sync_window=float(row['max_sync_window']),
                    sync_weight=float(row['sync_weight']),
                    walking_time=float(row.get('walking_time', 0)),
                    is_bidirectional=bool(row.get('is_bidirectional', True)),
                    priority=int(row.get('priority', 1)),
                    walking_time_between=float(row.get('walking_time_between', 0)),
                    sync_priority=str(row.get('sync_priority', 'normal'))
                )
                sync_pairs[(pair.line_i, pair.line_j, pair.zone_id)] = pair
            except KeyError as e:
                logger.warning(f"同步对数据缺少字段 {e}，跳过此记录")
                continue

        return sync_pairs

    def _process_service_constraints(self, df: pd.DataFrame) -> Dict[str, ServiceConstraint]:
        """处理服务约束数据 - 根据实际CSV列名调整"""
        logger.info(f"处理服务约束数据，共 {len(df)} 条记录")
        constraints = {}

        for _, row in df.iterrows():
            try:
                line_id = str(row['line_id'])
                constraint = ServiceConstraint(
                    line_id=line_id,
                    first_trip_min_time=float(row.get('first_trip_min_time', 0)),
                    first_trip_max_time=float(row.get('first_trip_max_time', 30)),
                    last_trip_min_time=float(row.get('last_trip_min_time', 0)),
                    last_trip_max_time=float(row.get('last_trip_max_time', 239)),
                    min_headway=float(row.get('min_headway', 10)),
                    max_headway=float(row.get('max_headway', 60)),
                    max_total_dwell_time=float(row.get('max_total_dwell_time', 10))
                )
                constraints[line_id] = constraint
            except KeyError as e:
                logger.warning(f"服务约束数据缺少字段 {e}，跳过此记录")
                continue

        return constraints

    def _process_model_parameters(self, df: pd.DataFrame) -> ModelParameters:
        """处理模型参数数据 - 根据实际CSV列名调整"""
        logger.info(f"处理模型参数数据，共 {len(df)} 个参数")

        # 清理和准备数据
        processed_df = df.copy()

        # 确保列名正确
        if 'parameter' not in processed_df.columns or 'value' not in processed_df.columns:
            logger.error("模型参数数据缺少必要的列: 'parameter' 或 'value'")
            # 尝试其他可能的列名
            column_mapping = {}
            for col in processed_df.columns:
                col_lower = col.lower()
                if 'param' in col_lower or 'name' in col_lower:
                    column_mapping[col] = 'parameter'
                elif 'value' in col_lower or 'val' in col_lower:
                    column_mapping[col] = 'value'

            if len(column_mapping) == 2:
                processed_df = processed_df.rename(columns=column_mapping)
                logger.info(f"已重命名列: {column_mapping}")
            else:
                logger.error("无法识别模型参数数据的列结构")
                return ModelParameters()

        # 使用ModelParameters类的from_dataframe方法
        return ModelParameters.from_dataframe(processed_df)

    def _update_line_zone_relations(self, data: ModelData):
        """更新线路与换乘区的关系"""
        logger.info("更新线路与换乘区的关系...")

        for (line_id, zone_id, stop_id), assignment in data.line_stop_assignments.items():
            # 更新线路经过的换乘区
            line = data.lines.get(line_id)
            zone = data.transfer_zones.get(zone_id)

            if line and zone:
                zone.lines_serving.add(line_id)

        # 确保line_zone_map正确初始化
        for line_id, line in data.lines.items():
            data.line_zone_map[line_id] = []

        for zone_id, zone in data.transfer_zones.items():
            data.zone_line_map[zone_id] = []
            for line_id in zone.lines_serving:
                if line_id in data.line_zone_map:
                    data.line_zone_map[line_id].append(zone_id)
                data.zone_line_map[zone_id].append(line_id)


# 简化的数据加载函数（方便使用）
def load_bstdt_data(data_dir: str = "./data_complete", validate: bool = True) -> ModelData:
    """
    加载BST-DT数据的简化接口

    Args:
        data_dir: 数据目录路径
        validate: 是否验证数据

    Returns:
        ModelData对象
    """
    loader = BSTDTDataLoader(data_dir)
    data = loader.load_all_data()

    if validate:
        # 调用ModelData的validate_data方法
        errors = data.validate_data()

        if errors:
            logger.error("数据验证失败:")
            for error in errors[:5]:  # 只显示前5个错误
                logger.error(f"  - {error}")
            if len(errors) > 5:
                logger.error(f"  ... 还有 {len(errors) - 5} 个错误")

            raise ValueError("数据验证失败，请检查数据文件")

    return data


# 测试函数
def test_data_loading():
    """测试数据加载"""
    import sys

    # 检查数据目录
    data_dir = "./data_complete"
    if not Path(data_dir).exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        print("请确保在正确的工作目录下运行，或指定正确的数据目录")
        return

    try:
        print("测试数据加载...")
        data = load_bstdt_data(data_dir, validate=True)

        print("\n数据加载成功!")
        print(f"线路数量: {len(data.lines)}")
        print(f"换乘区数量: {len(data.transfer_zones)}")
        print(f"站点数量: {len(data.bus_stops)}")
        print(f"同步对数量: {len(data.sync_pairs)}")
        print(f"计划时段: {data.planning_horizon}分钟")

        # 显示前3条线路信息
        print("\n前3条线路信息:")
        for i, (line_id, line) in enumerate(list(data.lines.items())[:3]):
            print(f"  线路 {line_id}: 发车间隔={line.headway}分钟, "
                  f"频率={line.frequency}班次")

        # 显示前3个同步对信息
        print("\n前3个同步对信息:")
        for i, ((line_i, line_j, zone_id), pair) in enumerate(list(data.sync_pairs.items())[:3]):
            print(f"  同步对 {line_i}-{line_j}@{zone_id}: "
                  f"窗口=[{pair.min_sync_window}, {pair.max_sync_window}], "
                  f"权重={pair.sync_weight}")

        return True

    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 直接运行此文件进行测试
    success = test_data_loading()
    if not success:
        sys.exit(1)