"""
BST-DT模型主实现文件
Bus Synchronization Timetabling with Dwelling Times
基于论文: Transportation Research Part B 174 (2023) 102773
"""

import gurobipy as gp
from gurobipy import GRB
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np

# 导入数据模型类
from src.data_models import (
    ModelData, BusLine, TransferZone, BusStop,
    LineStopAssignment, SynchronizationPair
)

class BSTDT_Model:
    """
    BST-DT模型主类
    实现带停留时间的公交同步时刻表问题
    """

    def __init__(self, data: ModelData, config=None, model_name: str = "BST-DT"):
        """
        初始化模型

        Args:
            data: 模型数据对象
            config: 配置对象（可选）
            model_name: 模型名称
        """
        self.data = data
        self.model_name = model_name
        self.config = config
        self.model = None
        self.vars = {}
        self.results = {}

        # 模型配置 - 优先使用 config 中的参数，否则使用 data 中的默认参数
        if config:
            self.planning_horizon = config.model.planning_horizon
            self.max_dwell_time = config.model.max_dwelling_time
        else:
            self.planning_horizon = data.model_parameters.planning_horizon
            self.max_dwell_time = data.model_parameters.max_dwelling_time

        # 状态跟踪
        self.is_built = False
        self.is_solved = False

    def build_model(self, include_capacity_constraints: bool = True,
                   include_valid_inequalities: bool = False):
        """
        构建完整模型

        Args:
            include_capacity_constraints: 是否包含站点容量约束
            include_valid_inequalities: 是否包含有效不等式
        """
        print(f"构建BST-DT模型: {self.model_name}")
        print("-" * 50)

        # 1. 创建Gurobi模型
        self.model = gp.Model(self.model_name)

        # 2. 创建所有变量
        print("创建决策变量...")
        self._create_variables()

        # 3. 设置目标函数
        print("设置目标函数...")
        self._set_objective()

        # 4. 添加基础约束
        print("添加基础约束...")
        self._add_basic_constraints()

        # 5. 添加到达时间定义约束（论文公式6）
        print("添加到达时间定义约束...")
        self._add_arrival_time_constraints()

        # 6. 添加同步约束
        print("添加同步约束...")
        self._add_synchronization_constraints()

        # 7. 添加站点容量约束（可选）
        if include_capacity_constraints:
            print("添加站点容量约束...")
            self._add_stop_capacity_constraints()

        # 8. 添加最大停留时间约束
        print("添加最大停留时间约束...")
        self._add_max_dwell_time_constraints()

        # 9. 添加有效不等式（可选）
        if include_valid_inequalities:
            print("添加有效不等式...")
            self._add_valid_inequalities()

        self.is_built = True

        # 打印模型统计信息
        self._print_model_stats()

    def _create_variables(self):
        """创建所有决策变量 - 修复版"""
        print(f"创建决策变量...")
        print(f"线路数量: {len(self.data.lines)}")
        print(f"换乘区数量: {len(self.data.transfer_zones)}")
        print(f"同步对数量: {len(self.data.sync_pairs)}")

        # 初始化所有变量字典
        self.vars = {
            'X': {},  # 首班车出发时间
            'Z': {},  # 停留时间
            'T': {},  # 到达时间
            'Y': {},  # 同步变量
            'V': {},  # 站点容量辅助变量
            'S': {}   # 站点容量辅助变量
        }

        # 打印线路频率信息，了解规模
        total_trips = 0
        for line_id, line in self.data.lines.items():
            total_trips += line.frequency

        print(f"总班次数: {total_trips}")

        # 1. 首班车出发时间 X^i (整数变量)
        print("  创建X变量...")
        for line_id, line in self.data.lines.items():
            # 获取边界约束
            constr = self.data.service_constraints.get(line_id)
            if constr:
                lb = constr.first_trip_min_time
                ub = min(constr.first_trip_max_time, line.headway)
            else:
                lb = 0
                ub = line.headway

            var = self.model.addVar(
                lb=lb, ub=ub, vtype=GRB.INTEGER,
                name=f"X_{line_id}"
            )
            self.vars['X'][line_id] = var
            line.first_departure_time = var

        print(f"    X变量: {len(self.vars['X'])}")

        # 2. 停留时间 Z_b^i (连续变量)
        print("  创建Z变量...")
        z_count = 0
        for line_id, line in self.data.lines.items():
            zones = self.data.get_zones_for_line(line_id)
            for zone_id in zones:
                # 检查是否允许停留
                assignment = None
                # 在line_stop_assignments中查找对应的分配
                for (l_id, z_id, s_id), assign in self.data.line_stop_assignments.items():
                    if l_id == line_id and z_id == zone_id:
                        assignment = assign
                        break

                if assignment and assignment.dwell_time_allowed:
                    max_dwell = min(assignment.max_dwelling_time, self.max_dwell_time)
                    var = self.model.addVar(
                        lb=0, ub=max_dwell, vtype=GRB.CONTINUOUS,
                        name=f"Z_{line_id}_{zone_id}"
                    )
                    self.vars['Z'][(line_id, zone_id)] = var
                    line.dwell_times[zone_id] = var
                    z_count += 1
                else:
                    # 不允许停留，停留时间为0
                    line.dwell_times[zone_id] = 0

        print(f"    Z变量: {z_count}")

        # 3. 到达时间 T_b^{ip} (连续变量)
        print("  创建T变量...")
        t_var_count = 0

        # 限制班次数量以减少变量 - 学生许可证限制
        MAX_TRIPS_PER_LINE = 3  # 只考虑前3个班次

        for line_id, line in self.data.lines.items():
            zones = self.data.get_zones_for_line(line_id)

            # 限制考虑的班次数量
            trips_to_consider = min(line.frequency, MAX_TRIPS_PER_LINE)

            for zone_id in zones:
                for trip_idx in range(1, trips_to_consider + 1):
                    var = self.model.addVar(
                        lb=0, ub=self.planning_horizon,
                        vtype=GRB.CONTINUOUS,
                        name=f"T_{line_id}_{zone_id}_{trip_idx}"
                    )
                    self.vars['T'][(line_id, zone_id, trip_idx)] = var
                    t_var_count += 1
                    line.arrival_times[(zone_id, trip_idx)] = var

        print(f"    T变量总数: {t_var_count}")

        # 4. 同步变量 Y_{pqb}^{ij} (二进制变量)
        print("  创建Y变量...")
        y_var_count = 0

        # 限制考虑的同步对数量
        SYNC_PAIRS_TO_CONSIDER = 50  # 只考虑前50个同步对

        for i, ((line_i_id, line_j_id, zone_id), sync_pair) in enumerate(self.data.sync_pairs.items()):
            # 限制同步对数量
            if i >= SYNC_PAIRS_TO_CONSIDER:
                break

            line_i = self.data.lines.get(line_i_id)
            line_j = self.data.lines.get(line_j_id)

            if not line_i or not line_j:
                continue

            # 检查这两个线路是否都经过这个换乘区
            zones_i = self.data.get_zones_for_line(line_i_id)
            zones_j = self.data.get_zones_for_line(line_j_id)

            if zone_id not in zones_i or zone_id not in zones_j:
                # 如果任何一个线路不经过这个换乘区，跳过这个同步对
                continue

            # 限制班次数量
            max_trips_i = min(line_i.frequency, MAX_TRIPS_PER_LINE)
            max_trips_j = min(line_j.frequency, MAX_TRIPS_PER_LINE)

            # 创建同步变量
            for trip_i in range(1, max_trips_i + 1):
                for trip_j in range(1, max_trips_j + 1):
                    # 确保T变量存在
                    if ((line_i_id, zone_id, trip_i) in self.vars['T'] and
                        (line_j_id, zone_id, trip_j) in self.vars['T']):

                        var = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"Y_{line_i_id}_{line_j_id}_{zone_id}_{trip_i}_{trip_j}"
                        )
                        self.vars['Y'][(line_i_id, line_j_id, zone_id, trip_i, trip_j)] = var
                        y_var_count += 1
                        sync_pair.sync_variables[(trip_i, trip_j)] = var

        print(f"    Y变量总数: {y_var_count}")

        # 5. 更新模型以反映变量
        self.model.update()

        print(f"\n变量创建完成:")
        print(f"  X变量: {len(self.vars['X'])}")
        print(f"  Z变量: {len(self.vars['Z'])}")
        print(f"  T变量: {len(self.vars['T'])}")
        print(f"  Y变量: {len(self.vars['Y'])}")
        print(f"  总计: {len(self.vars['X']) + len(self.vars['Z']) + len(self.vars['T']) + len(self.vars['Y'])}")

    def _set_objective(self):
        """设置目标函数：最大化加权同步次数"""
        objective = gp.LinExpr(0)

        # 遍历所有同步对
        for (line_i_id, line_j_id, zone_id), sync_pair in self.data.sync_pairs.items():
            weight = sync_pair.sync_weight

            # 遍历所有班次组合（只考虑已创建的Y变量）
            for (trip_i, trip_j), var in sync_pair.sync_variables.items():
                # 检查Y变量是否存在于模型中
                if (line_i_id, line_j_id, zone_id, trip_i, trip_j) in self.vars['Y']:
                    objective += weight * self.vars['Y'][(line_i_id, line_j_id, zone_id, trip_i, trip_j)]

        self.model.setObjective(objective, GRB.MAXIMIZE)
        print(f"  目标函数已设置，包含 {objective.size()} 项")

    def _add_basic_constraints(self):
        """添加基础约束（论文公式2-3）"""
        print("  添加基础约束...")

        # 约束(2): 首班车在发车间隔内出发
        for line_id, line in self.data.lines.items():
            if line_id in self.vars['X']:
                X_var = self.vars['X'][line_id]
                self.model.addConstr(
                    X_var <= line.headway,
                    name=f"first_trip_ub_{line_id}"
                )

        # 约束(3): 末班车在计划时段内
        for line_id, line in self.data.lines.items():
            if line_id in self.vars['X']:
                X_var = self.vars['X'][line_id]
                last_departure = X_var + (line.frequency - 1) * line.headway

                # 下界
                self.model.addConstr(
                    last_departure >= self.planning_horizon - line.headway,
                    name=f"last_trip_lb_{line_id}"
                )

                # 上界
                self.model.addConstr(
                    last_departure <= self.planning_horizon,
                    name=f"last_trip_ub_{line_id}"
                )

        print(f"    添加了 {len(self.data.lines) * 3} 个基础约束")

    def _add_arrival_time_constraints(self):
        """添加到达时间定义约束（论文公式6）"""
        print("  添加到达时间定义约束...")

        constraint_count = 0

        # 为每条线路构建换乘区顺序
        from collections import defaultdict

        # 构建线路的换乘区顺序映射
        line_zone_sequence = {}
        for line_id in self.data.lines:
            # 使用ModelData的方法获取顺序
            ordered_zones = self.data.get_line_zone_sequence(line_id)
            line_zone_sequence[line_id] = ordered_zones

        # 添加约束(6)
        for line_id, line in self.data.lines.items():
            zones_in_order = line_zone_sequence.get(line_id, [])

            for zone_idx, zone_id in enumerate(zones_in_order):
                # 获取旅行时间 t_b^i
                travel_time = self.data.line_travel_times.get((line_id, zone_id), 0)

                # 限制班次数量（与创建变量时一致）
                MAX_TRIPS_PER_LINE = 3
                trips_to_consider = min(line.frequency, MAX_TRIPS_PER_LINE)

                for trip_idx in range(1, trips_to_consider + 1):
                    T_key = (line_id, zone_id, trip_idx)
                    if T_key not in self.vars['T']:
                        continue

                    T_var = self.vars['T'][T_key]
                    X_var = self.vars['X'][line_id]

                    # 累计该换乘区之前的停留时间
                    dwell_sum = gp.LinExpr(0)
                    for prev_zone in zones_in_order[:zone_idx]:
                        Z_key = (line_id, prev_zone)
                        if Z_key in self.vars['Z']:
                            dwell_sum += self.vars['Z'][Z_key]
                        # 如果没有Z变量，停留时间为0

                    # 约束(6): T_b^{ip} = X^i + t_b^i + (p-1)*h^i + Σ Z_b^i
                    self.model.addConstr(
                        T_var == X_var + travel_time + (trip_idx - 1) * line.headway + dwell_sum,
                        name=f"arrival_{line_id}_{zone_id}_{trip_idx}"
                    )
                    constraint_count += 1

        print(f"    添加了 {constraint_count} 个到达时间约束")

    def _add_synchronization_constraints(self):
        """添加同步约束（论文公式4-5）"""
        print("  添加同步约束...")
        constraint_count = 0

        for (line_i_id, line_j_id, zone_id), sync_pair in self.data.sync_pairs.items():
            line_i = self.data.lines.get(line_i_id)
            line_j = self.data.lines.get(line_j_id)

            if not line_i or not line_j:
                continue

            # 检查这两个线路是否都经过这个换乘区
            if (line_i_id, zone_id, 1) not in self.vars['T'] or (line_j_id, zone_id, 1) not in self.vars['T']:
                # 如果有任何一个线路不经过这个换乘区，跳过这个同步对
                continue

            W_lower = sync_pair.min_sync_window
            W_upper = sync_pair.max_sync_window

            # 计算Big-M值
            M_lower = -self.planning_horizon
            M_upper = self.planning_horizon

            # 限制班次数量
            MAX_TRIPS_PER_LINE = 3
            max_trips_i = min(line_i.frequency, MAX_TRIPS_PER_LINE)
            max_trips_j = min(line_j.frequency, MAX_TRIPS_PER_LINE)

            for trip_i in range(1, max_trips_i + 1):
                for trip_j in range(1, max_trips_j + 1):
                    Y_key = (line_i_id, line_j_id, zone_id, trip_i, trip_j)
                    if Y_key not in self.vars['Y']:
                        # 如果Y变量不存在，跳过
                        continue

                    Y_var = self.vars['Y'][Y_key]
                    T_i_key = (line_i_id, zone_id, trip_i)
                    T_j_key = (line_j_id, zone_id, trip_j)

                    if T_i_key not in self.vars['T'] or T_j_key not in self.vars['T']:
                        # 如果T变量不存在，跳过
                        continue

                    T_i_var = self.vars['T'][T_i_key]
                    T_j_var = self.vars['T'][T_j_key]

                    # 获取停留时间变量（可能为0）
                    Z_key = (line_j_id, zone_id)
                    if Z_key in self.vars['Z']:
                        Z_var = self.vars['Z'][Z_key]
                    else:
                        Z_var = 0

                    # 约束(4): (T_b^{jq} + Z_b^j) - T_b^{ip} >= W_{ij}^b * Y - M * (1 - Y)
                    self.model.addConstr(
                        (T_j_var + Z_var - T_i_var) >=
                        W_lower * Y_var + M_lower * (1 - Y_var),
                        name=f"sync_lower_{line_i_id}_{line_j_id}_{zone_id}_{trip_i}_{trip_j}"
                    )

                    # 约束(5): T_b^{jq} - T_b^{ip} <= W̄_{ij}^b * Y + M * (1 - Y)
                    self.model.addConstr(
                        (T_j_var - T_i_var) <=
                        W_upper * Y_var + M_upper * (1 - Y_var),
                        name=f"sync_upper_{line_i_id}_{line_j_id}_{zone_id}_{trip_i}_{trip_j}"
                    )

                    constraint_count += 2

        print(f"    添加了 {constraint_count} 个同步约束")

    def _add_stop_capacity_constraints(self):
        """添加站点容量约束（论文公式7-11）"""
        if not hasattr(self.data, 'bus_stops') or not self.data.bus_stops:
            print("警告：没有站点数据，跳过站点容量约束")
            return

        print("  添加站点容量约束...")

        # 创建辅助变量 V_{tbp}^i
        print("    创建站点容量辅助变量...")
        # 注意：这会创建大量变量，可能超出许可证限制
        # 因此我们先禁用这个功能
        print("    注意：站点容量约束暂时禁用以避免许可证限制")
        return

    def _add_max_dwell_time_constraints(self):
        """添加最大停留时间约束（论文公式12）"""
        print("  添加最大停留时间约束...")
        constraint_count = 0

        for line_id, line in self.data.lines.items():
            zones = self.data.get_zones_for_line(line_id)
            if len(zones) < 2:
                continue

            # 对线路的每对换乘区
            for i in range(len(zones)):
                for j in range(i + 1, len(zones)):
                    zone_i = zones[i]
                    zone_j = zones[j]

                    # 获取旅行时间（简化处理）
                    travel_time = 10  # 占位值，实际需要根据数据计算

                    # 获取停留时间变量
                    Z_i_key = (line_id, zone_i)
                    Z_j_key = (line_id, zone_j)

                    Z_i = self.vars['Z'].get(Z_i_key)
                    Z_j = self.vars['Z'].get(Z_j_key)

                    if Z_i and Z_j:
                        # 约束(12): Σ Z_b^i ≤ 0.1 * t_{b'}^i
                        self.model.addConstr(
                            Z_i + Z_j <= 0.1 * travel_time,
                            name=f"max_dwell_{line_id}_{zone_i}_{zone_j}"
                        )
                        constraint_count += 1

        print(f"    添加了 {constraint_count} 个最大停留时间约束")

    def _add_valid_inequalities(self):
        """添加有效不等式（论文第3.2节）"""
        print("  添加有效不等式...")

        # 1. 零开始约束（至少有一条线路在时间0发车）
        print("    添加零开始约束...")
        zero_start_vars = []
        for line_id in self.data.lines.keys():
            var = self.model.addVar(vtype=GRB.BINARY, name=f"ST_{line_id}")
            zero_start_vars.append(var)

            # X^i <= CB_i * (1 - ST_i)
            constr = self.data.service_constraints.get(line_id)
            if constr:
                cb_i = constr.first_trip_max_time
            else:
                cb_i = self.data.lines[line_id].headway

            self.model.addConstr(
                self.vars['X'][line_id] <= cb_i * (1 - var),
                name=f"zero_start_bound_{line_id}"
            )

        # 至少有一条线路在时间0发车
        self.model.addConstr(
            gp.quicksum(zero_start_vars) >= 1,
            name="at_least_one_zero_start"
        )

        # 2. 周期性约束（公式14）
        print("    添加周期性约束...")
        periodic_constraint_count = 0

        for (line_i_id, line_j_id, zone_id), sync_pair in self.data.sync_pairs.items():
            line_i = self.data.lines.get(line_i_id)
            line_j = self.data.lines.get(line_j_id)

            if not line_i or not line_j:
                continue

            h_i = line_i.headway
            h_j = line_j.headway

            # 计算最小公倍数
            lcm_val = self._lcm(h_i, h_j)
            k = lcm_val // h_i
            m = lcm_val // h_j

            # 限制班次数量
            MAX_TRIPS_PER_LINE = 3
            max_trips_i = min(line_i.frequency - k, MAX_TRIPS_PER_LINE)
            max_trips_j = min(line_j.frequency - m, MAX_TRIPS_PER_LINE)

            # 添加周期性约束
            for trip_i in range(1, max_trips_i + 1):
                for trip_j in range(1, max_trips_j + 1):
                    Y1_key = (line_i_id, line_j_id, zone_id, trip_i, trip_j)
                    Y2_key = (line_i_id, line_j_id, zone_id, trip_i + k, trip_j + m)

                    Y1 = self.vars['Y'].get(Y1_key)
                    Y2 = self.vars['Y'].get(Y2_key)

                    if Y1 and Y2:
                        self.model.addConstr(
                            Y1 == Y2,
                            name=f"periodic_{line_i_id}_{line_j_id}_{zone_id}_{trip_i}_{trip_j}"
                        )
                        periodic_constraint_count += 1

        print(f"    添加了 {periodic_constraint_count} 个周期性约束")

    def _lcm(self, a: int, b: int) -> int:
        """计算最小公倍数"""
        from math import gcd
        return abs(a * b) // gcd(a, b) if a and b else 0

    def set_solver_parameters(self, time_limit: int = 28800, mip_gap: float = 0.01,
                             threads: int = 4, output_flag: bool = True):
        """
        设置求解器参数

        Args:
            time_limit: 时间限制（秒）
            mip_gap: MIP允许的gap
            threads: 使用的线程数
            output_flag: 是否显示求解过程
        """
        if not self.model:
            raise ValueError("模型未构建，请先调用build_model()")

        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', mip_gap)
        self.model.setParam('Threads', threads)
        self.model.setParam('OutputFlag', 1 if output_flag else 0)

        # 设置其他参数以改进性能
        self.model.setParam('Presolve', 2)  # 积极预处理
        self.model.setParam('Heuristics', 0.05)  # 启发式比例
        self.model.setParam('Cuts', 2)  # 中度切割生成

    def solve(self) -> Dict[str, Any]:
        """
        求解模型

        Returns:
            求解结果字典
        """
        if not self.model:
            raise ValueError("模型未构建，请先调用build_model()")

        print(f"\n求解BST-DT模型...")
        print("-" * 50)

        start_time = datetime.now()

        # 求解
        try:
            self.model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi求解错误: {e}")
            self.results = {
                'status': -1,
                'status_name': 'GUROBI_ERROR',
                'error_message': str(e),
                'feasible': False
            }
            return self.results

        end_time = datetime.now()
        solve_time = (end_time - start_time).total_seconds()

        # 收集结果
        self.results = {
            'status': self.model.status,
            'status_name': self._get_status_name(self.model.status),
            'objective_value': self.model.objVal if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None,
            'runtime': self.model.Runtime,
            'solve_time': solve_time,
            'mip_gap': self.model.MIPGap,
            'node_count': self.model.NodeCount,
            'solution_count': self.model.SolCount,
            'iter_count': self.model.IterCount,
            'optimal': self.model.status == GRB.OPTIMAL,
            'feasible': self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL],
            'timed_out': self.model.status == GRB.TIME_LIMIT,
        }

        # 提取详细结果
        if self.results['feasible']:
            self._extract_solution_details()

        self.is_solved = True

        # 打印求解摘要
        self._print_solve_summary()

        return self.results

    def _extract_solution_details(self):
        """提取解决方案的详细信息"""
        # 1. 同步次数统计
        total_syncs = 0
        syncs_by_line_pair = {}
        syncs_by_zone = {}

        for key, var in self.vars['Y'].items():
            if hasattr(var, 'X') and var.X > 0.5:  # 二进制变量，大于0.5视为1
                total_syncs += 1

                line_i, line_j, zone_id, trip_i, trip_j = key

                # 按线路对统计
                pair_key = (line_i, line_j)
                syncs_by_line_pair[pair_key] = syncs_by_line_pair.get(pair_key, 0) + 1

                # 按换乘区统计
                syncs_by_zone[zone_id] = syncs_by_zone.get(zone_id, 0) + 1

        # 2. 时刻表信息
        timetables = {}
        for line_id, line in self.data.lines.items():
            timetables[line_id] = {
                'first_departure': self.vars['X'][line_id].X if line_id in self.vars['X'] else 0,
                'dwell_times': {},
                'arrival_times': {}
            }

            # 提取停留时间
            for (l_id, zone_id), var in self.vars['Z'].items():
                if l_id == line_id and hasattr(var, 'X'):
                    timetables[line_id]['dwell_times'][zone_id] = var.X

            # 提取到达时间
            for (l_id, zone_id, trip_idx), var in self.vars['T'].items():
                if l_id == line_id and hasattr(var, 'X'):
                    timetables[line_id]['arrival_times'][f"{zone_id}_{trip_idx}"] = var.X

        # 3. 汇总结果
        self.results.update({
            'total_synchronizations': total_syncs,
            'synchronizations_by_line_pair': syncs_by_line_pair,
            'synchronizations_by_zone': syncs_by_zone,
            'timetables': timetables,
            'lines_count': len(self.data.lines),
            'transfer_zones_count': len(self.data.transfer_zones),
            'sync_pairs_count': len(self.data.sync_pairs),
        })

        # 4. 计算改进百分比（如果有基准值）
        baseline_syncs = self._estimate_baseline_syncs()
        if baseline_syncs > 0:
            improvement = (total_syncs - baseline_syncs) / baseline_syncs * 100
            self.results['improvement_percent'] = improvement
            self.results['baseline_synchronizations'] = baseline_syncs

    def _estimate_baseline_syncs(self) -> int:
        """估计基准同步次数（无同步的情况）"""
        total_possible_pairs = 0

        for (line_i_id, line_j_id, zone_id), sync_pair in self.data.sync_pairs.items():
            line_i = self.data.lines.get(line_i_id)
            line_j = self.data.lines.get(line_j_id)

            if not line_i or not line_j:
                continue

            # 可能的同步对数量
            possible_pairs = line_i.frequency * line_j.frequency

            # 假设没有同步时的同步概率
            sync_window = sync_pair.max_sync_window - sync_pair.min_sync_window
            probability = min(sync_window / (line_i.headway + line_j.headway), 1.0)

            total_possible_pairs += int(possible_pairs * probability)

        return total_possible_pairs

    def _get_status_name(self, status_code: int) -> str:
        """获取状态码的名称"""
        status_names = {
            GRB.LOADED: '已加载',
            GRB.OPTIMAL: '最优',
            GRB.INFEASIBLE: '不可行',
            GRB.INF_OR_UNBD: '不可行或无界',
            GRB.UNBOUNDED: '无界',
            GRB.CUTOFF: '超出截断值',
            GRB.ITERATION_LIMIT: '迭代限制',
            GRB.NODE_LIMIT: '节点限制',
            GRB.TIME_LIMIT: '时间限制',
            GRB.SOLUTION_LIMIT: '解限制',
            GRB.INTERRUPTED: '中断',
            GRB.NUMERIC: '数值问题',
            GRB.SUBOPTIMAL: '次优',
            GRB.INPROGRESS: '进行中',
            GRB.USER_OBJ_LIMIT: '用户目标限制',
        }

        return status_names.get(status_code, f'未知状态 ({status_code})')

    def _print_model_stats(self):
        """打印模型统计信息"""
        if not self.model:
            return

        print("\n模型统计信息:")
        print(f"  决策变量总数: {self.model.NumVars:,}")
        print(f"  约束总数: {self.model.NumConstrs:,}")

        # 变量类型统计
        bin_vars = sum(1 for v in self.model.getVars() if v.VType == GRB.BINARY)
        int_vars = sum(1 for v in self.model.getVars() if v.VType == GRB.INTEGER)
        cont_vars = sum(1 for v in self.model.getVars() if v.VType == GRB.CONTINUOUS)

        print(f"  二进制变量: {bin_vars:,}")
        print(f"  整数变量: {int_vars:,}")
        print(f"  连续变量: {cont_vars:,}")

        # 同步相关统计
        y_vars_count = len(self.vars.get('Y', {}))
        print(f"  同步变量(Y): {y_vars_count:,}")

        # 线路和换乘区统计
        print(f"\n问题规模:")
        print(f"  线路数: {len(self.data.lines)}")
        print(f"  换乘区数: {len(self.data.transfer_zones)}")
        print(f"  同步对数: {len(self.data.sync_pairs)}")
        print(f"  计划时段: {self.planning_horizon} 分钟")

        print("-" * 50)

    def _print_solve_summary(self):
        """打印求解摘要"""
        print("\n求解摘要:")
        print("-" * 50)

        status = self.results['status_name']
        print(f"求解状态: {status}")

        if self.results['feasible']:
            obj_val = self.results['objective_value']
            total_syncs = self.results.get('total_synchronizations', 0)

            print(f"目标函数值: {obj_val:.2f}")
            print(f"总同步次数: {total_syncs:,}")

            if 'improvement_percent' in self.results:
                improvement = self.results['improvement_percent']
                print(f"相对于基线的改进: {improvement:.1f}%")

        runtime = self.results['runtime']
        solve_time = self.results['solve_time']
        print(f"求解器运行时间: {runtime:.2f} 秒")
        print(f"总求解时间: {solve_time:.2f} 秒")

        if self.results['feasible']:
            mip_gap = self.results['mip_gap'] * 100
            print(f"MIP Gap: {mip_gap:.2f}%")
            print(f"搜索节点数: {self.results['node_count']:,}")
            print(f"找到的解数量: {self.results['solution_count']}")

        print("-" * 50)

    def save_model(self, filename: str = None):
        """
        保存模型到文件

        Args:
            filename: 文件名，如果为None则使用模型名称
        """
        if not self.model:
            raise ValueError("模型未构建")

        if filename is None:
            filename = f"{self.model_name}.mps"

        # 保存为MPS格式
        self.model.write(filename)
        print(f"模型已保存到: {filename}")

    def save_results(self, filename: str = None):
        """
        保存求解结果到JSON文件

        Args:
            filename: 文件名
        """
        if not self.results:
            print("警告：没有求解结果可保存")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{self.model_name}_{timestamp}.json"

        # 确保结果是可序列化的
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                serializable_results[key] = value
            elif isinstance(value, dict):
                # 处理嵌套字典
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(k, tuple):
                        k_str = '_'.join(map(str, k))
                    else:
                        k_str = str(k)

                    if isinstance(v, (int, float, str, bool, type(None))):
                        serializable_results[key][k_str] = v
                    else:
                        serializable_results[key][k_str] = str(v)
            else:
                serializable_results[key] = str(value)

        # 保存到文件
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {filename}")

    def get_solution_variable(self, var_type: str, key: tuple) -> Optional[gp.Var]:
        """
        获取解决方案变量

        Args:
            var_type: 变量类型 ('X', 'Y', 'Z', 'T')
            key: 变量键值

        Returns:
            变量对象，如果不存在则返回None
        """
        return self.vars.get(var_type, {}).get(key)

    def get_solution_value(self, var_type: str, key: tuple) -> Optional[float]:
        """
        获取解决方案变量的值

        Args:
            var_type: 变量类型 ('X', 'Y', 'Z', 'T')
            key: 变量键值

        Returns:
            变量的值，如果不存在则返回None
        """
        var = self.get_solution_variable(var_type, key)
        if var and hasattr(var, 'X'):
            return var.X
        return None

    def print_timetable_for_line(self, line_id: str):
        """
        打印线路的时刻表

        Args:
            line_id: 线路ID
        """
        if not self.is_solved or not self.results['feasible']:
            print("模型未求解或无可行解")
            return

        if line_id not in self.data.lines:
            print(f"线路 {line_id} 不存在")
            return

        line = self.data.lines[line_id]

        print(f"\n线路 {line_id} 时刻表:")
        print("-" * 60)
        print(f"方向: {line.direction}")
        print(f"发车间隔: {line.headway} 分钟")
        if line_id in self.vars['X'] and hasattr(self.vars['X'][line_id], 'X'):
            print(f"首班车出发时间: {self.vars['X'][line_id].X:.1f} 分钟")
        print(f"班次数: {line.frequency}")
        print()

        # 打印每个换乘区的到达时间
        zones = self.data.get_zones_for_line(line_id)
        print("换乘区到达时间:")
        for zone_id in zones:
            print(f"  换乘区 {zone_id}:")
            for trip_idx in range(1, min(5, line.frequency) + 1):  # 只打印前5个班次
                arrival_time = self.get_solution_value('T', (line_id, zone_id, trip_idx))
                if arrival_time is not None:
                    # 转换为小时:分钟格式
                    hours = int(arrival_time // 60)
                    minutes = int(arrival_time % 60)
                    print(f"    班次 {trip_idx}: {hours:02d}:{minutes:02d}")

        # 打印停留时间
        print("\n停留时间:")
        for (l_id, zone_id), var in self.vars.get('Z', {}).items():
            if l_id == line_id and hasattr(var, 'X') and var.X > 0:
                print(f"  换乘区 {zone_id}: {var.X:.1f} 分钟")

        print("-" * 60)


# 辅助函数
def create_model_from_data(data: ModelData, model_name: str = "BST-DT",
                          include_capacity: bool = True) -> BSTDT_Model:
    """
    从数据创建模型的便捷函数

    Args:
        data: 模型数据
        model_name: 模型名称
        include_capacity: 是否包含站点容量约束

    Returns:
        BSTDT_Model实例
    """
    model = BSTDT_Model(data, model_name)
    model.build_model(include_capacity_constraints=include_capacity)
    return model


def solve_model(data: ModelData, time_limit: int = 3600,
               include_capacity: bool = True, model_name: str = "BST-DT") -> Dict:
    """
    快速求解模型的便捷函数

    Args:
        data: 模型数据
        time_limit: 时间限制（秒）
        include_capacity: 是否包含站点容量约束
        model_name: 模型名称

    Returns:
        求解结果字典
    """
    # 创建模型
    model = create_model_from_data(data, model_name, include_capacity)

    # 设置求解参数
    model.set_solver_parameters(time_limit=time_limit)

    # 求解
    results = model.solve()

    return results, model