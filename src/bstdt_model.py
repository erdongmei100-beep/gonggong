import gurobipy as gp
from gurobipy import GRB
import logging
import pandas as pd
from typing import Dict, Tuple, List

# 核心引用：加上 src. 前缀，确保指向正确
from src.data_models import BSTDTData, Line, Stop 

# 设置 Logger
logger = logging.getLogger("BSTDT_Solver")
logging.basicConfig(level=logging.INFO)

class BSTDTModel:
    def __init__(self, data: BSTDTData):
        self.data = data
        self.model = gp.Model("BusSyncOptimization")
        
        # === 变量容器 ===
        # 时间变量: (line_id, trip_idx, stop_id) -> GurobiVar
        self.T_arr: Dict[Tuple[str, int, str], gp.Var] = {}
        self.T_dep: Dict[Tuple[str, int, str], gp.Var] = {}
        
        # 同步成功标识 (Soft Constraint): (sync_id, trip_i, trip_j) -> GurobiVar
        self.y_sync: Dict[Tuple[int, int, int], gp.Var] = {}

    def build_model(self):
        """主构建流程"""
        logger.info("[1/4] Creating variables...")
        self._create_variables()
        
        logger.info("[2/4] Adding operational constraints...")
        self._add_operational_constraints()
        
        logger.info("[3/4] Adding Soft Synchronization constraints (Big-M)...")
        self._add_synchronization_constraints()
        
        logger.info("[4/4] Setting objective function...")
        self._set_objective()
        
        self.model.update()
        logger.info(f"Model Built: {self.model.NumVars} vars, {self.model.NumConstrs} constrs.")

    def _create_variables(self):
        """创建核心时间变量"""
        for line in self.data.lines:
            # 1. 估算需要的班次数量
            sc = next((x for x in self.data.service_constraints if x.line_id == line.line_id), None)
            if not sc:
                logger.warning(f"Skipping line {line.line_id} due to missing constraints.")
                continue

            duration = sc.last_trip_max - sc.first_trip_min
            num_trips = int(duration / line.headway) + 2 
            
            # 获取该线路所有站点序列
            line_assignments = sorted(
                [x for x in self.data.line_stop_assignments if x.line_id == line.line_id],
                key=lambda x: x.sequence
            )

            for trip_idx in range(num_trips):
                for stop_assign in line_assignments:
                    s_id = stop_assign.stop_id
                    
                    v_name_arr = f"Arr_{line.line_id}_{trip_idx}_{s_id}"
                    v_name_dep = f"Dep_{line.line_id}_{trip_idx}_{s_id}"
                    
                    self.T_arr[(line.line_id, trip_idx, s_id)] = self.model.addVar(
                        lb=0, ub=300, vtype=GRB.CONTINUOUS, name=v_name_arr 
                    )
                    self.T_dep[(line.line_id, trip_idx, s_id)] = self.model.addVar(
                        lb=0, ub=300, vtype=GRB.CONTINUOUS, name=v_name_dep
                    )

    def _add_operational_constraints(self):
        """添加基础运行约束 (物理约束)"""
        for line in self.data.lines:
            l_id = line.line_id
            trips = sorted(list(set(k[1] for k in self.T_dep.keys() if k[0] == l_id)))
            
            assignments = sorted(
                [x for x in self.data.line_stop_assignments if x.line_id == l_id],
                key=lambda x: x.sequence
            )
            
            sc = next((x for x in self.data.service_constraints if x.line_id == l_id), None)

            for k in trips:
                start_node = assignments[0].stop_id
                
                # --- A. 首末班车时间窗 ---
                if k == 0 and sc:
                    self.model.addConstr(
                        self.T_dep[(l_id, k, start_node)] >= sc.first_trip_min,
                        name=f"FirstTripMin_{l_id}"
                    )
                    self.model.addConstr(
                        self.T_dep[(l_id, k, start_node)] <= sc.first_trip_max,
                        name=f"FirstTripMax_{l_id}"
                    )
                
                # --- B. 站内驻留 (Dwell) ---
                for idx, assign in enumerate(assignments):
                    s_curr = assign.stop_id
                    if idx > 0:
                        self.model.addConstr(
                            self.T_dep[(l_id, k, s_curr)] >= self.T_arr[(l_id, k, s_curr)],
                            name=f"Dwell_{l_id}_{k}_{s_curr}"
                        )

                # --- C. 站间运行 (Travel) ---
                for idx in range(len(assignments) - 1):
                    s_curr = assignments[idx].stop_id
                    s_next = assignments[idx+1].stop_id
                    
                    tt_obj = next((t for t in self.data.travel_times 
                                   if t.line_id == l_id and t.from_stop == s_curr and t.to_stop == s_next), None)
                    travel_time = tt_obj.time_min if tt_obj else 0
                    
                    self.model.addConstr(
                        self.T_arr[(l_id, k, s_next)] == self.T_dep[(l_id, k, s_curr)] + travel_time,
                        name=f"Travel_{l_id}_{k}_{s_curr}_{s_next}"
                    )

            # --- D. 班次间隔 (Headway) ---
            for i in range(len(trips) - 1):
                start_node = assignments[0].stop_id
                self.model.addConstr(
                    self.T_dep[(l_id, trips[i+1], start_node)] == self.T_dep[(l_id, trips[i], start_node)] + line.headway,
                    name=f"Headway_{l_id}_{trips[i]}"
                )

    def _add_synchronization_constraints(self):
        """核心逻辑：软约束同步 (Big-M Method)"""
        BIG_M = 1000

        for idx, sync in enumerate(self.data.sync_pairs):
            l_i, l_j = sync.line_i, sync.line_j
            zone = sync.zone_id
            
            try:
                stop_u = self.data.map_line_zone_to_stop[(l_i, zone)] 
                stop_v = self.data.map_line_zone_to_stop[(l_j, zone)] 
            except KeyError:
                logger.error(f"Mapping Error: {l_i} or {l_j} missing in zone {zone}")
                continue

            # 获取步行时间
            w_time = self.data.map_zone_transfer_time.get((zone, zone), 0.0) 
            if w_time == 0:
                 w_time = self.data.map_zone_transfer_time.get((sync.zone_id, sync.zone_id), 2.0)

            trips_i = sorted(list(set(k[1] for k in self.T_arr.keys() if k[0] == l_i)))
            trips_j = sorted(list(set(k[1] for k in self.T_dep.keys() if k[0] == l_j)))

            for u in trips_i:
                for v in trips_j:
                    y = self.model.addVar(vtype=GRB.BINARY, name=f"Sync_{idx}_{u}_{v}")
                    self.y_sync[(idx, u, v)] = y
                    
                    lhs = self.T_dep[(l_j, v, stop_v)]
                    rhs_arr = self.T_arr[(l_i, u, stop_u)]
                    
                    # 软约束: Min <= Gap <= Max (当y=1时生效)
                    self.model.addConstr(
                        lhs - rhs_arr - w_time >= sync.min_window - BIG_M * (1 - y),
                        name=f"SyncMin_{idx}_{u}_{v}"
                    )
                    self.model.addConstr(
                        lhs - rhs_arr - w_time <= sync.max_window + BIG_M * (1 - y),
                        name=f"SyncMax_{idx}_{u}_{v}"
                    )

    def _set_objective(self):
        """目标函数: 最大化换乘成功"""
        sync_score = gp.LinExpr()
        for (idx, u, v), y_var in self.y_sync.items():
            weight = self.data.sync_pairs[idx].weight
            sync_score += weight * y_var
        
        operational_cost = gp.LinExpr()
        for var in self.T_dep.values():
            operational_cost += 0.001 * var

        self.model.setObjective(sync_score - operational_cost, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        if self.model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            logger.info(f"Solved! Obj: {self.model.objVal}")
            return True
        elif self.model.status == GRB.INFEASIBLE:
            logger.error("Still Infeasible! Computing IIS...")
            self.model.computeIIS()
            self.model.write("debug_infeasible.ilp")
            return False
        else:
            logger.warning(f"Solver ended with status {self.model.status}")
            return False

    def extract_solution_dataframe(self) -> pd.DataFrame:
        """提取结果表"""
        if self.model.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            return pd.DataFrame()

        rows = []
        for (line_id, trip_idx, stop_id), var_dep in self.T_dep.items():
            var_arr = self.T_arr.get((line_id, trip_idx, stop_id))
            
            assign = next((x for x in self.data.line_stop_assignments 
                           if x.line_id == line_id and x.stop_id == stop_id), None)
            
            rows.append({
                "LineID": line_id,
                "TripIdx": trip_idx,
                "StopID": stop_id,
                "Sequence": assign.sequence if assign else -1,
                "ArrTime": var_arr.X if var_arr else 0,
                "DepTime": var_dep.X
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values(by=["LineID", "TripIdx", "Sequence"])

    def extract_sync_status(self) -> pd.DataFrame:
        """提取换乘状态"""
        if self.model.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            return pd.DataFrame()
        
        rows = []
        for (idx, u, v), y_var in self.y_sync.items():
            if y_var.X > 0.5: 
                pair = self.data.sync_pairs[idx]
                rows.append({
                    "FromLine": pair.line_i,
                    "ToLine": pair.line_j,
                    "Zone": pair.zone_id,
                    "FromTrip": u,
                    "ToTrip": v,
                    "Success": True
                })
        return pd.DataFrame(rows)