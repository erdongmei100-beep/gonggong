import sys
import os
import argparse
import logging
from pathlib import Path
import time
import inspect

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# --- 自动适配导入部分 ---
import src.data_loader
import src.config
import src.bstdt_model
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# 自动查找类
DataLoader = None
for name, obj in inspect.getmembers(src.data_loader):
    if inspect.isclass(obj) and 'Loader' in name:
        DataLoader = obj
        break
if DataLoader is None:
    raise ImportError("无法在 src/data_loader.py 中找到数据加载类")

BSTDTConfig = None
for name, obj in inspect.getmembers(src.config):
    if inspect.isclass(obj) and 'Config' in name:
        BSTDTConfig = obj
        break

BSTDT_Model = None
for name, obj in inspect.getmembers(src.bstdt_model):
    if inspect.isclass(obj) and 'Model' in name and 'BST' in name:
        BSTDT_Model = obj
        break

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class BSTDT_Small_Runner:
    def __init__(self, data_path, time_limit=1800, use_inequalities=True, use_capacity=False):
        self.data_path = Path(data_path)
        self.results_dir = project_root / "results" / "small_instance"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_params = {
            'time_limit': time_limit,
            'use_inequalities': use_inequalities,
            'use_capacity': use_capacity
        }

    def load_data(self):
        print(f"数据路径: {self.data_path}")
        print(f"结果目录: {self.results_dir}")
        loader = DataLoader(data_dir=str(self.data_path))
        data = loader.load_all_data()
        self._print_data_summary(data)
        return data

    def _print_data_summary(self, data):
        print("\n" + "="*60)
        print("步骤 1: 加载数据")
        print("="*60)
        print("✓ 成功加载数据")
        lines = getattr(data, 'lines', {})
        zones = getattr(data, 'transfer_zones', {})
        stops = getattr(data, 'bus_stops', {})
        pairs = getattr(data, 'synchronization_pairs', [])
        print(f"  线路数量: {len(lines)}")
        print(f"  换乘区数量: {len(zones)}")
        print(f"  站点数量: {len(stops)}")
        print(f"  同步对数量: {len(pairs)}")

    def setup_config(self):
        print("\n" + "="*60)
        print("步骤 2: 配置模型参数")
        print("="*60)
        config = BSTDTConfig()
        config.solver.time_limit = self.config_params['time_limit']
        config.solver.mip_gap = 0.01
        config.constraints.use_valid_inequalities = self.config_params['use_inequalities']
        config.constraints.use_bus_capacity_constraints = self.config_params['use_capacity']
        print("✓ 模型配置已创建")
        return config

    def build_and_solve(self, data, config):
        print("\n" + "="*60)
        print("步骤 3: 构建和求解模型")
        print("="*60)

        print("创建BST-DT模型实例...")
        model = BSTDT_Model(data, config)
        print("构建模型...")
        model.build_model()

        # 智能探测 Gurobi 对象
        gurobi_solver = None
        possible_names = ['m', 'model', '_model', 'solver', 'gmodel']
        for name in possible_names:
            if hasattr(model, name):
                gurobi_solver = getattr(model, name)
                break
        
        if gurobi_solver is None:
            # 深度搜索
            for attr_name, attr_val in model.__dict__.items():
                if hasattr(attr_val, 'optimize') and hasattr(attr_val, 'Status'):
                    gurobi_solver = attr_val
                    break

        if gurobi_solver is None:
            raise AttributeError("无法找到 Gurobi 模型对象")

        print(f"开始调用 Gurobi 求解器...")
        gurobi_solver.optimize()

        status = gurobi_solver.Status
        if status == GRB.OPTIMAL:
            print(f"\n✓ 找到最优解! 目标值 = {gurobi_solver.ObjVal}")
            results = self._save_results(gurobi_solver, data)

        elif gurobi_solver.Status == GRB.TIME_LIMIT:
            print(f"\n! 达到时间限制. 当前最优目标值 = {gurobi_solver.ObjVal}")
            results = self._save_results(gurobi_solver, data)
            
        elif gurobi_solver.Status == GRB.INFEASIBLE:
            print("\n✗ 模型不可行 (Infeasible)。正在计算 IIS...")
            try:
                gurobi_solver.computeIIS()
                gurobi_solver.write(os.path.join(self.results_dir, "model_iis.ilp"))
                print("  IIS 文件已保存。")
            except:
                print("  无法计算 IIS (可能模型未完全构建)。")
            
        else:
            print(f"\n✗ 求解结束，状态码: {gurobi_solver.Status}")
            results = None

        return results

    def _collect_results(self, model: BSTDT_Model, solve_time: float | None = None) -> dict:
        solver_model = model.m if hasattr(model, "m") else getattr(model, "model", None)
        if solver_model is None:
            return {}

        has_solution = solver_model.SolCount > 0
        results = {
            'status': solver_model.Status,
            'objective_value': solver_model.ObjVal if has_solution else None,
            'runtime': solver_model.Runtime,
            'solve_time_seconds': solve_time,
            'mip_gap': solver_model.MIPGap if solver_model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else None,
            'node_count': solver_model.NodeCount,
            'solution_count': solver_model.SolCount,
            'optimal': solver_model.Status == GRB.OPTIMAL,
        }

        if has_solution and hasattr(self, "_latest_model") and self._latest_model is model:
            results.update({
                'timetables': self._extract_timetables(model),
                'dwell_times': self._extract_dwell_times(model),
            })

        return results

    def _save_results(self, solver_model, data: ModelData, results: dict | None = None, solve_time: float | None = None):
        if results is None:
            results = self._extract_results_from_solver(solver_model)

        if not results:
            print("没有可保存的求解结果")
            return None

        self._save_detailed_results(
            results,
            data,
            solve_time if solve_time is not None else results.get('runtime', 0.0),
        )
        return results

    def _extract_results_from_solver(self, solver_model) -> dict:
        """从 Gurobi 模型对象中解析求解结果."""
        if solver_model is None or not hasattr(solver_model, "getVars"):
            return {}

        timetables: dict = {}
        dwell_times: dict = {}
        first_departures: dict = {}

        for var in solver_model.getVars():
            name = var.VarName

            if name.startswith("T_"):
                parts = name.split("_")
                if len(parts) < 4:
                    continue
                zone_id = parts[-1]
                trip_id = parts[-2]
                line_id = "_".join(parts[1:-2])
                try:
                    trip_int = int(trip_id)
                except ValueError:
                    continue
                timetable = timetables.setdefault(line_id, {"first_departure": None, "arrival_times": {}})
                timetable["arrival_times"][f"{zone_id}_{trip_int}"] = var.X

            elif name.startswith("Z_"):
                parts = name.split("_")
                if len(parts) < 3:
                    continue
                zone_id = parts[-1]
                line_id = "_".join(parts[1:-1])
                dwell_times.setdefault(line_id, {})[zone_id] = var.X

            elif name.startswith("X_"):
                # X_{LineID} -> First departure
                _, line_id = name.split("_", 1)
                first_departures[line_id] = var.X

            elif name.startswith("T[") and name.endswith("]"):
                try:
                    line_id, trip, zone_id = name[2:-1].split(",")
                    timetable = timetables.setdefault(line_id, {"first_departure": None, "arrival_times": {}})
                    timetable["arrival_times"][f"{zone_id}_{int(trip)}"] = var.X
                except Exception:
                    continue

            elif name.startswith("Z[") and name.endswith("]"):
                try:
                    line_id, zone_id = name[2:-1].split(",")
                    dwell_times.setdefault(line_id, {})[zone_id] = var.X
                except Exception:
                    continue

            elif name.startswith("X[") and name.endswith("]"):
                line_id = name[2:-1]
                first_departures[line_id] = var.X

        for line_id, dep_time in first_departures.items():
            timetable = timetables.setdefault(line_id, {"first_departure": None, "arrival_times": {}})
            timetable["first_departure"] = dep_time

        return {
            "status": getattr(solver_model, "Status", None),
            "objective_value": getattr(solver_model, "ObjVal", None),
            "runtime": getattr(solver_model, "Runtime", None),
            "mip_gap": getattr(solver_model, "MIPGap", None),
            "node_count": getattr(solver_model, "NodeCount", None),
            "solution_count": getattr(solver_model, "SolCount", None),
            "optimal": getattr(solver_model, "Status", None) == GRB.OPTIMAL,
            "timetables": timetables,
            "dwell_times": dwell_times,
        }

    def _save_detailed_results(self, results: dict, data: ModelData, solve_time: float):
        """保存详细结果"""
        if not results or 'objective_value' not in results:
            print("没有有效结果可保存")
            return

        # 创建结果文件名
        result_file = self.results_dir / f"detailed_results_{self.timestamp}.json"

        # 准备结果数据
        detailed_results = {
            "timestamp": self.timestamp,
            "solve_time_seconds": solve_time,
            "model_summary": {
                "line_count": len(data.lines),
                "zone_count": len(data.transfer_zones),
                "stop_count": len(data.bus_stops),
                "sync_pair_count": len(data.synchronization_pairs),
                "planning_horizon": data.planning_horizon
            },
            "solver_results": {
                "status": self._get_status_text(results.get('status')),
                "objective_value": results.get('objective_value'),
                "runtime": results.get('runtime'),
                "mip_gap": results.get('mip_gap'),
                "node_count": results.get('node_count'),
                "synchronization_count": results.get('synchronizations', 0),
                "optimal": results.get('optimal', False)
            },
            "timetables": results.get('timetables', {}),
            "dwell_times": results.get('dwell_times', {})
        }

        # 保存到文件
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        print(f"✓ 详细结果已保存到: {result_file}")

        # 也保存一个简化的CSV格式时刻表
        self._save_timetable_csv(results.get('timetables', {}))

    def _extract_timetables(self, model: BSTDT_Model) -> dict:
        """从模型变量中提取时刻表信息"""
        timetables = {}
        for line_id, first_departure_var in model.X.items():
            timetable = {
                'first_departure': first_departure_var.X,
                'arrival_times': {}
            }

            for (l_id, trip, zone_id), arrival_var in model.T.items():
                if l_id != line_id:
                    continue
                timetable['arrival_times'][f"{zone_id}_{trip}"] = arrival_var.X

    def _save_results(self, gurobi_model, data):
        """
        修正版：专门解析下划线分隔的变量名 (例如 T_L01O_1_Z001)
        """
        print("\n正在提取结果并生成排班表...")
        
        records = []
        
        for v in gurobi_model.getVars():
            # 这里允许值为0的变量通过，因为到达时间可能是0
            name = v.VarName
            val = v.X
            
            # --- 解析到达时间变量: T_line_trip_zone ---
            # 你的代码生成的格式: f"T_{line.line_id}_{trip_idx}_{zone_id}"
            if name.startswith("T_"):
                parts = name.split('_')
                # 假设 line_id 可能包含下划线，我们需要更健壮的分割
                # T 是 parts[0]
                # zone_id 是 parts[-1] (通常)
                # trip_idx 是 parts[-2]
                # line_id 是中间剩下的部分
                
                if len(parts) >= 4: # T, line..., trip, zone
                    zone = parts[-1]
                    trip = parts[-2]
                    line = "_".join(parts[1:-2]) # 重新组合 line_id
                    
                    records.append({
                        "type": "arrival",
                        "line_id": line,
                        "trip_id": trip,
                        "zone_id": zone,
                        "value": val
                    })

            # --- 解析驻留时间变量: Z_line_zone ---
            # 你的代码生成的格式: f"Z_{line.line_id}_{zone_id}"
            elif name.startswith("Z_"):
                parts = name.split('_')
                if len(parts) >= 3: # Z, line..., zone
                    zone = parts[-1]
                    line = "_".join(parts[1:-1])
                    
                    records.append({
                        "type": "dwell",
                        "line_id": line,
                        "zone_id": zone,
                        "value": val
                    })

        if not records:
            print("⚠️ 警告: 未提取到有效变量。请检查变量名格式。")
            # 打印前几个变量名帮助调试
            print("实际变量名示例:", [v.VarName for v in gurobi_model.getVars()[:3]])
            return

        df_raw = pd.DataFrame(records)
        
        arrivals = df_raw[df_raw['type'] == 'arrival'].copy()
        dwells = df_raw[df_raw['type'] == 'dwell'].copy()
        
        timetable = []
        
        for _, arr in arrivals.iterrows():
            line = arr['line_id']
            zone = arr['zone_id']
            trip = arr['trip_id']
            arr_time = arr['value']
            
            # 查找对应的驻留时间
            d_rows = dwells[(dwells['line_id'] == line) & (dwells['zone_id'] == zone)]
            dwell_time = d_rows['value'].iloc[0] if not d_rows.empty else 0.0
            
            timetable.append({
                "line_id": line,
                "trip_id": trip,
                "zone_id": zone,
                "arrival_time": arr_time,
                "dwell_time": dwell_time,
                "departure_time": arr_time + dwell_time
            })
            
        df_final = pd.DataFrame(timetable)
        output_path = self.results_dir / "timetable.csv"
        df_final.to_csv(output_path, index=False)
        print(f"✅ 排班表已成功保存至: {output_path}")
        print(f"   (快去画图吧！)")

    def run(self):
        print("\n" + "="*60)
        print("BST-DT 小规模实例求解")
        print("="*60)
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            data = self.load_data()
            config = self.setup_config()
            self.build_and_solve(data, config)
            print("\n" + "="*60)
            print("运行完成!")
            print("="*60)
        except Exception as e:
            print(f"\n运行过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BST-DT Small Instance')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--time-limit', type=int, default=1800, help='Solver time limit in seconds')
    parser.add_argument('--no-capacity', action='store_true', help='Disable bus stop capacity constraints')
    parser.add_argument('--no-inequalities', action='store_true', help='Disable valid inequalities')
    
    args = parser.parse_args()
    
    runner = BSTDT_Small_Runner(
        data_path=args.data_path,
        time_limit=args.time_limit,
        use_inequalities=not args.no_inequalities,
        use_capacity=not args.no_capacity
    )
    runner.run()