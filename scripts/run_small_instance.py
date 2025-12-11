import argparse
import sys
import os
import time
import pandas as pd
import logging
from pathlib import Path
import json

# ==========================================
# 核心修复：确保 Python 能找到 src 目录
# ==========================================
# 将项目根目录添加到 sys.path，防止 ModuleNotFoundError
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==========================================
# 核心修复：引用正确的类名 (加上 src. 前缀)
# ==========================================
from src.data_loader import DataLoader
from src.bstdt_model import BSTDTModel
# 修复点：这里原来没有引入 BSTDTData，或者引入的是旧名字 ModelData
from src.data_models import BSTDTData 

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Runner")

class BSTDT_Small_Runner:
    def __init__(self, args):
        self.args = args
        self.config = self._build_config(args)
        
        # 初始化组件
        logger.info(f"Initializing Runner with data: {self.config['data_path']}")
        
        # 1. Loader (Action 1 的成果)
        self.data_loader = DataLoader(self.config['data_path'])
        
        # 准备输出目录
        self.output_dir = Path("output") / f"run_{int(time.time())}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_config(self, args):
        """简单构建配置字典"""
        return {
            "data_path": args.data_path,
            "time_limit": args.time_limit,
            "gap": args.gap,
            "no_capacity": args.no_capacity,
            "no_inequalities": args.no_inequalities
        }

    def run(self):
        """主执行流程"""
        start_time = time.time()
        
        # --- Step 1: Load Data ---
        try:
            logger.info("Step 1: Loading Data...")
            data = self.data_loader.load_all()
            logger.info(f"Data loaded successfully. Lines: {len(data.lines)}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise e

        # --- Step 2: Build Model ---
        try:
            logger.info("Step 2: Building Optimization Model...")
            # Action 2 的成果：传入清洗好的 data
            solver = BSTDTModel(data)
            
            # 设置求解器参数 (TimeLimit 等)
            solver.model.setParam('TimeLimit', self.config['time_limit'])
            solver.model.setParam('MIPGap', self.config['gap'])
            
            solver.build_model()
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise e

        # --- Step 3: Solve ---
        logger.info("Step 3: Solving...")
        success = solver.solve()
        
        solve_duration = time.time() - start_time
        
        # --- Step 4: Save Results ---
        if success:
            logger.info("Step 4: Saving Results...")
            self._save_results(solver, data, solve_time=solve_duration)
            logger.info(f"Done! Results saved to: {self.output_dir.absolute()}")
        else:
            logger.warning("Solver failed to find an optimal solution (or Infeasible).")
            # 即使失败，如果计算了 IIS，通常会在 solve() 内部输出文件

    def _save_results(self, solver_model: BSTDTModel, data: BSTDTData, results: dict | None = None, solve_time: float | None = None):
        """
        保存结果到 CSV
        修复点：data 的类型提示改为 BSTDTData
        """
        # 1. 保存时刻表
        try:
            df_timetable = solver_model.extract_solution_dataframe()
            if not df_timetable.empty:
                save_path = self.output_dir / "solution_timetable.csv"
                df_timetable.to_csv(save_path, index=False)
                logger.info(f"Timetable saved: {save_path}")
            else:
                logger.warning("Timetable dataframe is empty.")
        except Exception as e:
            logger.error(f"Error saving timetable: {e}")

        # 2. 保存换乘同步状态
        try:
            df_sync = solver_model.extract_sync_status()
            if not df_sync.empty:
                save_path = self.output_dir / "solution_sync_status.csv"
                df_sync.to_csv(save_path, index=False)
                logger.info(f"Sync status saved: {save_path}")
        except Exception as e:
            logger.error(f"Error saving sync status: {e}")

        # 3. 保存运行统计 (Run Stats)
        stats = {
            "status": solver_model.model.status,
            "obj_val": solver_model.model.objVal if solver_model.model.status in [2, 3] else None, # 2=OPTIMAL
            "solve_time_seconds": solve_time,
            "data_path": self.config['data_path']
        }
        with open(self.output_dir / "run_stats.json", "w") as f:
            json.dump(stats, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BSTDT Small Instance")
    
    # 参数定义
    parser.add_argument("--data-path", type=str, required=True, help="Path to data folder")
    parser.add_argument("--time-limit", type=int, default=300, help="Solver time limit in seconds")
    parser.add_argument("--gap", type=float, default=0.01, help="MIP Gap relative tolerance")
    
    # 开关参数 (兼容你之前的旧参数，虽然现在模型里可能还没用到)
    parser.add_argument("--no-capacity", action="store_true", help="Disable capacity constraints")
    parser.add_argument("--no-inequalities", action="store_true", help="Disable valid inequalities")

    args = parser.parse_args()
    
    runner = BSTDT_Small_Runner(args)
    runner.run()