"""
BST-DT 小规模实例运行脚本
运行10条线路的夜间公交同步时刻表优化模型
"""

import sys
import os
from pathlib import Path
import time
import json
import argparse
from src.config import get_small_instance_config
from src.data_loader import BSTDTDataLoader


# 添加src目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# 现在导入模块
try:
    from src.data_loader import BSTDTDataLoader
    from src.bstdt_model import BSTDT_Model
    from src.config import BSTDTConfig
    from src.data_models import ModelData, BusLine, TransferZone

    print("成功导入所有模块")
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保项目结构正确，并且src目录存在相应的模块文件")
    sys.exit(1)


class SmallInstanceRunner:
    """小规模实例运行器"""

    def __init__(self, data_path: str = None):
        """初始化运行器"""
        if data_path is None:
            # 默认数据路径
            self.data_path = project_root / "data_complete\\data_small"
        else:
            self.data_path = Path(data_path)

        # 确保数据路径存在
        if not self.data_path.exists():
            print(f"数据路径不存在: {self.data_path}")
            sys.exit(1)

        # 输出目录
        self.results_dir = project_root / "results" / "small_instance"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 时间戳用于结果文件命名
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

        print(f"数据路径: {self.data_path}")
        print(f"结果目录: {self.results_dir}")

    def load_data(self) -> ModelData:
        """加载数据"""
        print("\n" + "=" * 60)
        print("步骤 1: 加载数据")
        print("=" * 60)

        try:
            loader = BSTDTDataLoader(self.data_path)
            data = loader.load_all_data()

            # 保存数据到实例变量，供后续使用
            self.data = data

            self._print_data_summary(data)
            return data
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise

    def _print_data_summary(self, data: ModelData):
        """打印数据摘要"""
        print(f"✓ 成功加载数据")
        print(f"  线路数量: {len(data.lines)}")
        print(f"  换乘区数量: {len(data.transfer_zones)}")
        print(f"  站点数量: {len(data.bus_stops)}")
        print(f"  线路-站点分配记录: {len(data.line_stop_assignments)}")
        print(f"  同步对数量: {len(data.synchronization_pairs)}")
        print(f"  计划时段长度: {data.model_parameters.planning_horizon} 分钟")

        # 打印线路信息
        print(f"\n  线路详细信息:")
        for line_id, line in list(data.lines.items())[:5]:  # 只显示前5条线路
            print(f"    {line_id}: 发车间隔={line.headway}分钟, 班次数={line.frequency}")
        if len(data.lines) > 5:
            print(f"    ... 还有 {len(data.lines) - 5} 条线路")

        # 打印换乘区信息
        print(f"\n  换乘区详细信息:")
        #for zone_id, zone in list(data.transfer_zones.items())[:3]:  # 只显示前3个换乘区
            #print(f"    {zone_id}: 允许停留={zone.dwelling_allowed}, "
                 # f"站点数={len(zone.bus_stops)}")
        if len(data.transfer_zones) > 3:
            print(f"    ... 还有 {len(data.transfer_zones) - 3} 个换乘区")

    def create_config(self, time_limit: int = 3600) -> BSTDTConfig:
        """创建模型配置"""
        print("\n" + "=" * 60)
        print("步骤 2: 配置模型参数")
        print("=" * 60)

        # 使用预定义的小规模实例配置
        config = get_small_instance_config()

        # 修改求解器配置
        config.solver.time_limit = time_limit
        config.solver.mip_gap = 0.01
        config.solver.thread_count = 4
        config.solver.output_flag = True

        # 修改模型配置
        config.model.planning_horizon = 239
        config.model.max_dwelling_time = 3
        config.model.big_m_value = 10000

        # 修改约束配置
        config.constraints.use_bus_capacity_constraints = False  # 小规模实例先不使用站点容量约束
        config.constraints.use_valid_inequalities = True
        config.constraints.use_periodic_constraints = True

        # 修改输出配置
        config.output.save_results = True
        config.output.results_dir = str(self.results_dir)
        config.verbose = True

        # 设置实例名称和描述
        config.instance_name = "small_instance_10_lines"
        config.description = f"小规模实例 - {len(self.data.lines)}条夜间公交线路"

        print(f"✓ 模型配置已创建")
        print(f"  求解时间限制: {time_limit}秒 ({time_limit / 3600:.1f}小时)")
        print(f"  MIP Gap: {config.solver.mip_gap * 100}%")
        print(f"  使用有效不等式: {config.constraints.use_valid_inequalities}")
        print(f"  使用站点容量约束: {config.constraints.use_bus_capacity_constraints}")

        return config

    def build_and_solve(self, data: ModelData, config: BSTDTConfig) -> dict:
        """构建并求解模型"""
        print("\n" + "=" * 60)
        print("步骤 3: 构建和求解模型")
        print("=" * 60)

        try:
            # 创建模型实例 - 传入 config
            print("创建BST-DT模型实例...")
            model = BSTDT_Model(data, config)  # 传入 config

            # 构建模型
            print("构建模型（创建变量、约束和目标函数）...")
            model.build_model()

            # 求解模型
            print("开始求解模型...")
            start_time = time.time()
            results = model.solve()
            solve_time = time.time() - start_time

            # 保存详细结果
            self._save_detailed_results(results, data, solve_time)

            return results

        except Exception as e:
            print(f"模型求解失败: {e}")
            import traceback
            traceback.print_exc()
            return None

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

    def _save_timetable_csv(self, timetables: dict):
        """保存时刻表为CSV格式"""
        if not timetables:
            return

        csv_file = self.results_dir / f"timetable_{self.timestamp}.csv"

        rows = []
        for line_id, timetable in timetables.items():
            first_departure = timetable.get('first_departure', 0)
            rows.append({
                'line_id': line_id,
                'first_departure_minutes': first_departure,
                'first_departure_time': self._minutes_to_time(first_departure)
            })

        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✓ 时刻表已保存到: {csv_file}")

    def _minutes_to_time(self, minutes: float) -> str:
        """将分钟数转换为时间字符串（从00:00开始）"""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"

    def _get_status_text(self, status_code: int) -> str:
        """获取求解状态文本"""
        status_map = {
            1: 'LOADED',
            2: 'OPTIMAL',
            3: 'INFEASIBLE',
            4: 'INF_OR_UNBD',
            5: 'UNBOUNDED',
            6: 'CUTOFF',
            7: 'ITERATION_LIMIT',
            8: 'NODE_LIMIT',
            9: 'TIME_LIMIT',
            10: 'SOLUTION_LIMIT',
            11: 'INTERRUPTED',
            12: 'NUMERIC',
            13: 'SUBOPTIMAL',
            14: 'INPROGRESS',
            15: 'USER_OBJ_LIMIT'
        }
        return status_map.get(status_code, f'UNKNOWN({status_code})')

    def analyze_results(self, results: dict, data: ModelData):
        """分析结果"""
        print("\n" + "=" * 60)
        print("步骤 4: 结果分析")
        print("=" * 60)

        if not results or 'objective_value' not in results:
            print("没有有效结果可分析")
            return

        obj_val = results.get('objective_value', 0)
        sync_count = results.get('synchronizations', 0)
        status = results.get('status')
        runtime = results.get('runtime', 0)
        mip_gap = results.get('mip_gap', 0)

        print(f"求解状态: {self._get_status_text(status)}")
        print(f"目标函数值: {obj_val:.2f}")
        print(f"同步次数: {sync_count}")
        print(f"求解时间: {runtime:.2f}秒")
        print(f"MIP Gap: {mip_gap:.4%}")

        # 计算可能的改进
        if sync_count > 0:
            # 这是一个简化的基准计算（实际应该有基准时刻表数据）
            # 假设每条线路的发车时间是均匀分布的
            baseline_sync = self._estimate_baseline_sync(data)
            improvement = 0
            if baseline_sync > 0:
                improvement = (sync_count - baseline_sync) / baseline_sync * 100
                print(f"估计的基准同步次数: {baseline_sync}")
                print(f"相对于基准的改进: {improvement:.1f}%")

            # 按换乘区分析同步情况
            self._analyze_by_zone(results, data)

    def _estimate_baseline_sync(self, data: ModelData) -> int:
        """估计基准同步次数（简化版）"""
        # 这是一个非常简化的估计，假设发车时间均匀分布
        # 实际应该有一个真实的基准时刻表
        total_possible_pairs = 0

        for pair in data.synchronization_pairs:
            f_i = data.lines[pair.line_i].frequency
            f_j = data.lines[pair.line_j].frequency
            total_possible_pairs += f_i * f_j

        # 假设有5%的同步概率（完全随机）
        return int(total_possible_pairs * 0.05)

    def _analyze_by_zone(self, results: dict, data: ModelData):
        """按换乘区分析同步情况"""
        print("\n按换乘区同步统计:")

        zone_stats = {}
        for pair in data.synchronization_pairs:
            zone_id = pair.zone_id
            if zone_id not in zone_stats:
                zone_stats[zone_id] = {
                    'total_pairs': 0,
                    'sync_count': 0,
                    'lines': set()
                }

            # 统计线路
            zone_stats[zone_id]['lines'].add(pair.line_i)
            zone_stats[zone_id]['lines'].add(pair.line_j)

            # 统计可能的同步对
            f_i = data.lines[pair.line_i].frequency
            f_j = data.lines[pair.line_j].frequency
            zone_stats[zone_id]['total_pairs'] += f_i * f_j

        # 计算实际的同步次数（需要从结果中提取，这里简化）
        # 在实际实现中，应该从模型的Y变量中统计每个换乘区的同步次数

        for zone_id, stats in zone_stats.items():
            line_count = len(stats['lines'])
            print(f"  {zone_id}: {line_count}条线路, "
                  f"{stats['total_pairs']}个可能的同步对")

    def run(self, time_limit: int = 3600):
        """运行完整的流程"""
        print("\n" + "=" * 60)
        print("BST-DT 小规模实例求解")
        print("=" * 60)

        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"实例规模: 10条夜间公交线路")
        print(f"时间限制: {time_limit}秒")

        try:
            # 步骤1: 加载数据
            data = self.load_data()

            # 步骤2: 创建配置
            config = self.create_config(time_limit)

            # 步骤3: 构建和求解模型
            results = self.build_and_solve(data, config)

            # 步骤4: 分析结果
            if results:
                self.analyze_results(results, data)

            print("\n" + "=" * 60)
            print("运行完成!")
            print("=" * 60)

            return results

        except KeyboardInterrupt:
            print("\n\n用户中断了程序")
            return None
        except Exception as e:
            print(f"\n运行过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行BST-DT小规模实例')

    parser.add_argument('--data-path', type=str, default=None,
                        help='数据目录路径（默认: ./data_complete）')

    parser.add_argument('--time-limit', type=int, default=3600,
                        help='求解时间限制（秒，默认: 3600）')

    parser.add_argument('--no-capacity', action='store_true',
                        help='不使用站点容量约束')

    parser.add_argument('--no-inequalities', action='store_true',
                        help='不使用有效不等式')

    parser.add_argument('--quick-test', action='store_true',
                        help='快速测试模式（时间限制300秒）')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 如果是快速测试模式
    if args.quick_test:
        args.time_limit = 300
        print("快速测试模式：时间限制设置为300秒")

    # 创建运行器
    runner = SmallInstanceRunner(args.data_path)

    # 运行模型
    results = runner.run(time_limit=args.time_limit)

    # 根据结果退出
    if results and results.get('optimal', False):
        print("\n✓ 找到最优解!")
        sys.exit(0)
    elif results and results.get('objective_value') is not None:
        print("\n✓ 找到可行解（可能不是最优）")
        sys.exit(0)
    else:
        print("\n✗ 求解失败或未找到可行解")
        sys.exit(1)


if __name__ == "__main__":
    main()