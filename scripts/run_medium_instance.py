# scripts/run_medium_instance.py
"""
运行中等规模实例（40条线路）
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加src目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from src.data_loader import BSTDTDataLoader
from src.bstdt_model import BSTDT_Model
from src.config import BSTDTConfig
from src.utils import create_directory_structure, validate_data_files, calculate_big_M_values


def main():
    print("=" * 60)
    print("BST-DT 中等规模实例求解")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"开始时间: {timestamp}")

    # 1. 创建目录结构
    base_dir = Path(__file__).parent.parent
    paths = create_directory_structure(base_dir)

    # 2. 加载数据（中等规模）
    data_dir = base_dir / 'data_complete'  # 假设中等规模数据在同一个目录
    print(f"数据目录: {data_dir}")

    # 验证数据文件
    file_status = validate_data_files(data_dir)
    missing_files = [f for f, exists in file_status.items() if not exists]
    if missing_files:
        print(f"警告: 缺少以下文件: {missing_files}")
        return

    # 加载数据
    try:
        loader = BSTDTDataLoader(str(data_dir))
        data = loader.load_all_data()
        print(f"✓ 成功加载数据:")
        print(f"  线路数量: {len(data.lines)}")
        print(f"  换乘区数量: {len(data.transfer_zones)}")
        print(f"  站点数量: {len(data.bus_stops)}")
        print(f"  同步对数: {len(data.synchronization_pairs)}")
        print(f"  计划时段: {data.planning_horizon} 分钟")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 配置模型（针对中等规模优化）
    # 计算合适的Big-M值
    M_lower, M_upper = calculate_big_M_values(data)

    config = BSTDTConfig(
        solver_time_limit=28800,  # 8小时，与论文一致
        mip_gap=0.02,  # 中等规模可以接受稍大的gap
        thread_count=8,  # 使用更多线程
        use_bus_capacity=True,
        use_valid_inequalities=True,
        use_periodic_constraints=True,
        big_m_value=max(M_lower, M_upper),
        results_dir=str(paths['results_medium']),
        verbose=True
    )

    print(f"模型配置:")
    print(f"  时间限制: {config.solver_time_limit} 秒 ({config.solver_time_limit / 3600:.1f} 小时)")
    print(f"  MIP Gap: {config.mip_gap:.1%}")
    print(f"  线程数: {config.thread_count}")
    print(f"  Big-M值: {config.big_m_value}")

    # 4. 构建模型
    try:
        print("\n构建模型中...")
        model = BSTDT_Model(data, config)
        model.build_model()
        print("✓ 模型构建完成")
    except Exception as e:
        print(f"✗ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 求解
    print("\n" + "=" * 60)
    print("开始求解...")
    print("=" * 60)

    try:
        results = model.solve()

        # 6. 保存详细结果
        if results.get('status') in [2, 9]:  # OPTIMAL or TIME_LIMIT
            # 保存完整结果
            result_file = paths['results_medium'] / f"medium_results_{timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ 详细结果已保存: {result_file}")

            # 保存时刻表
            timetables = results.get('timetables', {})
            if timetables:
                timetable_file = paths['results_medium'] / f"timetables_{timestamp}.csv"
                timetable_data = []

                for line_id, timetable in timetables.items():
                    for arrival_key, arrival_time in timetable['arrival_times'].items():
                        zone_id, trip = arrival_key.split('_')[:2] if '_' in arrival_key else ('', arrival_key)
                        timetable_data.append({
                            'line_id': line_id,
                            'zone_id': zone_id,
                            'trip': int(trip),
                            'arrival_time': arrival_time,
                            'first_departure': timetable['first_departure']
                        })

                timetable_df = pd.DataFrame(timetable_data)
                timetable_df.to_csv(timetable_file, index=False)
                print(f"✓ 时刻表已保存: {timetable_file}")

            # 与基线比较（如果有的话）
            baseline_file = paths['results_medium'] / "baseline_results.json"
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline = json.load(f)

                # 计算改进
                if baseline.get('objective_value') and results.get('objective_value'):
                    improvement = (results['objective_value'] - baseline['objective_value']) / baseline[
                        'objective_value'] * 100
                    print(f"\n与基线比较:")
                    print(f"  基线目标值: {baseline['objective_value']:.2f}")
                    print(f"  当前目标值: {results['objective_value']:.2f}")
                    print(f"  改进: {improvement:.2f}%")

        # 7. 性能分析
        print("\n" + "=" * 60)
        print("性能分析:")
        print("=" * 60)

        print(f"总计算时间: {results.get('solve_time', 0):.2f} 秒")
        print(f"Gurobi求解时间: {results.get('runtime', 0):.2f} 秒")
        print(f"搜索节点数: {results.get('node_count', 0):,}")
        print(f"找到的解数量: {results.get('solution_count', 0)}")

        if results.get('synchronizations'):
            print(f"总同步次数: {results.get('synchronizations'):,}")

            # 计算每线路平均同步次数
            avg_sync_per_line = results['synchronizations'] / len(data.lines) if data.lines else 0
            print(f"每线路平均同步次数: {avg_sync_per_line:.2f}")

    except Exception as e:
        print(f"✗ 求解过程出错: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("中等规模实例求解完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 确保必要的库已导入
    try:
        import pandas as pd
        import json
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请运行: pip install pandas")
        sys.exit(1)

    main()