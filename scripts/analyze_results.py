# scripts/analyze_results.py
"""
BST-DT结果分析脚本
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# 添加src目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from src.utils import format_time, calculate_performance_metrics, ModelStatus


class ResultAnalyzer:
    """结果分析器"""

    def __init__(self, result_files: List[str]):
        self.result_files = [Path(f) for f in result_files]
        self.results = []
        self.summary_df = None

    def load_all_results(self):
        """加载所有结果文件"""
        for result_file in self.result_files:
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    result_data['file_name'] = result_file.name
                    result_data['file_path'] = str(result_file)
                    self.results.append(result_data)
            else:
                print(f"警告: 文件不存在 {result_file}")

        print(f"成功加载 {len(self.results)} 个结果文件")

    def create_summary_dataframe(self) -> pd.DataFrame:
        """创建结果摘要DataFrame"""
        summary_data = []

        for result in self.results:
            row = {
                'file_name': result.get('file_name', ''),
                'timestamp': result.get('timestamp', ''),
                'status': result.get('status', ''),
                'objective_value': result.get('objective_value', 0),
                'synchronizations': result.get('synchronizations', 0),
                'runtime': result.get('runtime', 0),
                'mip_gap': result.get('mip_gap', 0),
                'node_count': result.get('node_count', 0),
                'solve_time': result.get('solve_time', 0),
                'optimal': result.get('optimal', False)
            }

            # 添加时刻表统计
            timetables = result.get('timetables', {})
            if timetables:
                first_departures = [t['first_departure'] for t in timetables.values()]
                row['avg_first_departure'] = np.mean(first_departures) if first_departures else 0
                row['std_first_departure'] = np.std(first_departures) if first_departures else 0

            summary_data.append(row)

        self.summary_df = pd.DataFrame(summary_data)
        return self.summary_df

    def analyze_synchronization_patterns(self, result_index: int = 0) -> Dict[str, Any]:
        """分析同步模式"""
        if result_index >= len(self.results):
            return {}

        result = self.results[result_index]
        timetables = result.get('timetables', {})
        sync_patterns = {
            'total_sync': result.get('synchronizations', 0),
            'by_line_pair': {},
            'by_zone': {},
            'by_time_period': {
                'early_night': 0,  # 00:00-02:00
                'mid_night': 0,  # 02:00-04:00
                'late_night': 0  # 04:00-06:00
            }
        }

        # 这里可以添加更详细的分析逻辑
        # 比如按线路对统计同步次数，按换乘区统计等

        return sync_patterns

    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """与基线结果比较"""
        baseline_path = Path(baseline_file)
        if not baseline_path.exists():
            print(f"基线文件不存在: {baseline_file}")
            return {}

        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)

        comparisons = []
        for result in self.results:
            comparison = calculate_performance_metrics(result, baseline_data)
            comparison['file_name'] = result.get('file_name')
            comparisons.append(comparison)

        return comparisons

    def generate_report(self, output_dir: str = "./reports"):
        """生成分析报告"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        report_file = output_path / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_file, 'w') as f:
            f.write("# BST-DT 模型结果分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 1. 结果摘要\n\n")
            if self.summary_df is not None:
                f.write(self.summary_df.to_markdown(index=False))
                f.write("\n\n")

            f.write("## 2. 关键指标\n\n")
            for i, result in enumerate(self.results):
                f.write(f"### 结果 {i + 1}: {result.get('file_name', 'N/A')}\n\n")
                f.write(f"- 目标函数值: {result.get('objective_value', 'N/A')}\n")
                f.write(f"- 总同步次数: {result.get('synchronizations', 'N/A')}\n")
                f.write(f"- 求解时间: {result.get('runtime', 'N/A'):.2f} 秒\n")
                f.write(f"- MIP Gap: {result.get('mip_gap', 'N/A'):.2%}\n")
                f.write(f"- 最优解: {'是' if result.get('optimal', False) else '否'}\n\n")

            f.write("## 3. 建议\n\n")
            if self.summary_df is not None and len(self.summary_df) > 1:
                best_result = self.summary_df.loc[self.summary_df['objective_value'].idxmax()]
                f.write(f"- 最佳结果: {best_result['file_name']} (目标值: {best_result['objective_value']:.2f})\n")
                f.write(f"- 平均求解时间: {self.summary_df['runtime'].mean():.2f} 秒\n")
                f.write(f"- 平均MIP Gap: {self.summary_df['mip_gap'].mean():.2%}\n")

        print(f"分析报告已生成: {report_file}")
        return report_file

    def plot_comparison_charts(self, output_dir: str = "./reports"):
        """绘制比较图表"""
        if self.summary_df is None or len(self.summary_df) < 2:
            print("需要至少两个结果进行比较")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 目标函数值比较
        axes[0, 0].bar(range(len(self.summary_df)), self.summary_df['objective_value'])
        axes[0, 0].set_xticks(range(len(self.summary_df)))
        axes[0, 0].set_xticklabels(self.summary_df['file_name'], rotation=45, ha='right')
        axes[0, 0].set_title('目标函数值比较')
        axes[0, 0].set_ylabel('目标值')

        # 同步次数比较
        axes[0, 1].bar(range(len(self.summary_df)), self.summary_df['synchronizations'])
        axes[0, 1].set_xticks(range(len(self.summary_df)))
        axes[0, 1].set_xticklabels(self.summary_df['file_name'], rotation=45, ha='right')
        axes[0, 1].set_title('同步次数比较')
        axes[0, 1].set_ylabel('同步次数')

        # 求解时间比较
        axes[1, 0].bar(range(len(self.summary_df)), self.summary_df['runtime'])
        axes[1, 0].set_xticks(range(len(self.summary_df)))
        axes[1, 0].set_xticklabels(self.summary_df['file_name'], rotation=45, ha='right')
        axes[1, 0].set_title('求解时间比较')
        axes[1, 0].set_ylabel('时间（秒）')

        # MIP Gap比较
        axes[1, 1].bar(range(len(self.summary_df)), self.summary_df['mip_gap'])
        axes[1, 1].set_xticks(range(len(self.summary_df)))
        axes[1, 1].set_xticklabels(self.summary_df['file_name'], rotation=45, ha='right')
        axes[1, 1].set_title('MIP Gap比较')
        axes[1, 1].set_ylabel('Gap')

        plt.tight_layout()
        chart_file = output_path / f"comparison_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"比较图表已保存: {chart_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='BST-DT结果分析工具')
    parser.add_argument('result_files', nargs='+', help='结果文件路径')
    parser.add_argument('--baseline', type=str, help='基线结果文件路径')
    parser.add_argument('--output-dir', type=str, default='./reports', help='输出目录')
    parser.add_argument('--generate-charts', action='store_true', help='生成比较图表')

    args = parser.parse_args()

    # 创建分析器
    analyzer = ResultAnalyzer(args.result_files)
    analyzer.load_all_results()

    if len(analyzer.results) == 0:
        print("未找到有效结果文件")
        return

    # 创建摘要表格
    analyzer.create_summary_dataframe()

    # 与基线比较
    if args.baseline:
        comparisons = analyzer.compare_with_baseline(args.baseline)
        if comparisons:
            print(f"与基线比较结果:")
            for comp in comparisons:
                print(f"  文件: {comp['file_name']}")
                if 'improvement_percentage' in comp:
                    print(f"    改进百分比: {comp['improvement_percentage']:.2f}%")

    # 生成报告
    analyzer.generate_report(args.output_dir)

    # 生成图表
    if args.generate_charts and len(analyzer.results) > 1:
        analyzer.plot_comparison_charts(args.output_dir)

    print("分析完成!")


if __name__ == "__main__":
    main()