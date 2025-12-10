import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# 配置：结果文件路径
RESULT_PATH = r"results/small_instance/timetable_20251203_163123.csv"
# 配置：你想展示哪个换乘区的同步情况？
# (建议先打开 timetable.csv 看看哪个 Zone 出现的次数多，比如 'Z2' 或 'Z5')
TARGET_ZONE = "Z2" 

def plot_gantt():
    if not os.path.exists(RESULT_PATH):
        print(f"❌ 找不到结果文件: {RESULT_PATH}")
        return

    # 1. 读取数据
    df = pd.read_csv(RESULT_PATH)
    
    # 打印列名以检查 (防止列名大小写不一致)
    print("数据列名:", df.columns.tolist())
    
    # 2. 数据清洗与筛选
    # 假设列名可能是 'line_id', 'zone_id', 'arrival_time', 'dwell_time'
    # 如果你的 CSV 列名不一样，请在这里修改映射
    col_map = {
        'line': 'line_id', 
        'zone': 'zone_id', 
        'arr': 'arrival_time', 
        'dwell': 'dwell_time'
    }
    
    # 尝试自动匹配列名
    clean_df = pd.DataFrame()
    try:
        # 根据实际生成的 CSV 调整这里
        # 这里做一个简单的模糊匹配逻辑
        for col in df.columns:
            c = col.lower()
            if 'line' in c: clean_df['line_id'] = df[col]
            if 'zone' in c: clean_df['zone_id'] = df[col]
            if 'arr' in c: clean_df['arrival_time'] = df[col]
            if 'dwell' in c: clean_df['dwell_time'] = df[col]
    except Exception as e:
        print(f"数据解析错误: {e}")
        print("请检查 CSV 文件头")
        return

    # 筛选指定换乘区
    df_zone = clean_df[clean_df['zone_id'] == TARGET_ZONE].copy()
    
    if df_zone.empty:
        print(f"⚠️ 换乘区 {TARGET_ZONE} 没有数据。")
        print("尝试使用的换乘区列表:", clean_df['zone_id'].unique())
        return

    # 按线路排序
    lines = sorted(df_zone['line_id'].unique())
    
    # 3. 开始绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_color = '#1f77b4'  # 论文同款蓝色
    bar_height = 0.5

    for i, line in enumerate(lines):
        line_data = df_zone[df_zone['line_id'] == line]
        
        for _, row in line_data.iterrows():
            start = row['arrival_time']
            duration = row['dwell_time']
            
            # 画驻留条 (蓝色)
            if duration > 0.1: # 只有大于0才画条
                ax.broken_barh([(start, duration)], (i - bar_height/2, bar_height),
                               facecolors=bar_color, edgecolors='black', alpha=0.9, zorder=10)
                # 标注时长
                ax.text(start + duration/2, i, f"{duration:.1f}m", 
                        ha='center', va='center', color='white', fontsize=8, fontweight='bold')
            else:
                # 不驻留画竖线
                ax.vlines(start, i - bar_height/2, i + bar_height/2, colors='red', linewidth=2, zorder=10)

    # 4. 装饰
    ax.set_yticks(range(len(lines)))
    ax.set_yticklabels(lines, fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_title(f"Bus Synchronization Schedule at {TARGET_ZONE}", fontsize=14)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # 图例
    patch = mpatches.Patch(color=bar_color, label='Dwell Time (Sync)')
    plt.legend(handles=[patch], loc='upper right')
    
    plt.tight_layout()
    output_file = f"result_chart_{TARGET_ZONE}.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ 图表已生成: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_gantt()