import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

# ================= 配置区域 =================
# 如果你有真实的结果文件，请修改这里为 True 并指定路径
USE_REAL_DATA = True
REAL_DATA_PATH = "results/small_instance/timetable_20251203_163123.csv"  # 假设你的运行结果在这里

# 指定要画哪个换乘区 (你的数据里改成了 Z2 是 L1/L2 的交集)
TARGET_ZONE = "Z2"
# 指定要展示哪 5 条线路 (根据你的 24 条线路数据)
TARGET_LINES = ["L01O", "L02O", "L03O", "L13O", "L24O"] 

# ===========================================

def load_data():
    if USE_REAL_DATA:
        try:
            # 假设 CSV 格式: line_id, trip_id, zone_id, arrival_time, dwell_time
            df = pd.read_csv(REAL_DATA_PATH)
            return df
        except Exception as e:
            print(f"读取文件失败: {e}, 使用模拟数据演示。")
    
    # --- 生成模拟数据 (完全模仿论文 Figure 4 的风格) ---
    print("正在生成演示数据...")
    data = []
    
    # 模拟 L01O (类似论文里的 508R)
    data.append({"line_id": "L01O", "arrival_time": 30, "dwell_time": 3.0, "zone_id": "Z2"})
    data.append({"line_id": "L01O", "arrival_time": 60, "dwell_time": 3.0, "zone_id": "Z2"})
    
    # 模拟 L02O (与 L01O 同步)
    data.append({"line_id": "L02O", "arrival_time": 32, "dwell_time": 0.0, "zone_id": "Z2"}) # 刚赶上
    data.append({"line_id": "L02O", "arrival_time": 61, "dwell_time": 2.0, "zone_id": "Z2"})
    
    # 模拟 L03O (稍微晚一点)
    data.append({"line_id": "L03O", "arrival_time": 35, "dwell_time": 0.0, "zone_id": "Z2"})
    
    # 模拟 L13O (长停留)
    data.append({"line_id": "L13O", "arrival_time": 28, "dwell_time": 5.0, "zone_id": "Z2"})
    
    # 模拟 L24O
    data.append({"line_id": "L24O", "arrival_time": 58, "dwell_time": 4.0, "zone_id": "Z2"})

    return pd.DataFrame(data)

def plot_synchronization_gantt(df, zone_id, selected_lines):
    # 1. 筛选数据
    df_zone = df[df['zone_id'] == zone_id].copy()
    df_zone = df_zone[df_zone['line_id'].isin(selected_lines)]
    
    if df_zone.empty:
        print(f"警告: 在换乘区 {zone_id} 没有找到指定线路的数据。")
        return

    # 2. 设置画布
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 颜色设置 (仿论文风格: 蓝色表示驻留)
    bar_color = '#1f77b4'  # Blue
    bar_height = 0.6
    
    # 3. 绘制甘特图
    # Y轴映射: 线路 -> 整数坐标
    line_map = {line: i for i, line in enumerate(selected_lines)}
    
    for _, row in df_zone.iterrows():
        line = row['line_id']
        y = line_map.get(line)
        start = row['arrival_time']
        duration = row['dwell_time']
        end = start + duration
        
        # 绘制驻留条 (Dwell Time Bar)
        if duration > 0:
            ax.broken_barh([(start, duration)], (y - bar_height/2, bar_height), 
                           facecolors=bar_color, edgecolors='black', alpha=0.8, zorder=10)
            # 标注时长
            ax.text(start + duration/2, y, f"{duration}m", 
                    ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        else:
            # 如果没有驻留，画一条竖线表示“路过”
            ax.vlines(start, y - bar_height/2, y + bar_height/2, colors='red', linewidth=2, zorder=10)
            
    # 4. 装饰图表
    ax.set_yticks(range(len(selected_lines)))
    ax.set_yticklabels(selected_lines, fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_title(f"Synchronization Timeline at Transfer Zone: {zone_id}", fontsize=14)
    
    # 添加网格线 (每5分钟一条，模拟论文里的竖线)
    min_time = df_zone['arrival_time'].min() - 5
    max_time = df_zone['arrival_time'].max() + 10
    ax.set_xlim(min_time, max_time)
    
    # 竖向网格
    ax.xaxis.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # 图例
    dwell_patch = mpatches.Patch(color=bar_color, label='Dwelling Time (Stop & Wait)')
    pass_patch = mpatches.Patch(color='red', label='Immediate Departure (No Dwell)')
    plt.legend(handles=[dwell_patch, pass_patch], loc='upper right')

    plt.tight_layout()
    
    # 保存
    output_file = "figure_1_sync_timeline.png"
    plt.savefig(output_file, dpi=300)
    print(f"图表已生成: {output_file}")
    plt.show()

if __name__ == "__main__":
    df = load_data()
    plot_synchronization_gantt(df, TARGET_ZONE, TARGET_LINES)