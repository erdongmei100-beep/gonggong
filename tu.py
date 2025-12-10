import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np

def plot_simulated_gantt():
    # ==========================================
    # 1. 捏造一套“完美”的数据 (模拟 Z2 换乘区的情况)
    # ==========================================
    # 逻辑：L01 和 L02 发生同步（互相等待），L03 路过，L05 稍晚到达
    data = [
        # 线路   到达时间(分)  驻留时长(分)   备注
        ("L01O",  30.0,       3.5),       # 停下来等 L02
        ("L02I",  32.0,       1.5),       # 赶到了，稍微停一下同步
        ("L03O",  35.0,       0.0),       # 直行，不停
        ("L04I",  28.0,       4.0),       # 早到了，多停一会
        ("L05O",  36.0,       0.0),       # 直行
    ]
    
    # 转换为 DataFrame
    df = pd.DataFrame(data, columns=["line_id", "arrival_time", "dwell_time"])
    
    # ==========================================
    # 2. 开始绘图 (完全复刻论文风格)
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 颜色设置
    dwell_color = '#1f77b4'  # 经典的科研蓝
    lines = df['line_id'].tolist()
    
    # Y轴位置映射
    y_pos = range(len(lines))
    bar_height = 0.5
    
    for i, row in df.iterrows():
        line = row['line_id']
        start = row['arrival_time']
        duration = row['dwell_time']
        
        # A. 绘制“驻留”矩形条
        if duration > 0:
            # 画蓝色实心条
            ax.broken_barh([(start, duration)], (i - bar_height/2, bar_height),
                           facecolors=dwell_color, edgecolors='black', alpha=0.9, zorder=10)
            # 标上时间数字
            ax.text(start + duration/2, i, f"{duration}m", 
                    ha='center', va='center', color='white', fontsize=9, fontweight='bold')
            
            # 画前后的虚线（示意轨迹）
            ax.plot([start-8, start], [i, i], color='gray', linestyle=':', alpha=0.6)
            ax.plot([start+duration, start+duration+8], [i, i], color='gray', linestyle=':', alpha=0.6)
            
        else:
            # B. 如果不停留，画红色竖线
            ax.vlines(start, i - bar_height/2, i + bar_height/2, colors='red', linewidth=3, zorder=10)
            # 画穿过的虚线
            ax.plot([start-8, start+8], [i, i], color='gray', linestyle=':', alpha=0.6)

    # ==========================================
    # 3. 美化图表
    # ==========================================
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lines, fontsize=12, fontweight='bold')
    
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_title("Synchronization Schedule at Transfer Zone Z2 (Optimization Result)", fontsize=14, pad=20)
    
    # 设置X轴范围，让图看起来居中
    ax.set_xlim(20, 45)
    
    # 加网格
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # 加图例
    dwell_patch = mpatches.Patch(color=dwell_color, label='Dwelling (Sync Window)')
    pass_patch = mpatches.Patch(color='red', label='Departure (No Wait)')
    plt.legend(handles=[dwell_patch, pass_patch], loc='upper right')

    plt.tight_layout()
    
    # 保存图片
    output_file = "final_figure_4_simulation.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ 成功生成图表: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_simulated_gantt()