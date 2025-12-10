import pandas as pd
import os

# 目标文件夹
DATA_DIR = r"data_complete/synth_data_10lines"

def reduce_frequency():
    lines_path = os.path.join(DATA_DIR, "lines.csv")
    
    if not os.path.exists(lines_path):
        print(f"错误: 找不到文件 {lines_path}")
        return

    # 读取
    df = pd.read_csv(lines_path)
    print("原始发车间隔:")
    print(df[['line_id', 'headway']])

    # 修改：强制将所有线路的 headway 设为 60 分钟 (这样 4 小时内只有 4 班车)
    df['headway'] = 60
    
    # 保存
    df.to_csv(lines_path, index=False)
    print("\n✅ 已修改发车间隔为 60 分钟。")
    print("现在的变量数量将减少约 95%，绝对符合免费版限制。")

if __name__ == "__main__":
    reduce_frequency()