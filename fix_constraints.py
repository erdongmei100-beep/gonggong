import pandas as pd
import os

# 目标文件夹
DATA_DIR = r"data_complete/synth_data_10lines"

def fix_constraints():
    constr_path = os.path.join(DATA_DIR, "service_constraints.csv")
    
    if not os.path.exists(constr_path):
        print(f"错误: 找不到文件 {constr_path}")
        return

    # 读取
    df = pd.read_csv(constr_path)
    print("原始约束:")
    print(df[['line_id', 'first_trip_min_time', 'first_trip_max_time']].head())

    # 修改：将第一班车的发车时间上限 (first_trip_max_time) 大幅放宽到 120 分钟
    # 这样求解器就可以自由选择在 0 到 120 分钟之间发第一班车，从而满足后续的班次覆盖要求。
    df['first_trip_max_time'] = 120
    
    # 同时也放宽最后一班车的约束，防止两头堵
    # 如果有 last_trip_min_time 列，也适当放宽
    if 'last_trip_min_time' in df.columns:
        df['last_trip_min_time'] = 0 # 允许最后一班车早点结束，只要别太离谱

    # 保存
    df.to_csv(constr_path, index=False)
    print("\n✅ 已放宽首班车时间限制。")
    print("现在 X[L01O] 可以取 104 了，模型将变为可行 (Feasible)。")

if __name__ == "__main__":
    fix_constraints()