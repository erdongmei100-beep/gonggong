import pandas as pd
from pathlib import Path
import os
from src.data_models import (
    BSTDTData, Line, Stop, TravelTime, ServiceConstraint, 
    SyncPair, LineStopAssignment, TransferZone
)

class DataLoader:
    def __init__(self, data_dir: str):
        # 自动处理相对路径/绝对路径
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir.absolute()}")

    def load_all(self) -> BSTDTData:
        """
        加载所有 CSV 并构建高速索引
        """
        print(f"[*] Loading data from {self.data_dir}...")
        data = BSTDTData()

        # 1. Load Lines
        df_lines = self._read_csv("lines.csv")
        for _, row in df_lines.iterrows():
            line = Line(
                line_id=str(row['line_id']),
                headway=int(row['headway']),
                frequency=int(row['frequency']),
                direction=row.get('direction', 'outbound'),
                depot_x=float(row.get('depot_location_x', 0)),
                depot_y=float(row.get('depot_location_y', 0))
            )
            data.lines.append(line)
            data.map_lines[line.line_id] = line  # 构建 Line 索引

        # 2. Load Stops (如果有这个文件)
        if (self.data_dir / "bus_stops.csv").exists():
            df_stops = self._read_csv("bus_stops.csv")
            for _, row in df_stops.iterrows():
                data.stops.append(Stop(
                    stop_id=str(row['stop_id']),
                    name=row.get('stop_name', ''),
                    lat=float(row.get('lat', 0)),
                    lon=float(row.get('lon', 0)),
                    zone_id=str(row.get('zone_id', ''))
                ))

        # 3. Load Travel Times
        df_tt = self._read_csv("travel_times.csv")
        for _, row in df_tt.iterrows():
            data.travel_times.append(TravelTime(
                line_id=str(row['line_id']),
                from_stop=str(row['from_stop_id']),
                to_stop=str(row['to_stop_id']),
                time_min=float(row['travel_time_min']),
                is_transfer=bool(row.get('is_transfer_segment', False))
            ))

        # 4. Load Service Constraints
        df_sc = self._read_csv("service_constraints.csv")
        for _, row in df_sc.iterrows():
            data.service_constraints.append(ServiceConstraint(
                line_id=str(row['line_id']),
                first_trip_min=int(row['first_trip_min_time']),
                first_trip_max=int(row['first_trip_max_time']),
                last_trip_min=int(row['last_trip_min_time']),
                last_trip_max=int(row['last_trip_max_time'])
            ))

        # 5. Load Line-Stop Assignments (核心映射构建)
        df_lsa = self._read_csv("line_stop_assignments.csv")
        for _, row in df_lsa.iterrows():
            l_id = str(row['line_id'])
            z_id = str(row['zone_id'])
            s_id = str(row['stop_id'])
            
            data.line_stop_assignments.append(LineStopAssignment(
                line_id=l_id,
                zone_id=z_id,
                stop_id=s_id,
                sequence=int(row['stop_sequence']),
                max_dwell=int(row.get('max_dwelling_time', 0)),
                dwell_allowed=bool(row.get('dwell_time_allowed', True))
            ))
            
            # --- 构建 (Line, Zone) -> Stop 映射 ---
            # 这样模型里就能直接查：Line 1 在 Zone A 停在哪个物理站点？
            data.map_line_zone_to_stop[(l_id, z_id)] = s_id

        # 6. Load Synchronization Pairs
        df_sync = self._read_csv("synchronization_pairs.csv")
        for _, row in df_sync.iterrows():
            data.sync_pairs.append(SyncPair(
                line_i=str(row['line_i']),
                line_j=str(row['line_j']),
                zone_id=str(row['zone_id']),
                min_window=int(row['min_sync_window']),
                max_window=int(row['max_sync_window']),
                weight=float(row.get('sync_weight', 1.0))
            ))

        # 7. Load Transfer Zones (步行时间映射构建)
        if (self.data_dir / "transfer_zones.csv").exists():
            df_tz = self._read_csv("transfer_zones.csv")
            # 兼容列名：有时候叫 zone_id，有时候叫 from_zone/to_zone
            # 如果是单行表示同站换乘 (ZoneID, WalkTime)，把它视为 From=To
            is_simple_format = 'from_zone' not in df_tz.columns and 'zone_id' in df_tz.columns
            
            for _, row in df_tz.iterrows():
                if is_simple_format:
                    f_z = str(row['zone_id'])
                    t_z = str(row['zone_id']) # 站内换乘
                else:
                    f_z = str(row['from_zone'])
                    t_z = str(row['to_zone'])
                
                w_time = float(row.get('walking_time_min', 0) if 'walking_time_min' in row else row.get('walking_time', 0))
                
                data.transfer_zones.append(TransferZone(
                    from_zone=f_z,
                    to_zone=t_z,
                    walking_time=w_time
                ))
                
                # --- 构建 Walking Time 映射 ---
                data.map_zone_transfer_time[(f_z, t_z)] = w_time
                # 如果是无向的，建议反向也存一份 (Optional)
                if f_z != t_z:
                    data.map_zone_transfer_time[(t_z, f_z)] = w_time

        print("[*] Data loading complete. Indexes built.")
        return data

    def _read_csv(self, filename: str) -> pd.DataFrame:
        """Safe CSV reading helper"""
        path = self.data_dir / filename
        if not path.exists():
            print(f"[WARNING] File not found: {filename}, skipping.")
            return pd.DataFrame()
        return pd.read_csv(path)

# 测试用的 Main
if __name__ == "__main__":
    # 使用相对路径测试
    loader = DataLoader("synth_data_perfect") 
    try:
        data = loader.load_all()
        print(f"Loaded {len(data.lines)} lines.")
        print(f"Map check: (L01O, Z2) -> {data.map_line_zone_to_stop.get(('L01O', 'Z2'))}")
    except Exception as e:
        print(f"Error: {e}")