
# BST-DT模型数据文件说明

## 核心文件（8个）

### 1. lines.csv - 线路基本信息
- `line_id`: 线路唯一标识 (如 L01O, L01I)
- `headway`: 发车间隔（分钟）
- `frequency`: 计划时段内班次数
- `direction`: 方向 (outbound/inbound)
- `depot_location_x/y`: 车场位置坐标

### 2. transfer_zones.csv - 换乘区信息
- `zone_id`: 换乘区唯一标识
- `dwelling_allowed`: 是否允许停留
- `location_x/y`: 中心位置坐标
- `is_major_transfer`: 是否为主要换乘区

### 3. bus_stops.csv - 公交站点信息
- `stop_id`: 站点唯一标识
- `zone_id`: 所属换乘区（外键）
- `capacity`: 同时停靠最大车辆数
- `boarding_position`: 停靠位置序号

### 4. line_stop_assignments.csv - 线路-站点分配（关键文件）
- `line_id`: 线路ID（外键）
- `zone_id`: 换乘区ID（外键）
- `stop_id`: 站点ID（外键）
- `stop_sequence`: 停靠顺序
- `max_dwelling_time`: 最大停留时间 L_b^i（关键参数）
- `dwell_time_allowed`: 是否允许停留

### 5. travel_times.csv - 详细旅行时间
- `line_id`: 线路ID
- `from_stop_id`: 起点站ID（或"DEPOT"）
- `to_stop_id`: 终点站ID
- `travel_time_min`: 旅行时间（分钟）
- `is_transfer_segment`: 是否为换乘段

### 6. synchronization_pairs.csv - 同步参数
- `line_i`, `line_j`: 同步线路对
- `zone_id`: 同步换乘区
- `min/max_sync_window`: 同步时间窗口
- `sync_weight`: 同步权重

### 7. model_parameters.csv - 全局参数
- `parameter`: 参数名
- `value`: 参数值
- `unit`: 单位
- `description`: 描述

### 8. service_constraints.csv - 服务约束
- `line_id`: 线路ID
- `first/last_trip_min/max_time`: 首末班车时间窗口

## 关键关系
1. **站点属于换乘区**: bus_stops.zone_id → transfer_zones.zone_id
2. **线路停靠站点**: line_stop_assignments 表建立多对多关系
3. **最大停留时间**: 存储在 line_stop_assignments.max_dwelling_time
4. **同步关系**: synchronization_pairs 定义哪些线路在哪些换乘区需要同步
        