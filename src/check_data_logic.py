import pandas as pd
import os

def check_data_logic(data_dir):
    """
    Checks if the synchronization pairs requested in synchronization_pairs.csv
    are actually physically possible based on travel_times.csv and bus_stops.csv.
    """
    print(f"Checking data in: {data_dir}...")
    
    # 1. Load Files
    try:
        stops_df = pd.read_csv(os.path.join(data_dir, "bus_stops.csv"))
        travel_df = pd.read_csv(os.path.join(data_dir, "travel_times.csv"))
        sync_df = pd.read_csv(os.path.join(data_dir, "synchronization_pairs.csv"))
    except Exception as e:
        print(f"CRITICAL: Failed to load CSV files. Error: {e}")
        return

    # 2. Normalize Helper (remove spaces, handle types)
    def clean(val):
        return str(val).strip()

    stops_df['stop_id'] = stops_df['stop_id'].apply(clean)
    stops_df['zone_id'] = stops_df['zone_id'].apply(clean)
    travel_df['to_stop_id'] = travel_df['to_stop_id'].apply(clean)
    travel_df['line_id'] = travel_df['line_id'].apply(clean)
    sync_df['line_i'] = sync_df['line_i'].apply(clean)
    sync_df['line_j'] = sync_df['line_j'].apply(clean)
    sync_df['zone_id'] = sync_df['zone_id'].apply(clean)

    # 3. Build Reachability Map (Which line visits which zone?)
    # Map Stop -> Zone
    stop_to_zone = dict(zip(stops_df['stop_id'], stops_df['zone_id']))
    
    # Map Line -> Set of Zones
    line_to_zones = {}
    
    missing_stops = set()

    for idx, row in travel_df.iterrows():
        line = row['line_id']
        stop = row['to_stop_id']
        
        if stop == "DEPOT": continue # Skip Depot
        
        if line not in line_to_zones:
            line_to_zones[line] = set()
            
        if stop in stop_to_zone:
            zone = stop_to_zone[stop]
            line_to_zones[line].add(zone)
        else:
            missing_stops.add(stop)

    if missing_stops:
        print(f"\n[WARNING] Found {len(missing_stops)} stops in travel_times not present in bus_stops (e.g., {list(missing_stops)[:3]})")

    # 4. Check Sync Pairs
    print("\n--- Logic Validation Results ---")
    valid_count = 0
    total_count = 0
    
    for idx, row in sync_df.iterrows():
        total_count += 1
        l1 = row['line_i']
        l2 = row['line_j']
        zone = row['zone_id']
        
        # Check L1
        l1_ok = (l1 in line_to_zones) and (zone in line_to_zones[l1])
        # Check L2
        l2_ok = (l2 in line_to_zones) and (zone in line_to_zones[l2])
        
        if l1_ok and l2_ok:
            valid_count += 1
        else:
            print(f"[FAIL] Pair #{idx+1} ({l1} & {l2} @ {zone}):")
            if not l1_ok:
                visited = line_to_zones.get(l1, "None")
                print(f"   -> {l1} DOES NOT visit {zone}. (Visits: {visited})")
            if not l2_ok:
                visited = line_to_zones.get(l2, "None")
                print(f"   -> {l2} DOES NOT visit {zone}. (Visits: {visited})")

    print("\n--------------------------------")
    print(f"Summary: {valid_count} / {total_count} pairs are valid.")
    
    if valid_count == 0:
        print("\n[CRITICAL] No valid synchronization pairs found.")
        print("This will cause 'GurobiError: Element 0 of a list of size 0' because no Y variables can be created.")
    else:
        print("\nData looks feasible!")

# --- 使用方法 ---
if __name__ == "__main__":
    # 修改这里的路径为你数据的实际路径
    target_folder = r"data_complete\synth_data_24lines" 
    check_data_logic(target_folder)