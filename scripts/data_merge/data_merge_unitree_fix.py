import os
import json
import pandas as pd
from pathlib import Path
import shutil

# ====================== 你提供的所有目录 ======================
DATA_DIRS = [
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_bottle_workbench_80cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_lemon_workbench_75cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_metal_bowl_workbench_70cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_metal_bowl_workbench_80cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_tissue_box_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_apple_workbench_70cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_toy_bear_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_plastic_bowl_workbench_80cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_cup_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_bottle_workbench_75cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_apple_basket_workbench_75cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_storage_apple_rectangle_plate_workbench_70cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_metal_bowl_workbench_70cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_plastic_bowl_workbench_80cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_tissue_box_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_lemon_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_lemon_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_empty_bottle_workbench_75cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_bottle_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_toy_bear_workbench_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_leftover_food_workbench_75cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_apple_workbench_80cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_apple_workbench_80cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_storage_lemon_rectangle_plate_workbench_75cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_bottle_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_storage_lemon_round_plate_workbench_75cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_storage_apple_round_plate_workbench_70cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_plastic_bowl_workbench_70cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_metal_bowl_workbench_80cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_bread_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_bottle_workbench_80cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_bottle_workbench_75cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_cup_workbench_80cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_crumpled_paper_workbench_75cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_cup_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_plastic_bowl_workbench_70cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_place_cup_workbench_80cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_pick_up_bread_workbench_65cm",
    "/mnt/nas/synnas/成功区/Unitree_G1edu-u3_storage_lemon_plate_workbench_65cm",
]

KEEP_DIM = 28  # 保留前28维


def process_parquet(file_path):
    """处理单个parquet：裁剪 state & action"""
    df = pd.read_parquet(file_path)

    modified = False

    if "observation.state" in df.columns:
        arr = df["observation.state"].to_numpy()
        new_arr = [v[:KEEP_DIM] for v in arr]
        df["observation.state"] = new_arr
        modified = True

    if "action" in df.columns:
        arr = df["action"].to_numpy()
        new_arr = [v[:KEEP_DIM] for v in arr]
        df["action"] = new_arr
        modified = True

    if modified:
        # 备份
        shutil.copy(file_path, file_path.with_suffix(".bak"))
        df.to_parquet(file_path, index=False)
        print(f"  ✅ 已处理 parquet: {file_path.name}")


def process_stats_file(stats_path):
    """处理 episodes_stats.jsonl：裁剪 stats"""
    lines = []
    with open(stats_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            # 处理 observation.state
            if "observation.state" in d["stats"]:
                s = d["stats"]["observation.state"]
                for k in ["min", "max", "mean", "std"]:
                    if k in s and len(s[k]) >= KEEP_DIM:
                        s[k] = s[k][:KEEP_DIM]

            # 处理 action
            if "action" in d["stats"]:
                s = d["stats"]["action"]
                for k in ["min", "max", "mean", "std"]:
                    if k in s and len(s[k]) >= KEEP_DIM:
                        s[k] = s[k][:KEEP_DIM]

            lines.append(json.dumps(d, ensure_ascii=False))

    # 备份
    shutil.copy(stats_path, stats_path.with_suffix(".bak"))

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  ✅ 已处理 stats: {stats_path.name}")


def process_one_dataset(root):
    root = Path(root)
    if not root.exists():
        print(f"⚠️  不存在: {root}")
        return

    print(f"\n==================================================")
    print(f"📂 处理目录: {root.name}")

    # 1) 处理 parquet
    chunk_dir = root / "data" / "chunk-000"
    if chunk_dir.exists():
        for f in chunk_dir.glob("episode_*.parquet"):
            process_parquet(f)

    # 2) 处理 stats
    stats_path = root / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        process_stats_file(stats_path)

    print(f"✅ 完成: {root.name}")


if __name__ == "__main__":
    print("🚀 开始批量裁剪 observation.state & action (30 → 28)")
    for d in DATA_DIRS:
        process_one_dataset(d)

    print("\n🎉 所有数据集处理完成！")
