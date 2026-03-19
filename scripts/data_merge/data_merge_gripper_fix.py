import os
import json
import jsonlines
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

# ===================== 【全局配置】无需修改 =====================
# 固定配置（与你的服务器路径一致）
BASE_DIR = Path("/mnt/nas/synnas/成功区/二次成功区")  # 处理后数据集根目录
ORIGINAL_BASE_DIR = Path("/mnt/nas/synnas/成功区")   # 原始数据集根目录
# 字段名固定配置
CURRENT_LEFT_GRIPPER_NAME = "left_gripper_open"   # 处理后数据左夹爪（固定）
CURRENT_RIGHT_GRIPPER_NAME = "right_gripper_open" # 处理后数据右夹爪（固定）

# 🔥 核心：缺失episode索引（仅不一致数据集）
MISSING_EPISODES = {
    "Agilex_Cobot_Magic_move_object_black_tablecloth_qced_hardlink": [],
"Agilex_Cobot_Magic_pour_drink_bottle_cup_qced_hardlink": [98, 198, 117, 94, 103],
"Agilex_Cobot_Magic_storage_bread_basket_qced_hardlink": [67, 65]
}
#     "Agilex_Cobot_Magic_close_drawer_upper_qced_hardlink": [1],
#     "Agilex_Cobot_Magic_erase_board_left_qced_hardlink": [8],
#     "Agilex_Cobot_Magic_move_mouse_qced_hardlink": [0, 16, 35],
#     "Agilex_Cobot_Magic_move_object_beige_tablecloth_qced_hardlink": [174],
#     "Agilex_Cobot_Magic_move_object_green_tablecloth_qced_hardlink": [116, 134, 156],
#     "Agilex_Cobot_Magic_move_pencil_sharpener_qced_hardlink": [18, 32],
#     "Agilex_Cobot_Magic_open_drawer_bottom_qced_hardlink": [38],
#     "Agilex_Cobot_Magic_storage_object_closest_apple_qced_hardlink": [0],
#     "Agilex_Cobot_Magic_storage_object_qced_hardlink": [43],
#     "Agilex_Cobot_Magic_storage_orange_basket_left_qced_hardlink": [85],
#     "Agilex_Cobot_Magic_storage_orange_basket_right_qced_hardlink": [2, 22],
#     "Agilex_Cobot_Magic_storage_peach_left_qced_hardlink": [50],
#     "Agilex_Cobot_Magic_storage_peach_right_qced_hardlink": [99],
#     "Agilex_Cobot_Magic_move_object_red_tablecloth_qced_hardlink": [52, 109],
# }

# ✅ 仅保留【不一致数据集】，已处理的全部删除
DATASET_LIST = list(MISSING_EPISODES.keys())
# =================================================================

# ===================== 核心工具函数 =====================
def backup_dataset(current_root: Path):
    """🔥 取消备份：直接返回成功"""
    print(f"ℹ️ 已跳过备份: {current_root.name}")
    return True

def check_episode_count(current_root: Path, original_root: Path):
    """🔥 取消校验：直接继续修复"""
    print(f"ℹ️ 已跳过episode数量校验，继续修复")
    return True

def get_original_episode_idx(dataset_name: str, curr_idx: int) -> int:
    """
    核心映射：处理后idx → 原始正确idx
    规则：原始idx = 处理后idx + 小于等于当前idx的缺失数量
    """
    missing_list = MISSING_EPISODES.get(dataset_name, [])
    count = sum(1 for x in missing_list if x <= curr_idx)
    orig_idx = curr_idx + count
    print(f"🔄 索引映射: 处理后={curr_idx:06d} → 原始={orig_idx:06d}")
    return orig_idx

def get_field_index(info_json_path: Path, target_name: str) -> int:
    """解析info.json获取字段索引"""
    with open(info_json_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    state_names = info["features"]["observation.state"]["names"]

    if target_name in state_names:
        return state_names.index(target_name)
    for idx, name in enumerate(state_names):
        if target_name in name:
            return idx
    raise ValueError(f"❌ 字段 {target_name} 未找到！")

def print_full_frame_and_choose_mode(current_parquet: Path, orig_parquet: Path,
                                     curr_left, curr_right):
    """打印完整数据+选择夹爪顺序"""
    print("\n" + "-"*80)
    print("📊 【第一帧完整数据 - 选择原始数据夹爪顺序】")
    print("-"*80)

    df_curr = pd.read_parquet(current_parquet)
    curr_state = np.array(df_curr["observation.state"].iloc[0])
    curr_act = np.array(df_curr["action"].iloc[0])

    df_orig = pd.read_parquet(orig_parquet)
    orig_state = np.array(df_orig["observation.state"].iloc[0])
    orig_act = np.array(df_orig["action"].iloc[0])

    print(f"🔹 当前处理后数据 - 完整 observation.state: \n{curr_state}")
    print(f"🔹 当前处理后数据 - 完整 action: \n{curr_act}")
    print(f"\n🔹 原始备份数据 - 完整 observation.state: \n{orig_state}")
    print(f"🔹 原始备份数据 - 完整 action: \n{orig_act}")
    print(f"\n🔸 处理后数据索引：左夹爪={curr_left} | 右夹爪={curr_right}")
    print("-"*80)

    while True:
        print("🤔 请选择原始数据的夹爪顺序：")
        print("  1 = 原始颠倒：ORIGINAL_LEFT=right_gripper_open, ORIGINAL_RIGHT=left_gripper_open")
        print("  2 = 原始正常：ORIGINAL_LEFT=left_gripper_open, ORIGINAL_RIGHT=right_gripper_open")
        choice = input("请输入 1 或 2: ").strip()
        if choice == "1":
            return "reverse"
        elif choice == "2":
            return "normal"
        print("❌ 输入错误，请输入 1 或 2！")

def load_indexes(current_root: Path, original_root: Path, mode: str):
    """加载夹爪索引"""
    current_info = current_root / "meta" / "info.json"
    orig_info = original_root / "backup" / "meta" / "info.json"

    curr_left = get_field_index(current_info, CURRENT_LEFT_GRIPPER_NAME)
    curr_right = get_field_index(current_info, CURRENT_RIGHT_GRIPPER_NAME)

    if mode == "normal":
        orig_left = get_field_index(orig_info, "left_gripper_open")
        orig_right = get_field_index(orig_info, "right_gripper_open")
    else:
        orig_left = get_field_index(orig_info, "right_gripper_open")
        orig_right = get_field_index(orig_info, "left_gripper_open")

    print(f"✅ 索引解析完成: 处理后(左={curr_left},右={curr_right}) | 原始(左={orig_left},右={orig_right})")
    return curr_left, curr_right, orig_left, orig_right

def recalculate_stats(values: np.ndarray):
    """计算统计值"""
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }

def repair_single_episode(parquet_file: Path, orig_parquet: Path,
                          curr_left, curr_right, orig_left, orig_right):
    """修复单个parquet"""
    df_orig = pd.read_parquet(orig_parquet)
    state_orig = np.array(df_orig["observation.state"].to_list())
    correct_left = state_orig[:, orig_left]
    correct_right = state_orig[:, orig_right]

    df_curr = pd.read_parquet(parquet_file)
    state_curr = np.array(df_curr["observation.state"].to_list())
    action_curr = np.array(df_curr["action"].to_list())

    state_curr[:, curr_left] = correct_left
    state_curr[:, curr_right] = correct_right
    action_curr[:, curr_left] = correct_left
    action_curr[:, curr_right] = correct_right

    df_curr["observation.state"] = state_curr.tolist()
    df_curr["action"] = action_curr.tolist()
    df_curr.to_parquet(parquet_file, index=False)
    return correct_left, correct_right

def update_single_stats(stats_path: Path, episode_idx: int,
                        left_vals, right_vals, curr_left, curr_right):
    """更新统计文件"""
    try:
        stats = [line for line in jsonlines.open(stats_path)]
        for item in stats:
            if item["episode_index"] == episode_idx:
                s = item["stats"]["observation.state"]
                a = item["stats"]["action"]
                # 更新state
                s["min"][curr_left] = recalculate_stats(left_vals)["min"]
                s["max"][curr_left] = recalculate_stats(left_vals)["max"]
                s["mean"][curr_left] = recalculate_stats(left_vals)["mean"]
                s["std"][curr_left] = recalculate_stats(left_vals)["std"]
                s["min"][curr_right] = recalculate_stats(right_vals)["min"]
                s["max"][curr_right] = recalculate_stats(right_vals)["max"]
                s["mean"][curr_right] = recalculate_stats(right_vals)["mean"]
                s["std"][curr_right] = recalculate_stats(right_vals)["std"]
                # 更新action
                a["min"][curr_left] = recalculate_stats(left_vals)["min"]
                a["max"][curr_left] = recalculate_stats(left_vals)["max"]
                a["mean"][curr_left] = recalculate_stats(left_vals)["mean"]
                a["std"][curr_left] = recalculate_stats(left_vals)["std"]
                a["min"][curr_right] = recalculate_stats(right_vals)["min"]
                a["max"][curr_right] = recalculate_stats(right_vals)["max"]
                a["mean"][curr_right] = recalculate_stats(right_vals)["mean"]
                a["std"][curr_right] = recalculate_stats(right_vals)["std"]
                break

        with jsonlines.open(stats_path, "w") as f:
            for item in stats:
                f.write(item)
    except Exception as e:
        print(f"⚠️ 更新统计失败: {e}")

def process_single_dataset(dataset_name: str):
    """处理单个不一致数据集"""
    print("\n" + "="*80)
    print(f"🚀 开始处理【不一致数据集】: {dataset_name}")
    print("="*80)

    current_root = BASE_DIR / dataset_name
    original_root = ORIGINAL_BASE_DIR / dataset_name.replace("_qced_hardlink", "")
    data_dir = current_root / "data"
    stats_path = current_root / "meta" / "episodes_stats.jsonl"

    if not current_root.exists() or not (original_root / "backup").exists():
        print(f"❌ 路径异常，跳过")
        return

    backup_dataset(current_root)
    check_episode_count(current_root, original_root)

    # 查找第一个parquet
    first_parquet = None
    first_orig_parquet = None
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        if chunk_dir.is_dir():
            parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
            if parquet_files:
                first_parquet = parquet_files[0]
                first_curr_idx = int(first_parquet.stem.split("_")[-1])
                first_orig_idx = get_original_episode_idx(dataset_name, first_curr_idx)
                first_orig_parquet = original_root / "backup" / "data" / chunk_dir.name / f"episode_{first_orig_idx:06d}.parquet"
                break
    if not first_parquet or not first_orig_parquet.exists():
        print(f"❌ 未找到parquet，跳过")
        return

    # 选择修复模式
    curr_left_fixed = get_field_index(current_root / "meta" / "info.json", CURRENT_LEFT_GRIPPER_NAME)
    curr_right_fixed = get_field_index(current_root / "meta" / "info.json", CURRENT_RIGHT_GRIPPER_NAME)
    fix_mode = print_full_frame_and_choose_mode(first_parquet, first_orig_parquet, curr_left_fixed, curr_right_fixed)

    # 加载索引
    try:
        curr_left, curr_right, orig_left, orig_right = load_indexes(current_root, original_root, fix_mode)
    except Exception as e:
        print(f"❌ 加载索引失败: {e}")
        return

    # 批量修复
    print(f"\n🔧 开始执行夹爪修复...")
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        print(f"\n📂 处理 {chunk_dir.name}")

        for parquet_file in sorted(chunk_dir.glob("episode_*.parquet")):
            curr_episode_idx = int(parquet_file.stem.split("_")[-1])
            orig_episode_idx = get_original_episode_idx(dataset_name, curr_episode_idx)
            orig_parquet = original_root / "backup" / "data" / chunk_dir.name / f"episode_{orig_episode_idx:06d}.parquet"

            if not orig_parquet.exists():
                print(f"⚠️ 跳过 {parquet_file.name}")
                continue

            left_vals, right_vals = repair_single_episode(parquet_file, orig_parquet, curr_left, curr_right, orig_left, orig_right)
            update_single_stats(stats_path, curr_episode_idx, left_vals, right_vals, curr_left, curr_right)
            print(f"✅ 完成: episode_{curr_episode_idx:06d}")

    print(f"\n🎉 【不一致数据集】修复完成: {dataset_name}")

# ===================== 主函数 =====================
if __name__ == "__main__":
    print("⚠️  仅处理【episode数量不一致】的数据集！无备份+自动索引映射")
    print("="*80)

    for name in DATASET_LIST:
        try:
            process_single_dataset(name)
        except Exception as e:
            print(f"\n💥 处理失败: {e}")
            continue

    print("\n" + "="*80)
    print("🏁 所有【不一致数据集】修复完毕！")
    print("="*80)
