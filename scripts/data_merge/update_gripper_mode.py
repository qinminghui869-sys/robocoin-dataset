import numpy as np
import pandas as pd
from pathlib import Path
import logging

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== 配置项 =====================
# 根目录（和你之前完全一致）
BASE_DIR = Path("/mnt/nas/synnas/成功区/二次成功区")

# 需要处理的全部数据集
DATASET_LIST = [
#    "Agilex_Cobot_Magic_Agilex_Cobot_Magic_move_object_qced_hardlink",
#         "Agilex_Cobot_Magic_classify_objects_eight_qced_hardlink",
#         "Agilex_Cobot_Magic_classify_objects_six_qced_hardlink",
#         "Agilex_Cobot_Magic_connect_block_qced_hardlink",
#         "Agilex_Cobot_Magic_erase_board_passing_left_to_right_qced_hardlink",
#         "Agilex_Cobot_Magic_erase_board_passing_right_to_left_qced_hardlink",
#         "Agilex_Cobot_Magic_erase_board_right_qced_hardlink",
#         "Agilex_Cobot_Magic_fold_shorts_khaki_qced_hardlink",
#         "Agilex_Cobot_Magic_fold_short_sleeve_black_qced_hardlink",
#         "Agilex_Cobot_Magic_fold_towel_blue_tray_qced_hardlink",
#         "Agilex_Cobot_Magic_fold_towel_grey_tray_qced_hardlink",
#         "Agilex_Cobot_Magic_fold_towel_pink_tray_qced_hardlink",
#         "Agilex_Cobot_Magic_fold_towel_qced_hardlink",
#         "Agilex_Cobot_Magic_fold_towel_yellow_tray_qced_hardlink",
#         "Agilex_Cobot_Magic_move_mouse_pen_black_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_move_mouse_pen_green_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_move_mouse_pen_khaki_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_move_mouse_pen_qced_hardlink",
#         "Agilex_Cobot_Magic_move_mouse_pen_red_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_organize_test_tube_qced_hardlink",
#         "Agilex_Cobot_Magic_pass_object_left_to_right_black_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_pass_object_left_to_right_green_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_pass_object_left_to_right_khaki_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_pass_object_left_to_right_white_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_pass_object_right_to_left_black_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_pass_object_right_to_left_green_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_pass_object_right_to_left_khaki_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_pass_object_right_to_left_red_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_pass_object_right_to_left_white_tablecloth_qced_hardlink",
#         "Agilex_Cobot_Magic_storage_fruit_bowl_qced_hardlink",
#         "Agilex_Cobot_Magic_storage_orange_white_bag_qced_hardlink",
#         "Agilex_Cobot_Magic_storage_peach_brown_bag_qced_hardlink",
#         "Agilex_Cobot_Magic_storage_peach_white_bag_qced_hardlink",
#         "Agilex_Split_Aloha_organize_desk_fail_qced_hardlink"
"Agilex_Cobot_Magic_move_object_black_tablecloth_qced_hardlink",
"Agilex_Cobot_Magic_pour_drink_bottle_cup_qced_hardlink",
"Agilex_Cobot_Magic_storage_bread_basket_qced_hardlink"
]
# ===================================================

def calculate_gripper_mode(scale_values):
    """
    根据归一化scale计算夹爪模式
    输入: [left_scale, right_scale]
    输出: [left_mode, right_mode] (int32)
    规则: >0.5 → 0 (open), ≤0.5 →1 (closed)
    """
    modes = []
    for val in scale_values:
        mode = 0 if float(val) > 0.5 else 1
        modes.append(np.int32(mode))
    return np.array(modes, dtype=np.int32)

def process_dataset(dataset_path: Path):
    """处理单个数据集的所有parquet文件"""
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        logger.warning(f"数据目录不存在: {dataset_path}")
        return False

    # 遍历所有parquet
    parquet_files = list(data_dir.rglob("*.parquet"))
    if not parquet_files:
        logger.warning(f"无parquet文件: {dataset_path}")
        return False

    logger.info(f"开始处理 {len(parquet_files)} 个文件...")

    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            modified = False

            # ================= 处理 gripper_mode_state =================
            if "gripper_open_scale_state" in df.columns:
                scale_state = np.vstack(df["gripper_open_scale_state"].values)
                mode_state = [calculate_gripper_mode(row) for row in scale_state]
                df["gripper_mode_state"] = mode_state
                modified = True

            # ================= 处理 gripper_mode_action =================
            if "gripper_open_scale_action" in df.columns:
                scale_action = np.vstack(df["gripper_open_scale_action"].values)
                mode_action = [calculate_gripper_mode(row) for row in scale_action]
                df["gripper_mode_action"] = mode_action
                modified = True

            # 保存
            if modified:
                df.to_parquet(file, index=False)
                logger.info(f"✅ 更新: {file.name}")

        except Exception as e:
            logger.error(f"❌ 处理失败 {file.name}: {str(e)}")

    return True

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("开始批量更新 gripper_mode 字段")
    logger.info("规则：scale>0.5=0(open) | scale≤0.5=1(closed)")
    logger.info("="*60)

    success = 0
    failed = 0

    for name in DATASET_LIST:
        path = BASE_DIR / name
        logger.info(f"\n【处理数据集】{name}")

        try:
            process_dataset(path)
            success += 1
            logger.info(f"✅ 完成: {name}")
        except Exception as e:
            failed += 1
            logger.error(f"❌ 失败: {name} | {str(e)}")

    # 最终统计
    logger.info("\n" + "="*60)
    logger.info(f"处理完成 | 成功：{success} 个 | 失败：{failed} 个")
    logger.info("="*60)
