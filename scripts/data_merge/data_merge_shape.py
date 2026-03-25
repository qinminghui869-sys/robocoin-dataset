import json
import logging
from pathlib import Path

# 数据库相关
from sqlalchemy.orm import Session
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB

# ====================== 配置 ======================
DB_CONFIG = "db/postgresql_config.yaml"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ====================== 【严格按照你的标准】 dtype 规则 ======================
DTYPE_MAP = {
    "timestamp": "float32",
    "eef_sim_pose_state": "float32",
    "eef_sim_pose_action": "float32",
    "gripper_open_scale_state": "float32",
    "gripper_open_scale_action": "float32",

    # 整数
    "frame_index": "int64",
    "episode_index": "int64",
    "index": "int64",
    "task_index": "int64",
    "subtask_annotation": "int32",
    "scene_annotation": "int32",
    "eef_direction_state": "int32",
    "eef_direction_action": "int32",
    "eef_velocity_state": "int32",
    "eef_velocity_action": "int32",
    "eef_acc_mag_state": "int32",
    "eef_acc_mag_action": "int32",
    "gripper_mode_state": "int32",
    "gripper_mode_action": "int32",
    "gripper_activity_state": "int32",
    "gripper_activity_action": "int32",
}

# ====================== 补全缺失的 dtype ======================
def fix_all_dtype(info_path: Path):
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "features" not in data:
            return False

        features = data["features"]
        modified = False

        for key, correct_dtype in DTYPE_MAP.items():
            if key not in features:
                continue

            feat = features[key]
            if feat.get("dtype") == correct_dtype:
                continue  # 已经正确，不动

            # 补全 / 修正为正确的 dtype
            feat["dtype"] = correct_dtype
            modified = True
            logger.info(f"  ✅ {key}  dtype = {correct_dtype}")

        if modified:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return modified

    except Exception as e:
        logger.error(f"  ❌ 失败: {e}")
        return False

# ====================== 主函数 ======================
def main():
    db = DatasetDatabase(Path(DB_CONFIG).expanduser().absolute())
    path_list = []

    # 1. 取出所有符合条件的路径
    with db.with_session() as session:
        datasets = session.query(DatasetDB).filter(
            DatasetDB.qc_status == "COMPLETED",
            DatasetDB.qced_repo_gen_status == "COMPLETED"
        ).all()

        logger.info(f"\n符合条件数据集：{len(datasets)} 个")

        for ds in datasets:
            try:
                p = ds.qced_repo_gen_path
                if p:
                    path_list.append(p)
            except:
                continue

    # 2. 批量补全 dtype
    fixed = 0
    for p in path_list:
        root = Path(p)
        info = root / "meta" / "info.json"
        if not info.exists():
            info = root / "info.json"
        if not info.exists():
            continue

        logger.info(f"\n处理：{info}")
        if fix_all_dtype(info):
            fixed += 1

    logger.info("\n" + "=" * 50)
    logger.info(f"✅ 全部完成！修复 dtype 文件：{fixed} 个")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
