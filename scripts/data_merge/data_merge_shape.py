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

# ====================== 只处理这些 UUID ======================
TARGET_UUIDS = [
     '7b86b017-93e7-4031-9f8e-be2ae24a8f4e',
    'd4e9a4ab-1a29-4636-9bb8-1f5183e429d2',
    '203e1be7-f38a-49a6-b1ae-11477c673f79',
    '983154c9-dcf9-446b-b5ee-2707b470f185',
    'aa677292-5a32-4e80-ab78-0aa598818f4c',
    '0298514b-57cd-4d91-913a-81bce8b531f5',
    'c5049fbd-1ffe-41b3-82f9-9906a5dd9d4d',
    '12c90605-38c4-4fce-b333-8a75fb70309d',
    '4bdf2875-4362-44c6-a196-0257414dc31f',
    '92423310-27fd-4dad-b980-d6beee4745d0',
    '21d87d92-9aef-4aa2-b370-d702ce87f24e',
    '09c1f613-16b7-4bd9-8885-6c6c56880804',
    'd856a081-4998-43c9-8ba9-1b28f52a0f6e',
    '119b6a9a-5408-4f57-83fe-4156b1afd9f0',
    '60cf7d0f-2656-4819-baab-ce1b9ecb5bf1',
    '22417619-aae5-4c2d-b18b-d07990f5fcac',
    'fc6e3f46-105a-456c-afc4-b3ca8e701857',
    '0b8d5fcf-c684-4e26-9c6a-5ddca5d94ede',
    '774e7025-eb94-4c42-841f-ab64f9849b1f',
    'ecebb15f-6434-4ba0-bb23-73f5be418b3b',
    '943e9777-7526-4d63-8000-b8b9b20942f4',
    '8c8eae1d-086d-4ef3-b908-bb55c1d82660',
    '615bb6b3-b56c-4608-ac3b-73231a7f8844'
]

# ====================== 标准 dtype + shape ======================
FIELD_SPECS = {
    "timestamp":              {"dtype": "float32", "shape": [1]},
    "eef_sim_pose_state":     {"dtype": "float32", "shape": [12]},
    "eef_sim_pose_action":    {"dtype": "float32", "shape": [12]},
    "gripper_open_scale_state":  {"dtype": "float32", "shape": [2]},
    "gripper_open_scale_action": {"dtype": "float32", "shape": [2]},

    "frame_index":            {"dtype": "int64",   "shape": [1]},
    "episode_index":          {"dtype": "int64",   "shape": [1]},
    "index":                  {"dtype": "int64",   "shape": [1]},
    "task_index":             {"dtype": "int64",   "shape": [1]},

    "subtask_annotation":     {"dtype": "int32",   "shape": [5]},
    "scene_annotation":       {"dtype": "int32",   "shape": [1]},

    "eef_direction_state":    {"dtype": "int32",   "shape": [2]},
    "eef_direction_action":   {"dtype": "int32",   "shape": [2]},
    "eef_velocity_state":     {"dtype": "int32",   "shape": [2]},
    "eef_velocity_action":    {"dtype": "int32",   "shape": [2]},
    "eef_acc_mag_state":      {"dtype": "int32",   "shape": [2]},
    "eef_acc_mag_action":     {"dtype": "int32",   "shape": [2]},

    "gripper_mode_state":     {"dtype": "int32",   "shape": [2]},
    "gripper_mode_action":    {"dtype": "int32",   "shape": [2]},
    "gripper_activity_state": {"dtype": "int32",   "shape": [2]},
    "gripper_activity_action": {"dtype": "int32",  "shape": [2]},
}

# ====================== 修复 dtype + shape ======================
def fix_feature_fields(info_path: Path):
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "features" not in data:
            return False

        features = data["features"]
        modified = False

        for key, spec in FIELD_SPECS.items():
            if key not in features:
                continue

            feat = features[key]
            correct_dtype = spec["dtype"]
            correct_shape = spec["shape"]

            # 修正 dtype
            if feat.get("dtype") != correct_dtype:
                feat["dtype"] = correct_dtype
                modified = True
                logger.info(f"  ✅ {key}  dtype = {correct_dtype}")

            # 修正 shape（缺失或不对都补）
            if feat.get("shape") != correct_shape:
                feat["shape"] = correct_shape
                modified = True
                logger.info(f"  ✅ {key}  shape = {correct_shape}")

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

    with db.with_session() as session:
        datasets = session.query(DatasetDB).filter(
            DatasetDB.dataset_uuid.in_(TARGET_UUIDS),
            DatasetDB.qc_status == "COMPLETED",
            DatasetDB.qced_repo_gen_status == "COMPLETED"
        ).all()

        logger.info(f"\n符合条件数据集：{len(datasets)} 个")
        logger.info(f"🎯 目标UUID数量：{len(TARGET_UUIDS)}")

        for ds in datasets:
            try:
                p = ds.qced_repo_gen_path
                if p:
                    path_list.append(p)
                    logger.info(f"✅ 加入处理：{ds.id}")
            except:
                continue

    fixed = 0
    for p in path_list:
        root = Path(p)
        info = root / "meta" / "info.json"
        if not info.exists():
            info = root / "info.json"
        if not info.exists():
            continue

        logger.info(f"\n处理：{info}")
        if fix_feature_fields(info):
            fixed += 1

    logger.info("\n" + "=" * 50)
    logger.info(f"✅ 全部完成！修复 info.json：{fixed} 个")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
