import logging
from collections import defaultdict
import sys
from typing import List, Dict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from sqlalchemy.orm import Session
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, TaskStatus
import json
import numpy as np
import pandas as pd

DB_FILE_PATH = "db/postgresql_config.yaml"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_validation_report7.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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



# ==================== ✅ 新增：自动从info.json获取gripper_open索引并修复负值 ====================
def fix_gripper_open_negatives(dataset_path: Path):
    # 找到info.json
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        info_path = dataset_path / "info.json"
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    # 自动获取所有gripper_open索引
    def get_gripper_indices(feature_key):
        names = info['features'][feature_key]['names']
        return [i for i, n in enumerate(names) if 'gripper_open' in n]

    state_indices = get_gripper_indices('observation.state')
    action_indices = get_gripper_indices('action')

    # 遍历parquet
    data_dir = dataset_path / "data"
    parquet_files = list(data_dir.rglob("*.parquet"))
    modified_count = 0

    for fp in parquet_files:
        try:
            df = pd.read_parquet(fp)
            changed = False

            # 修复 observation.state
            if 'observation.state' in df.columns:
                arr = np.array(df['observation.state'].to_list(), dtype=np.float32)
                for dim in state_indices:
                    mask = arr[:, dim] < 0.0
                    if np.any(mask):
                        arr[mask, dim] = 0.0
                        changed = True
                df['observation.state'] = arr.tolist()

            # 修复 action
            if 'action' in df.columns:
                arr = np.array(df['action'].to_list(), dtype=np.float32)
                for dim in action_indices:
                    mask = arr[:, dim] < 0.0
                    if np.any(mask):
                        arr[mask, dim] = 0.0
                        changed = True
                df['action'] = arr.tolist()

            if changed:
                df.to_parquet(fp, index=False)
                modified_count += 1
        except Exception as e:
            logger.warning(f"修复文件失败 {fp.name}: {str(e)}")

    logger.info(f"🔧 数据集修复完成：处理 {len(parquet_files)} 个文件，修改 {modified_count} 个文件")

# ===============================================================================================

def run_validation_on_all_datasets():
    db = DatasetDatabase(Path(DB_FILE_PATH).expanduser().absolute())
    valid_datasets: List[Dict] = []
    invalid_datasets: List[Dict] = []

    with db.with_session() as session:
        datasets: List[DatasetDB] = (
            session.query(DatasetDB)
            .filter(DatasetDB.dataset_uuid.in_(TARGET_UUIDS))
            .all()
        )

        if not datasets:
            logger.info("✅ 数据库中无待校验的数据集")
            return

        logger.info(f"📊 共找到 {len(datasets)} 个转换完成的数据集，开始批量校验...\n")

        for idx, ds in enumerate(datasets, 1):
            dataset_uuid = ds.dataset_uuid
            convert_path = ds.convert_path
            device_model = ds.device_model.strip() if ds.device_model else "未知设备"

            logger.info(f"===== 正在校验第 {idx}/{len(datasets)} 个数据集 =====")
            logger.info(f"UUID: {dataset_uuid}")
            logger.info(f"设备型号: {device_model}")
            logger.info(f"路径: {convert_path}")

            def mark_as_invalid_and_ignore(error_desc: str):
                invalid_datasets.append({"uuid": dataset_uuid, "device_model": device_model})
                ds.is_ignore = True
                session.commit()
                logger.error(f"❌ {error_desc}")
                logger.info(f"✅ 已自动标记该数据集 is_ignore = True\n")

            if not convert_path or not Path(convert_path).exists():
                mark_as_invalid_and_ignore("数据集异常：路径不存在/为空，已跳过")
                continue

            try:
                # 👇 只在这里加一行：自动修复gripper_open负值
                fix_gripper_open_negatives(Path(convert_path))

                # 👇 原有逻辑完全不动
                validator = LerobotDatasetValidator(convert_path)
                is_valid, error_msg = validator.run()

                if is_valid:
                    valid_datasets.append({"uuid": dataset_uuid, "device_model": device_model})
                    ds.is_ignore = False
                    session.commit()
                    logger.info(f"✅ 数据集校验通过")
                else:
                    mark_as_invalid_and_ignore(f"数据集存在异常：{error_msg[:200]}...")

            except Exception as e:
                mark_as_invalid_and_ignore(f"校验程序异常：{str(e)}")

    generate_final_report(valid_datasets, invalid_datasets)

def calculate_device_ratio(dataset_list: List[Dict]) -> Dict:
    total = len(dataset_list)
    device_count = defaultdict(int)
    for item in dataset_list:
        device_count[item["device_model"]] += 1

    device_stats = {}
    for device, count in device_count.items():
        ratio = (count / total * 100) if total > 0 else 0.0
        device_stats[device] = {"count": count, "ratio": round(ratio, 2)}
    return device_stats

def generate_final_report(valid_ds: List[Dict], invalid_ds: List[Dict]):
    total_count = len(valid_ds) + len(invalid_ds)
    valid_count = len(valid_ds)
    invalid_count = len(invalid_ds)

    valid_device_stats = calculate_device_ratio(valid_ds)
    invalid_device_stats = calculate_device_ratio(invalid_ds)

    logger.info("=" * 120)
    logger.info("📋 数据集校验最终汇总报告（放宽弧度阈值、自动修复gripper_open负值）")
    logger.info("=" * 120)
    logger.info(f"📊 总计校验数据集：{total_count} 个")
    logger.info(f"✅ 正常数据集：{valid_count} 个")
    logger.info(f"❌ 异常数据集：{invalid_count} 个")
    logger.info("-" * 120)

    logger.info("\n📡 【正常数据集】设备型号统计：")
    if valid_device_stats:
        for device, stats in valid_device_stats.items():
            logger.info(f"   {device}：{stats['count']} 个，占比 {stats['ratio']}%")
    else:
        logger.info("   无正常数据集")

    logger.info("\n📡 【异常数据集】设备型号统计：")
    if invalid_device_stats:
        for device, stats in invalid_device_stats.items():
            logger.info(f"   {device}：{stats['count']} 个，占比 {stats['ratio']}%")
    else:
        logger.info("   无异常数据集")

    logger.info("\n" + "-" * 120)
    logger.info("\n✅ 正常数据集 UUID 列表：")
    for item in valid_ds:
        logger.info(f"   [{item['device_model']}] {item['uuid']}")

    logger.info("\n❌ 异常数据集 UUID 列表：")
    for item in invalid_ds:
        logger.info(f"   [{item['device_model']}] {item['uuid']}")

    logger.info("=" * 120)

# ==================== 仅检测、不修复、阈值放宽校验器（完全没动） ====================
class LerobotDatasetValidator:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path).expanduser().absolute()
        self.info_path = self.dataset_path / "meta" / "info.json"
        self.data_dir = self.dataset_path / "data"
        if not self.info_path.exists():
            self.info_path = self.dataset_path / "info.json"
        self._validate_paths()
        self.info_data = self._load_info_json()
        self.expected_features = self.info_data.get('features', {})
        self.feature_names_map = self._extract_feature_names()

        self.FINE_GRAINED_BOUNDS = {
            "_rad": {"min": -3.1425926, "max": 3.1425926},
            "_m": {"min": -2.0, "max": 2.0},
            "gripper_open_scale": {"min": 0.0, "max": 1.0},
            "gripper_open": {"min": 0.0, "max": 1010.0},
        }
        self.GLOBAL_FALLBACK_BOUNDS = {"min": -10.0, "max": 10.0}

    def _validate_paths(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
        if not self.info_path.exists():
            raise FileNotFoundError(f"找不到 info.json: {self.info_path}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

    def _load_info_json(self) -> dict:
        with open(self.info_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_feature_names(self) -> Dict[str, List[str]]:
        names_map = {}
        for feat_name, feat_info in self.expected_features.items():
            if "names" in feat_info and isinstance(feat_info["names"], list):
                names_map[feat_name] = feat_info["names"]
        return names_map

    def _check_single_parquet_fast(self, file_path: Path) -> str:
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            return [f"Parquet读取失败: {str(e)}"]

        for feature_name, feature_info in self.expected_features.items():
            if not (feature_name.startswith("action") or feature_name.startswith("observation.state")):
                continue
            if feature_name not in df.columns:
                errors.append(f"缺失特征列: {feature_name}")
                continue

            expected_shape = tuple(feature_info.get('shape', []))
            dtype = feature_info.get('dtype')
            if not expected_shape or dtype in ['video', 'image']:
                continue

            try:
                np_data = np.vstack(df[feature_name].values)
            except ValueError:
                errors.append(f"特征 {feature_name} 数据格式错乱")
                continue

            actual_shape = np_data.shape[1:]
            if actual_shape != expected_shape:
                errors.append(f"{feature_name} 维度不匹配: 预期{expected_shape} 实际{actual_shape}")

            if np_data.dtype.kind in 'fc':
                if np.isnan(np_data).any() or np.isinf(np_data).any():
                    errors.append(f"{feature_name} 存在NaN/Inf异常值")

            if feature_name in self.feature_names_map:
                sub_names = self.feature_names_map[feature_name]
                for col_idx, sub_name in enumerate(sub_names):
                    if col_idx >= np_data.shape[1]:
                        continue
                    col_data_np = np_data[:, col_idx]
                    matched_bounds = None
                    for keyword, bounds in self.FINE_GRAINED_BOUNDS.items():
                        if keyword in sub_name:
                            matched_bounds = bounds
                            break
                    if matched_bounds:
                        hard_min, hard_max = matched_bounds["min"], matched_bounds["max"]
                        invalid_mask = (col_data_np < hard_min - 1e-4) | (col_data_np > hard_max + 1e-4)
                        if np.any(invalid_mask):
                            invalid_frame_idx = np.argmax(invalid_mask)
                            invalid_value = col_data_np[invalid_frame_idx]
                            return (
                                f"【精确异常】{feature_name}.{sub_name} 超出放宽后物理极值\n"
                                f"   异常帧索引：{invalid_frame_idx}\n"
                                f"   异常数值：{invalid_value:.6f}\n"
                                f"   放宽允许范围：[{hard_min}, {hard_max}]"
                            )
            else:
                actual_min, actual_max = np.min(np_data), np.max(np_data)
                if actual_min < self.GLOBAL_FALLBACK_BOUNDS["min"] or actual_max > self.GLOBAL_FALLBACK_BOUNDS["max"]:
                    errors.append(f"{feature_name} 存在飞点数据(允许:[{self.GLOBAL_FALLBACK_BOUNDS['min']}, {self.GLOBAL_FALLBACK_BOUNDS['max']}], 实际:[{actual_min:.4f}, {actual_max:.4f}])")

        return errors

    def run(self) -> tuple[bool, str]:
        parquet_files = list(self.data_dir.rglob("*.parquet"))
        if not parquet_files:
            return False, "无Parquet文件"

        for file in parquet_files:
            errors = self._check_single_parquet(file)
            if errors:
                for err in errors:
                    all_errors.append(f"文件{file.name}: {err}")

        if all_errors:
            return False, " | ".join(all_errors)
        return True, ""

if __name__ == "__main__":
    run_validation_on_all_datasets()
