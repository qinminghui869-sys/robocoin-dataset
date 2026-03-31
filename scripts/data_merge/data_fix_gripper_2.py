import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional
import sys

# 自动加载你的数据库配置
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_FILE_PATH = "db/postgresql_config.yaml"

# ===================== 在这里填入你要归一化的 UUID =====================
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
# ====================================================================

class GripperOpenNormalizer:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path).expanduser().absolute()
        self.info_path = self.dataset_path / "meta" / "info.json"
        self.stats_path = self.dataset_path / "meta" / "episodes_stats.jsonl"
        self.data_dir = self.dataset_path / "data"
        self._validate_paths()
        self.info_data = self._load_info_json()
        self.gripper_indices = self._find_gripper_indices()
        self.valid_gripper_types = self._get_valid_gripper_types()
        self.gripper_stats = {}
        self.gripper_ranges = {}

    def _validate_paths(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
        if not self.info_path.exists():
            raise FileNotFoundError(f"info.json不存在: {self.info_path}")
        if not self.stats_path.exists():
            raise FileNotFoundError(f"episodes_stats.jsonl不存在: {self.stats_path}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

    def _load_info_json(self) -> dict:
        with open(self.info_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _find_gripper_indices(self) -> Dict[str, Dict[str, Optional[int]]]:
        indices = {}
        state_names = self.info_data['features']['observation.state']['names']
        obs_indices = {}
        try:
            obs_indices['left_gripper_open'] = state_names.index('left_gripper_open')
        except ValueError:
            obs_indices['left_gripper_open'] = None
        try:
            obs_indices['right_gripper_open'] = state_names.index('right_gripper_open')
        except ValueError:
            obs_indices['right_gripper_open'] = None
        indices['observation.state'] = obs_indices

        action_names = self.info_data['features']['action']['names']
        act_indices = {}
        try:
            act_indices['left_gripper_open'] = action_names.index('left_gripper_open')
        except ValueError:
            act_indices['left_gripper_open'] = None
        try:
            act_indices['right_gripper_open'] = action_names.index('right_gripper_open')
        except ValueError:
            act_indices['right_gripper_open'] = None
        indices['action'] = act_indices
        return indices

    def _get_valid_gripper_types(self) -> List[str]:
        valid_types = []
        for g in ['left_gripper_open', 'right_gripper_open']:
            if self.gripper_indices['observation.state'][g] is not None or self.gripper_indices['action'][g] is not None:
                valid_types.append(g)
        return valid_types

    def _detect_gripper_range(self, min_val: float, max_val: float) -> int:
        ranges = [(1.1, 1), (10.5, 10), (105.0, 100), (1050.0, 1000)]
        max_abs = max(abs(min_val), abs(max_val))
        for threshold, range_val in ranges:
            if max_abs <= threshold:
                return range_val
        logger.warning(f"无法识别量程，默认使用 0-1")
        return 1

    def collect_gripper_stats(self):
        stats = {g: {'min': float('inf'), 'max': -float('inf')} for g in self.valid_gripper_types}
        with open(self.stats_path, 'r', encoding='utf-8') as f:
            for line in f:
                es = json.loads(line)
                st = es['stats']
                for g in self.valid_gripper_types:
                    idx = self.gripper_indices['observation.state'][g]
                    if idx is not None and idx < len(st.get('observation.state', {}).get('min', [])):
                        mn = st['observation.state']['min'][idx]
                        mx = st['observation.state']['max'][idx]
                        stats[g]['min'] = min(stats[g]['min'], mn)
                        stats[g]['max'] = max(stats[g]['max'], mx)
        need_re = False
        for g in self.valid_gripper_types:
            if stats[g]['min'] >= stats[g]['max']:
                need_re = True
                break
        if need_re:
            stats = self._collect_stats_from_parquet()
        for g in self.valid_gripper_types:
            self.gripper_ranges[g] = self._detect_gripper_range(stats[g]['min'], stats[g]['max'])
        self.gripper_stats = stats
        return stats

    def _collect_stats_from_parquet(self):
        stats = {g: {'min': float('inf'), 'max': -float('inf')} for g in self.valid_gripper_types}
        vals = {g: [] for g in self.valid_gripper_types}
        for f in self.data_dir.rglob("*.parquet"):
            df = pd.read_parquet(f)
            if 'observation.state' in df.columns:
                arr = np.vstack(df['observation.state'])
                for g in self.valid_gripper_types:
                    idx = self.gripper_indices['observation.state'][g]
                    if idx is not None:
                        vals[g].extend(arr[:, idx])
            if 'action' in df.columns:
                arr = np.vstack(df['action'])
                for g in self.valid_gripper_types:
                    idx = self.gripper_indices['action'][g]
                    if idx is not None:
                        vals[g].extend(arr[:, idx])
        for g in self.valid_gripper_types:
            if vals[g]:
                stats[g]['min'] = np.min(vals[g])
                stats[g]['max'] = np.max(vals[g])
        return stats

    def normalize_gripper_value(self, v, gtype):
        rng = self.gripper_ranges.get(gtype, 1)
        return float(np.clip(v / rng, 0.0, 1.0))

    def process_parquet_files(self):
        if not self.gripper_stats:
            self.collect_gripper_stats()
        files = list(self.data_dir.rglob("*.parquet"))
        for f in files:
            try:
                df = pd.read_parquet(f)
                mod = False
                if 'observation.state' in df.columns:
                    arr = np.vstack(df['observation.state'])
                    nv = []
                    for g in self.valid_gripper_types:
                        idx = self.gripper_indices['observation.state'][g]
                        if idx is not None:
                            nv.append([self.normalize_gripper_value(x, g) for x in arr[:, idx]])
                    if nv:
                        df['gripper_open_scale_state'] = list(np.column_stack(nv))
                        mod = True
                if 'action' in df.columns:
                    arr = np.vstack(df['action'])
                    nv = []
                    for g in self.valid_gripper_types:
                        idx = self.gripper_indices['action'][g]
                        if idx is not None:
                            nv.append([self.normalize_gripper_value(x, g) for x in arr[:, idx]])
                    if nv:
                        df['gripper_open_scale_action'] = list(np.column_stack(nv))
                        mod = True
                if mod:
                    df.to_parquet(f)
                    logger.info(f"已处理: {f.name}")
            except Exception as e:
                logger.error(f"失败 {f.name}: {e}")
        return True

    def update_info_json(self):
        for k in ['gripper_open_scale_state', 'gripper_open_scale_action']:
            if k in self.info_data['features']:
                del self.info_data['features'][k]
        names = [g.replace("_open", "_open_scale") for g in self.valid_gripper_types]
        shape = [len(names)] if len(names) > 1 else [1]
        self.info_data['features']['gripper_open_scale_state'] = {"names": names, "dtype": "float32", "shape": shape}
        self.info_data['features']['gripper_open_scale_action'] = {"names": names, "dtype": "float32", "shape": shape}
        with open(self.info_path, 'w', encoding='utf-8') as f:
            json.dump(self.info_data, f, indent=2, ensure_ascii=False)
        return True

    def update_episodes_stats(self):
        new_lines = []
        with open(self.stats_path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                epi = d['episode_index']
                pq = None
                for cand in self.data_dir.rglob(f"episode_{epi:06d}.parquet"):
                    pq = cand
                    break
                if not pq:
                    new_lines.append(line.strip())
                    continue
                df = pd.read_parquet(pq)
                for col in ['gripper_open_scale_state', 'gripper_open_scale_action']:
                    if col in df.columns:
                        arr = np.vstack(df[col])
                        d['stats'][col] = {
                            "min": arr.min(0).tolist(),
                            "max": arr.max(0).tolist(),
                            "mean": arr.mean(0).tolist(),
                            "std": arr.std(0).tolist(),
                            "count": [len(arr)]
                        }
                new_lines.append(json.dumps(d))
        with open(self.stats_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        return True

    def run(self):
        logger.info(f"开始处理数据集: {self.dataset_path}")
        self.collect_gripper_stats()
        self.process_parquet_files()
        self.update_info_json()
        self.update_episodes_stats()
        logger.info("✅ 归一化完成！\n")
        return True

# ==================== 批量通过 UUID 自动执行 ====================
def get_path_by_uuid(uuid):
    db = DatasetDatabase(Path(DB_FILE_PATH).expanduser())
    with db.with_session() as s:
        ds = s.query(DatasetDB).filter(DatasetDB.dataset_uuid == uuid).first()
        return ds.convert_path if ds else None

if __name__ == "__main__":
    logger.info("🚀 开始批量归一化夹爪数据（根据UUID自动处理）")
    for uid in TARGET_UUIDS:
        path = get_path_by_uuid(uid)
        if not path:
            logger.error(f"❌ UUID 不存在: {uid}")
            continue
        try:
            norm = GripperOpenNormalizer(path)
            norm.run()
        except Exception as e:
            logger.error(f"❌ 处理失败 {uid}: {e}")
    logger.info("🎉 所有 UUID 归一化任务完成！")
