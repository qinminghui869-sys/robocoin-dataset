from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB

# ====================== 在这里填你要检查的 UUID ======================
TARGET_UUIDS = [
    "d4e9a4ab-1a29-4636-9bb8-1f5183e429d2",
    "aa677292-5a32-4e80-ab78-0aa598818f4c",
    "c5049fbd-1ffe-41b3-82f9-9906a5dd9d4d",
    "983154c9-dcf9-446b-b5ee-2707b470f185",
    "21d87d92-9aef-4aa2-b370-d702ce87f24e",
    "8c8eae1d-086d-4ef3-b908-bb55c1d82660",
]
# ====================================================================

DB_FILE_PATH = "db/postgresql_config.yaml"

def get_convert_path_by_uuid(dataset_uuid: str) -> str:
    db = DatasetDatabase(Path(DB_FILE_PATH).expanduser().absolute())
    with db.with_session() as session:
        ds = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
        return ds.convert_path if ds else None

def inspect_bad_data_by_uuid(dataset_uuid: str):
    convert_path = get_convert_path_by_uuid(dataset_uuid)
    if not convert_path or not Path(convert_path).exists():
        print(f"❌ UUID {dataset_uuid} 路径不存在")
        return

    print(f"\n🚀 开始检查 UUID: {dataset_uuid}")
    print(f"📂 路径: {convert_path}")

    root = Path(convert_path)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        info_path = root / "info.json"

    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    features = info['features']
    parquet_files = list((root / "data").rglob("*.parquet"))

    bounds_map = {
        "_rad": (-3.1425926, 3.1425926),
        "_m": (-2.0, 2.0),
        "gripper_open": (0.0, 1010.0),
        "gripper_open_scale": (0.0, 1.0),
    }

    # 只存坏文件路径
    bad_files = set()

    for fp in parquet_files:
        try:
            df = pd.read_parquet(fp)
        except:
            bad_files.add(str(fp))
            continue

        for feat_name in ["observation.state", "action"]:
            if feat_name not in df.columns:
                continue

            names = features[feat_name]["names"]
            arr = np.array(df[feat_name].to_list(), dtype=np.float32)

            for dim_idx, dim_name in enumerate(names):
                data = arr[:, dim_idx]
                min_val, max_val = None, None

                for key, (vmin, vmax) in bounds_map.items():
                    if key in dim_name:
                        min_val = vmin
                        max_val = vmax
                        break

                if min_val is None or max_val is None:
                    continue

                # 只要有任一越界，就标记这个文件坏了
                if np.any((data < min_val - 1e-4) | (data > max_val + 1e-4)):
                    bad_files.add(str(fp))
                    break  # 只要发现坏，直接跳过这个字段

    # 最终输出
    if bad_files:
        print(f"🚨 发现【{len(bad_files)} 个坏 parquet】，建议整文件删除：")
        for f in sorted(bad_files):
            print(f"   🗑️ {f}")
    else:
        print(f"✅ 无坏文件")

if __name__ == "__main__":
    print("🔍 批量检查坏数据（仅输出坏文件）")
    print("=" * 80)
    for uuid in TARGET_UUIDS:
        inspect_bad_data_by_uuid(uuid)
    print("\n🎉 所有UUID检查完成！")
