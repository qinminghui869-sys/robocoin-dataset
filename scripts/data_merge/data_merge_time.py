import json
import logging
from pathlib import Path
from collections import defaultdict

# 数据库相关
from sqlalchemy.orm import Session
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB

# ====================== 配置 ======================
DB_CONFIG = "db/postgresql_config.yaml"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ====================== 统计函数 ======================
def get_stat_from_info(repo_path: str):
    """从 meta/info.json 读取 episodes 和时长"""
    try:
        base = Path(repo_path)
        info_path = base / "meta" / "info.json"
        if not info_path.exists():
            info_path = base / "info.json"
        if not info_path.exists():
            return None

        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        total_episodes = info.get("total_episodes", 0)
        total_frames = info.get("total_frames", 0)
        fps = info.get("fps", 30)
        hours = total_frames / fps / 3600 if fps else 0

        return {
            "episodes": total_episodes,
            "hours": round(hours, 4)
        }
    except Exception as e:
        logger.warning(f"读取失败: {repo_path} | {e}")
        return None

# ====================== 主函数 ======================
def main():
    db = DatasetDatabase(Path(DB_CONFIG).expanduser().absolute())

    total_episodes = 0
    total_hours = 0.0

    with db.with_session() as session:
        datasets = session.query(DatasetDB).filter(
            DatasetDB.qc_status == "COMPLETED",
            DatasetDB.qced_repo_gen_status == "COMPLETED"
        ).all()

        for ds in datasets:
            path = ds.qced_repo_gen_path
            stat = get_stat_from_info(path)
            if not stat:
                continue

            total_episodes += stat["episodes"]
            total_hours += stat["hours"]

    # ====================== 只输出总和 ======================
    print("===== 所有满足条件数据集 统计总和 =====")
    print(f"总 Episodes 数量：{total_episodes}")
    print(f"总时长（小时）：{total_hours:.2f} h")

if __name__ == "__main__":
    main()
