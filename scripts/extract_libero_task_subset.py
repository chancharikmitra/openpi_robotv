"""从 LeRobot v2.0 的 libero 合集里抽出某个 task 的 N 个 episode, 写成独立子集。

输入:  --src   LeRobot v2.0 数据集根目录 (含 data/chunk-XXX/ 与 meta/)
输出:  --dst   输出根目录, 自带完整 data/ + meta/, 可被 LeRobotDataset 直接加载

每个 episode 的 episode_index / task_index / index / frame_index 会重新编号到 [0, N) 连续。
默认深拷贝 parquet 行 (read_table -> write_table), 仅改写四个索引列。
"""
import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_meta(src: Path):
    info = json.loads((src / "meta" / "info.json").read_text())
    tasks = [json.loads(l) for l in (src / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()]
    episodes = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    return info, tasks, episodes


def src_episode_path(src: Path, info: dict, ep_index: int) -> Path:
    rel = info["data_path"].format(
        episode_chunk=ep_index // info["chunks_size"],
        episode_index=ep_index,
    )
    return src / rel


def rewrite_parquet(src_pq: Path, dst_pq: Path, new_ep_index: int, new_global_start: int, new_task_index: int):
    """读 parquet, 改写 episode_index / task_index / index / frame_index 后写到 dst_pq。"""
    table = pq.read_table(src_pq)
    n = table.num_rows
    cols = {name: table.column(name) for name in table.column_names}
    cols["episode_index"] = pa.array([new_ep_index] * n, type=pa.int64())
    cols["task_index"] = pa.array([new_task_index] * n, type=pa.int64())
    cols["index"] = pa.array(list(range(new_global_start, new_global_start + n)), type=pa.int64())
    cols["frame_index"] = pa.array(list(range(n)), type=pa.int64())
    new_table = pa.table(cols, schema=table.schema)
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_pq)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def extract(src: Path, dst: Path, task_index: int, num_episodes: int, force: bool):
    info, tasks, episodes = load_meta(src)

    task_lookup = {t["task_index"]: t["task"] for t in tasks}
    if task_index not in task_lookup:
        raise SystemExit(f"task_index {task_index} 不在 meta/tasks.jsonl 中, 可选: {sorted(task_lookup)[:10]}...")
    task_str = task_lookup[task_index]

    matching = [ep for ep in episodes if ep["tasks"] == [task_str]]
    if len(matching) < num_episodes:
        raise SystemExit(f"task_index {task_index} 只有 {len(matching)} 个 episode, 要 {num_episodes} 个不够")
    matching.sort(key=lambda e: e["episode_index"])
    selected = matching[:num_episodes]
    print(f"task_index={task_index}: '{task_str}'")
    print(f"  原数据集匹配 {len(matching)} eps, 取前 {num_episodes} eps -> {dst}")

    if dst.exists():
        if not force:
            raise SystemExit(f"输出目录 {dst} 已存在, 加 --force 覆盖")
        shutil.rmtree(dst)

    new_episodes = []
    global_cursor = 0
    new_task_idx = 0
    for new_ep_idx, ep in enumerate(selected):
        old_ep_idx = ep["episode_index"]
        length = ep["length"]
        src_pq = src_episode_path(src, info, old_ep_idx)
        dst_rel = info["data_path"].format(
            episode_chunk=new_ep_idx // info["chunks_size"],
            episode_index=new_ep_idx,
        )
        dst_pq = dst / dst_rel
        rewrite_parquet(src_pq, dst_pq, new_ep_idx, global_cursor, new_task_idx)
        new_episodes.append({"episode_index": new_ep_idx, "tasks": [task_str], "length": length})
        global_cursor += length
        print(f"    [{new_ep_idx:3d}] old_ep={old_ep_idx:4d}  len={length:4d}  -> {dst_rel}")

    new_info = dict(info)
    n_eps = len(selected)
    n_chunks = (n_eps + info["chunks_size"] - 1) // info["chunks_size"]
    new_info["total_episodes"] = n_eps
    new_info["total_frames"] = global_cursor
    new_info["total_tasks"] = 1
    new_info["total_chunks"] = n_chunks
    new_info["splits"] = {"train": f"0:{n_eps}"}

    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))
    write_jsonl(dst / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task_str}])
    write_jsonl(dst / "meta" / "episodes.jsonl", new_episodes)

    src_stats = src / "meta" / "stats.json"
    if src_stats.exists():
        shutil.copy(src_stats, dst / "meta" / "stats.json")

    print(f"完成: {n_eps} eps, {global_cursor} frames -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="LeRobot 数据集根目录")
    ap.add_argument("--dst", required=True, type=Path, help="输出子集目录")
    ap.add_argument("--task-index", required=True, type=int, help="原 meta/tasks.jsonl 里的 task_index")
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--force", action="store_true", help="若 dst 已存在, 删除后重写")
    args = ap.parse_args()
    extract(args.src, args.dst, args.task_index, args.num_episodes, args.force)
