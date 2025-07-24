# -*- coding: utf-8 -*-
"""Merge two SAV attention HDF5 files.

Usage:
    python merge_sav_h5.py file_a.h5 file_b.h5 merged.h5

规则:
1. 两个文件的层级均为 /<task>/episode_xxx/frame_xxxx/last_token_attn。
2. 若 task 在两个文件中都存在, 则按已有 episode 序号继续顺延编号 (episode_001, episode_002, ...)。
3. 不修改任何 frame 及其内部结构, 仅做浅拷贝。
"""
from __future__ import annotations
import sys, re, pathlib
# mypy: ignore-errors
import h5py  # type: ignore

def next_ep_index(task_grp):
    """返回 task_grp 下当前最大 episode 序号, 若不存在则返回 0"""
    pattern = re.compile(r"episode_(\d+)")
    idxs = []
    for name in task_grp.keys():  # type: ignore[attr-defined]
        m = pattern.match(name)
        if m:
            idxs.append(int(m.group(1)))
    return max(idxs) if idxs else 0

def merge_one(src_file: h5py.File, dst_file: h5py.File):
    """将 src_file 的内容合并到 dst_file"""
    for task in src_file.keys():  # type: ignore[attr-defined]
        src_task_grp = src_file[task]  # type: ignore[index]
        if task in dst_file:
            dst_task_grp = dst_file[task]  # type: ignore[index]
        else:
            dst_task_grp = dst_file.create_group(task)

        ep_idx = next_ep_index(dst_task_grp)
        for ep_name in sorted(src_task_grp.keys()):  # type: ignore[attr-defined]
            if not ep_name.startswith("episode_"):
                continue
            ep_idx += 1
            new_ep = f"episode_{ep_idx:03d}"
            # 直接复制整个 episode group
            src_file.copy(src_task_grp[ep_name], dst_task_grp, name=new_ep)  # type: ignore[arg-type]

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python merge_sav_h5.py file_a.h5 file_b.h5 merged.h5")
        sys.exit(1)

    path_a, path_b, path_out = map(pathlib.Path, sys.argv[1:])
    with h5py.File(path_out, "w") as fout:
        with h5py.File(path_a, "r") as fa:
            merge_one(fa, fout)
        with h5py.File(path_b, "r") as fb:
            merge_one(fb, fout)
    print(f"✅ 已合并生成: {path_out}") 