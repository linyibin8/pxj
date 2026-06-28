#!/usr/bin/env python3
"""清理历史"脏错题"——旧逻辑把智能观察报告里的小节标题/画面描述（题目耗时线索、差异题目、
无新增纸质题目、画面静止…）当成错题写进了正式错题本。新逻辑已不再这样做（观察改走"候选池+
人工确认"），但远端各账号库里仍残留这些噪声错题，需要一次性清掉。

特性：
- 逐个扫描 data/accounts/*.sqlite3（每账号一库）。
- 默认 DRY-RUN：只打印将删除哪些，不动数据。加 --apply 才真正删除。
- --apply 前会自动给每个库做一份 .bak 备份。
- 两类目标：
  1) 命中观察噪声词的错题（标题/证据/错因/题面）。
  2) detection_method='observation' 的自动错题（旧逻辑下基本都是噪声；新逻辑这些改进候选池）。
     用 --keep-observation 可只删噪声词命中项、保留其它 observation 错题。

用法（在 ydz 后端容器或宿主机的 backend 目录）：
    python3 scripts/cleanup_mistake_noise.py            # 预览（dry-run）
    python3 scripts/cleanup_mistake_noise.py --apply    # 备份并删除
    python3 scripts/cleanup_mistake_noise.py --data-dir data --apply
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sqlite3
import sys

# 与后端 _MISTAKE_OBSERVATION_NOISE 保持一致的观察噪声词。
NOISE_MARKERS = (
    "题目耗时线索", "耗时线索", "差异题目", "无新增", "画面", "界面", "屏幕", "截图",
    "静止", "动作证据", "阅读动作", "工作台", "持续显示", "停留", "翻页", "在场",
    "未检测到", "秒拍摄", "无法判定具体题目耗时", "连续", "疑似正在",
)

NOISE_COLUMNS = ("title", "evidence", "error_reason", "question_text")


def matches_noise(row: sqlite3.Row) -> bool:
    for col in NOISE_COLUMNS:
        value = (row[col] or "") if col in row.keys() else ""
        if any(marker in value for marker in NOISE_MARKERS):
            return True
    return False


def scan_db(path: str, keep_observation: bool) -> list[sqlite3.Row]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, title, status, detection_method, evidence, error_reason,
                   question_text, student_answer
            FROM mistake_items
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return []  # 没有 mistake_items 表（空库）
    finally:
        conn.close()
    targets = []
    for row in rows:
        is_noise = matches_noise(row)
        is_observation = (row["detection_method"] or "") == "observation"
        is_candidate = (row["status"] or "") == "candidate"
        # 噪声词命中的一律删；observation 旧自动错题默认删，但保护新的"候选池"(status=candidate)
        # 除非它本身就是噪声。
        if is_noise:
            targets.append(row)
        elif is_observation and not keep_observation and not is_candidate:
            targets.append(row)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="清理历史脏错题（观察噪声）")
    parser.add_argument("--data-dir", default="data", help="数据目录（含 accounts/*.sqlite3），默认 data")
    parser.add_argument("--apply", action="store_true", help="真正删除（默认只预览）")
    parser.add_argument("--keep-observation", action="store_true",
                        help="只删命中噪声词的错题，保留其它 detection_method=observation 的错题")
    args = parser.parse_args()

    pattern = os.path.join(args.data_dir, "accounts", "*.sqlite3")
    db_paths = sorted(glob.glob(pattern))
    if not db_paths:
        print(f"没找到账号库：{pattern}", file=sys.stderr)
        return 1

    grand_total = 0
    for path in db_paths:
        targets = scan_db(path, args.keep_observation)
        if not targets:
            continue
        grand_total += len(targets)
        print(f"\n== {path} ：命中 {len(targets)} 条 ==")
        for row in targets[:20]:
            print(f"   - [{row['status']}/{row['detection_method'] or '?'}] {(row['title'] or '')[:48]}")
        if len(targets) > 20:
            print(f"   …… 其余 {len(targets) - 20} 条略")
        if args.apply:
            backup = path + ".bak"
            if not os.path.exists(backup):
                shutil.copy2(path, backup)
                print(f"   已备份 -> {backup}")
            conn = sqlite3.connect(path)
            try:
                ids = [row["id"] for row in targets]
                conn.executemany("DELETE FROM mistake_items WHERE id=?", [(i,) for i in ids])
                conn.commit()
                print(f"   已删除 {len(ids)} 条")
            finally:
                conn.close()

    print(f"\n合计命中 {grand_total} 条。", "已删除。" if args.apply else "这是预览（dry-run），加 --apply 才会删除。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
