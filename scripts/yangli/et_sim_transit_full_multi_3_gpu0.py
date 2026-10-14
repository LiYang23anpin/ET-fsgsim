import argparse
import os
import sys
import subprocess
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量运行导星 1 分钟连续仿真，可按序号范围或具体序号执行。"
    )
    parser.add_argument(
        "--indices",
        default="1",
        help="运行序号，支持 2-12、2,5,9、2-5,9,11-12（默认 2-99）",
    )
    parser.add_argument(
        "--parent-run-folder",
        default="guide_1min_10p5m_100star",
        help="输出 run 的父目录名",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=12345,
        help="随机种子基数，单个 run 种子为 seed_base + index",
    )
    return parser.parse_args()


def parse_indices(indices_spec):
    values = set()
    for raw_part in indices_spec.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            pieces = part.split("-", 1)
            if len(pieces) != 2:
                raise ValueError(f"非法范围写法: {part}")
            start = int(pieces[0].strip())
            end = int(pieces[1].strip())
            if start > end:
                raise ValueError(f"范围起点不能大于终点: {part}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))

    if not values:
        raise ValueError("未解析出任何有效序号，请检查 --indices 参数")

    if min(values) < 0:
        raise ValueError("序号不能为负数")

    return sorted(values)


def main():
    args = parse_args()
    print("[DEBUG] args.indices:", args.indices)
    root = Path(__file__).resolve().parent
    target = root / "et_sim_transit_full_3.py"

    if not target.exists():
        raise FileNotFoundError(f"找不到目标脚本: {target}")

    indices = parse_indices(args.indices)
    print("[DEBUG] parse_indices(args.indices):", indices)
    parent_run_folder = args.parent_run_folder
    seed_base = args.seed_base
    total = len(indices)

    for run_no, i in enumerate(indices, start=1):
        run_name = f"{parent_run_folder}/transit_full_{int(i):02d}"
        run_seed = seed_base + int(i)

        env = os.environ.copy()
        env["ET_OUTPUT_RUN_NAME_OVERRIDE"] = run_name
        env["GLOBAL_SEED"] = str(run_seed)

        print(f"[Run {run_no}/{total}] Suffix={i:02d} Seed={run_seed} -> {run_name}")
        cmd = [
            sys.executable,
            str(target),
            "--parent-run-folder", parent_run_folder,
            "--seed-base", str(seed_base),
            "--indices", str(i)
        ]
        subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    main()

    