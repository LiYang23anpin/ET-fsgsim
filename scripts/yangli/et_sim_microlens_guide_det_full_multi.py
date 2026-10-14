#0620: 1. 关闭帧间漂移；2. 最暗到16等，取所有星中最亮的top500（或350，原版）；3. optical efficiency 0.52, quantum effeciency 0.5；4. 每个detector batch生成100帧；5. 输出目录命名为microlens_guide_photsim7_100f；6. jitter 用1s内累积放大到0.6arcsec（万分之五的三分之一）后的逐帧psf模糊；7. PSF bundle指定为microlens/RebinnedPSF_LE_BEST_GD1_Field1_multiband_Defocus0；8. 其他参数同原版。0623: 更新输出目录命名规范，增加可选的最大仿真星数限制.
# ET_GUIDE_OPTICAL_EFFICIENCY=0.52 ET_GUIDE_QUANTUM_EFFICIENCY=0.5 python ./scripts/yangli/et_sim_microlens_guide_det_full_multi.py --indices 1 --parent-run-folder ly5_500star_jit0p6 --output-root /home/liyang/ET7/Results/Microlens/Simulation99m

## [Output] frames_dir=/home/liyang/ETguider/FSG_guide_sims/Micro_simulation/guide_full_1s_test/guide_full_100f_10/batch0_guide_top_ra269.0166_dec-28.1059/frames
## /home/liyang/ETguider/FSG_guide_sims/Micro_simulation是output_root, guide_full_1s_test是parent-run-folder output_run_name, guide_full_100f_10是子目录，batch0_guide_top_ra269.0166_dec-28.1059是更细的子目录，frames是最终存放帧数据的目录。

import argparse
import os
import sys
import subprocess
from pathlib import Path
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


ROOT = Path(__file__).resolve().parent
ET_FSGSIM_ROOT = ROOT.parent.parent
CLUSTER_ET7_ROOT = ET_FSGSIM_ROOT.parent
DEFAULT_PHOTSIM7_ROOT = CLUSTER_ET7_ROOT / "Photsim7"
DEFAULT_CONFIG_XLSX = ET_FSGSIM_ROOT / "config" / "et_inputs_microlens_guider.xlsx"
DEFAULT_ET_FOCALPLANE_ROOT = CLUSTER_ET7_ROOT / "et_focalplane"
DEFAULT_OUTPUT_ROOT = CLUSTER_ET7_ROOT / "Results" / "FSG_guide_sims" / "Micro_simulation"
DEFAULT_ET_DATA_DIR = CLUSTER_ET7_ROOT / "Photsim7-data"


def first_existing_path(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def first_complete_gaia_catalog_dir(*candidates: Path) -> str:
    for candidate in candidates:
        if len(tuple(candidate.glob("healpix_n05_nested_*.csv"))) >= 12 * 32 * 32:
            return str(candidate)
    return first_existing_path(*candidates)


DEFAULT_GAIA_CATALOG_DIR = first_complete_gaia_catalog_dir(
    CLUSTER_ET7_ROOT / "gaia_dr3_19mag",
    Path.home() / "ET7" / "gaia_dr3_19mag",
    Path.home() / "ETguider" / "gaia_dr3_19mag",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="批量运行导星连续仿真（microlens, full），可按序号范围或具体序号执行。"
    )
    parser.add_argument(
        "--indices",
        default="2-99",
        help="运行序号，支持 2-12、2,5,9、2-5,9,11-12（默认 2-99）",
    )
    parser.add_argument(
        "--parent-run-folder",
        default="micro_full_100f_10p5m_0523",
        help="输出 run 的父目录名",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="输出根目录（默认 cluster_mount/cluster_ET7/Results/FSG_guide_sims/Micro_simulation）",
    )
    # parser.add_argument(
    #     "--target-frames",
    #     type=int,
    #     default=100,
    #     help="每个 detector batch 生成的帧数（写入 ET_PROFILE_TARGET_FRAMES，默认 100）",
    # )
    # parser.add_argument(
    #     "--photsim7-root",
    #     default=str(DEFAULT_PHOTSIM7_ROOT),
    #     help="当前 Photsim7 源码根目录",
    # )
    # parser.add_argument(
    #     "--data-dir",
    #     default=str(DEFAULT_ET_DATA_DIR),
    #     help="ET_DATA_DIR；PSF bundle 将从其 psf/et 子目录解析",
    # )
    # parser.add_argument(
    #     "--config-xlsx",
    #     default=str(DEFAULT_CONFIG_XLSX),
    #     help="微引力导星配置表格路径",
    # )
    # parser.add_argument(
    #     "--focalplane-root",
    #     default=str(DEFAULT_ET_FOCALPLANE_ROOT),
    #     help="et_focalplane 根目录",
    # )
    # parser.add_argument(
    #     "--catalog-dir",
    #     default=str(DEFAULT_GAIA_CATALOG_DIR),
    #     help="Gaia DR3 19mag 星表目录",
    # )
    parser.add_argument(
        "--max-sim-stars",
        type=int,
        default=None,
        help="可选：限制参与仿真的星数（烟测可设小一些）",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=12345,
        help="随机种子基数，单个 run 种子为 seed_base + index",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,  # 0 or 1
        help="CUDA 设备号（写入 CUDA_VISIBLE_DEVICES）",
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
    target = ROOT / "et_sim_microlens_guide_det_full.py"

    if not target.exists():
        raise FileNotFoundError(f"找不到目标脚本: {target}")
        
    indices = parse_indices(args.indices)
    parent_run_folder = args.parent_run_folder
    seed_base = args.seed_base
    gpu_id = args.gpu_id
    total = len(indices)

    for run_no, i in enumerate(indices, start=1):
        run_name = f"{parent_run_folder}/micro_single_{i:02d}"
        run_seed = seed_base + i

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # env["PHOTSIM7_ROOT"] = str(Path(args.photsim7_root).expanduser())
        # env["ET_DATA_DIR"] = str(Path(args.data_dir).expanduser())
        # env["ET_CONFIG_XLSX"] = str(Path(args.config_xlsx).expanduser())
        # env["ET_FOCALPLANE_ROOT"] = str(Path(args.focalplane_root).expanduser())
        # env["GUIDE_GAIA_CATALOG_DIR"] = str(Path(args.catalog_dir).expanduser())
        env.setdefault(
            "ET_OUTPUT_ROOT_OVERRIDE", str(Path(args.output_root).expanduser())
        )
        # env["ET_PSF_BUNDLE_NAME_OVERRIDE"] = "microlens/RebinnedPSF_LE_BEST_GD1_Field1_multiband_Defocus0"
        # env["ET_RUN_ALL_BATCHES"] = "1"
        # env["ET_EFFECT_PROFILE"] = "full"
        # env["ET_PROFILE_TARGET_FRAMES"] = str(args.target_frames)
        env["ET_OUTPUT_RUN_NAME_OVERRIDE"] = run_name
        env["GLOBAL_SEED"] = str(run_seed)
        if args.max_sim_stars is not None:
            env["ET_MAX_SIM_STARS"] = str(args.max_sim_stars)

        print(
            f"[{run_no}/{total}] 开始运行: {run_name}  GLOBAL_SEED={run_seed}  "
            f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}  "
            f"ET_OUTPUT_ROOT_OVERRIDE={env['ET_OUTPUT_ROOT_OVERRIDE']}  "
        )
        subprocess.run([sys.executable, str(target)], env=env, check=True)

    print(f"全部 {total} 次运行完成。")


if __name__ == "__main__":
    main()
