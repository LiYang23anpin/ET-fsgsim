# 关闭了frame_to_frame_drift，可在full_3.py中使用环境变量 ET_ENABLE_FRAME_TO_FRAME_DRIFT=1 来开启。
# 给每一个batch（即每个 FIELD_CENTER_INDEX）单独使用一个PSF。

import sys
import types
from pathlib import Path
import os

# photsim6 兼容导入层：默认使用 Photsim7 的 photsim6 compatibility shim。
PHOTSIM7_ROOT_CANDIDATES = [
    os.environ.get("ET_PHOTSIM7_ROOT"),
    "/home/liyang/ET7/Photsim7",
]
PHOTSIM7_ROOT = next(
    (path for path in PHOTSIM7_ROOT_CANDIDATES if path and os.path.isdir(path)),
    None,
)
if PHOTSIM7_ROOT is None:
    raise FileNotFoundError(
        "未找到 Photsim7 源码目录；请设置 ET_PHOTSIM7_ROOT。"
    )
if PHOTSIM7_ROOT not in sys.path:
    sys.path.insert(0, PHOTSIM7_ROOT)
for m in list(sys.modules):
    if m == "photsim6" or m.startswith("photsim6."):
        sys.modules.pop(m)
print(f"[Photsim core] using Photsim7 via photsim6 compatibility shim: {PHOTSIM7_ROOT}")

PHOTSIM7_DATA_DIR_CANDIDATES = [
    os.environ.get("ET_PHOTSIM7_DATA_DIR"),
    "/home/liyang/ET7/Photsim7-data",
]
PHOTSIM7_DATA_DIR = next(
    (path for path in PHOTSIM7_DATA_DIR_CANDIDATES if path and os.path.isdir(path)),
    None,
)
if PHOTSIM7_DATA_DIR is None:
    raise FileNotFoundError(
        "未找到 Photsim7 数据目录；请设置 ET_PHOTSIM7_DATA_DIR。"
    )
os.environ["ET_DATA_DIR"] = PHOTSIM7_DATA_DIR
print(f"[Photsim core] ET_DATA_DIR={os.environ['ET_DATA_DIR']}")

# dash stub（防止未装 dash 崩溃）
try:
    import dash
except Exception:
    dash_stub = types.ModuleType("dash")
    def _dash_component(*args, **kwargs): return {"args": args, "kwargs": kwargs}
    class _DashNamespace:
        def __getattr__(self, name): return _dash_component
    class _DummyDashApp:
        def __init__(self, *args, **kwargs): pass
        def callback(self, *args, **kwargs): return lambda f: f
        def run(self, *args, **kwargs): return None
    dash_stub.Dash = _DummyDashApp
    dash_stub.html = _DashNamespace()
    dash_stub.dcc = _DashNamespace()
    dash_stub.dash_table = _DashNamespace()
    sys.modules["dash"] = dash_stub
    dash_dependencies = types.ModuleType("dash.dependencies")
    dash_dependencies.Input = _dash_component
    dash_dependencies.Output = _dash_component
    sys.modules["dash.dependencies"] = dash_dependencies
    dash_exceptions = types.ModuleType("dash.exceptions")
    class _PreventUpdate(Exception): pass
    dash_exceptions.PreventUpdate = _PreventUpdate
    sys.modules["dash.exceptions"] = dash_exceptions

### 基于Photosim6ft/et_sim_guide_v2_jitter_filtstar_motion_addbright.py修改了选星规则：
'''
步骤

查询星表时直接获取 G<14 的所有星。
新增两个参数：N_FAINT（默认150）、N_BRIGHT（默认160），用于控制两组topK数量。
用 Gaia G Mag 字段分组：
暗星：11≤G<14，按星等升序排序，取前N_FAINT颗。
亮星：G<11，按星等升序排序，取前N_BRIGHT颗。
'''

import subprocess

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import json
import random

import torch


# -----------------------------------------------------------------------------
# Minimal replacements for symbols previously provided by photsim6 common imports

opj = os.path.join

try:
    from tqdm.auto import trange  # Notebook/terminal friendly
except Exception:  # pragma: no cover
    trange = range


# --------------------------------------------------------------
# Import commonly used packages
# from photsim6.config import common_import_script
# %run $common_import_script

# Import managers
from photsim6.simulator import Simulator
from photsim6.configurator import ConfigurationManager
from photsim6.time import resolve_detector_frame_timing
from photsim6.variants import VariantManager

# Import additional utilities
from photsim6.plot import (
    plot_jitter_integrated_psf_models,
    plot_single_frame_tpf,
    plot_gif_tpfs,
    plot_base_psf_models,
    det_motion_and_component_plot,
    meas_jit_drift_rms,
    plot_psd,
)
from photsim6.data_generators import (
    real_freqs,
    real_mags,
    psd_to_motion,
    gen_psd_motion,
    fft_bins_to_name_str,
    fft_bins_to_fullname_str,
    gen_power_spec,
    load_power_spectrum_config,
    create_power_spec_components,
    generate_jitter,
)
from photsim6.config import BASE_DATA_DIR

import pickle


ET_SINGLE_SCOPE_ZEROPOINT_E_PER_S = 615.75 * 1_961_225


def surface_brightness_mag_per_arcsec2_to_flux_per_pix(
    mag_per_arcsec2: float, pixel_scale_arcsec_per_pix: float
) -> float:
    """Convert ET surface brightness to per-pixel electron rate.

    The conversion uses the same ET zero point already hard-coded in
    `photsim6ft.field.Stars.build_catalog()` to keep stars and diffuse
    background on a consistent flux scale.
    """

    pixel_area_arcsec2 = pixel_scale_arcsec_per_pix**2
    return float(
        ET_SINGLE_SCOPE_ZEROPOINT_E_PER_S
        * 10 ** (-0.4 * mag_per_arcsec2)
        * pixel_area_arcsec2
    )


# ============================================================================
# 导星探测器参数（由用户提供）
# ============================================================================
GUIDE_PIXEL_SCALE_ARCSEC_PER_PIX = 3.146
GUIDE_PIXEL_WIDTH_UM = 6.5
GUIDE_FOV_DEG = 1.79   ## 1.7
GUIDE_EXPOSURE_S = 0.25
GUIDE_READOUT_S = 0.0
GUIDE_OBSERVING_DURATION_S = 15  #***
GUIDE_READOUT_NOISE_ADU_PER_PIX = 5.0
GUIDE_GAIN_ELECTRONS_PER_ADU = 1/2.2
GUIDE_READOUT_NOISE_E_PER_PIX = (
    GUIDE_READOUT_NOISE_ADU_PER_PIX * GUIDE_GAIN_ELECTRONS_PER_ADU
)
GUIDE_FULL_WELL_ELECTRONS = 1800.0
GUIDE_ADC_BIT_DEPTH = 12
GUIDE_BIAS_LEVEL_ADU = 135.0
GUIDE_COLUMN_NOISE_SIGMA_ADU = 0.0
GUIDE_ENABLE_COSMIC_RAYS = False
GUIDE_SUBTRACT_NONSTELLAR_MEAN = False
# GUIDE_DARK_PLUS_SCATTERED_E_PER_S_PER_PIX = 10.0
# 用户只给了暗电流+杂散光总和；两者都开启时，任意拆分都等价于同一个总泊松过程。
GUIDE_DARK_CURRENT_E_PER_S_PER_PIX = 0.7
#GUIDE_DARK_PLUS_SCATTERED_E_PER_S_PER_PIX / 2.0  #0.7  
GUIDE_SCATTERED_LIGHT_E_PER_S_PER_PIX = 2.11 #GUIDE_DARK_PLUS_SCATTERED_E_PER_S_PER_PIX / 2.0  #2.11  
GUIDE_INTER_PRV_RMS_PERCENT = 3.0
# GUIDE_BACKGROUND_SURFACE_BRIGHTNESS_MAG_PER_ARCSEC2 = 22.0
# GUIDE_BACKGROUND_FLUX_E_PER_S_PER_PIX = (
#     surface_brightness_mag_per_arcsec2_to_flux_per_pix(
#         GUIDE_BACKGROUND_SURFACE_BRIGHTNESS_MAG_PER_ARCSEC2,
#         GUIDE_PIXEL_SCALE_ARCSEC_PER_PIX,
#     )
# )
GUIDE_PSF_FIELD_ANGLE_DEG = 14.0
GUIDE_PSF_FIELD_ID = 7
GUIDE_OPTICAL_EFFICIENCY_RATIO_DEFAULT = 0.5
GUIDE_ET_QUANTUM_EFFICIENCY_DEFAULT = 0.7
# 默认 PSF bundle；不设环境变量时直接使用这里。
# 示例："Photosim6/data/psf/et/241006/D280mm-focus"
# PSF_BUNDLE_NAME_DEFAULT = "241006/D280mm-focus"
PSF_BUNDLE_NAME_DEFAULT = "241006/D280mm-focus-guide6p5um" # 凌星导星
FIELD_SPECIFIC_PSF_BUNDLE_NAMES = {
    0: "241006/D280mm-focus-guide6p5um",
    1: "241006_left/D280mm_focus-guide6p5um",
    2: "241006_top/D280mm_focus-guide6p5um",
    3: "241006_bottom/D280mm_focus-guide6p5um",
}


# ============================================================================
# 用户可改的“导星探测器仿真”关键设置（本脚本层实现，不改 photsim6ft 包）
# ============================================================================
# 1) 单个导星探测器视场。
FOV_DEG = GUIDE_FOV_DEG

# 1.5) 4 个导星探测器的天区中心接口。
FIELD_CENTERS_DEG = [
    (278.18437, 37.57037),
    (269.58206, 57.89966),
    (310.62391, 59.26764),
    (305.03563, 38.47553),
]

# ===== 批量 batch 机制（兼容 v2_jitter_filtstar_motion.py） =====
RUN_ALL_BATCHES = True
BATCH_INDICES_TO_RUN = None  # None 表示全部 batch
FIELD_CENTER_INDEX = 0
_FIELD_CENTER_INDEX_ENV = os.environ.get("ET_FIELD_CENTER_INDEX")
if _FIELD_CENTER_INDEX_ENV is not None:
    FIELD_CENTER_INDEX = int(_FIELD_CENTER_INDEX_ENV)

IS_CHILD_BATCH_RUN = os.environ.get("ET_CHILD_BATCH_RUN") == "1"
_RUN_ALL_BATCHES_ENV = os.environ.get("ET_RUN_ALL_BATCHES")
if _RUN_ALL_BATCHES_ENV is not None:
    RUN_ALL_BATCHES = _RUN_ALL_BATCHES_ENV.strip().lower() in {
        "1", "true", "yes", "y", "on"
    }

def run_all_batches_sequentially():
    batch_indices = (
        list(range(len(FIELD_CENTERS_DEG)))
        if BATCH_INDICES_TO_RUN is None
        else [int(idx) for idx in BATCH_INDICES_TO_RUN]
    )
    invalid = [idx for idx in batch_indices if not (0 <= idx < len(FIELD_CENTERS_DEG))]
    if invalid:
        raise ValueError(
            f"BATCH_INDICES_TO_RUN 中存在越界索引 {invalid}；"
            f"合法范围应为 [0, {len(FIELD_CENTERS_DEG) - 1}]。"
        )
    script_path = str(Path(__file__).resolve())
    for batch_index in batch_indices:
        batch_ra_deg, batch_dec_deg = FIELD_CENTERS_DEG[batch_index]
        print(
            f"[Batch launcher] start batch={batch_index}  "
            f"RA={batch_ra_deg:.6f} deg  DEC={batch_dec_deg:.6f} deg"
        )
        env = os.environ.copy()
        env["ET_CHILD_BATCH_RUN"] = "1"
        env["ET_FIELD_CENTER_INDEX"] = str(batch_index)
        subprocess.run([sys.executable, script_path], env=env, check=True)

if RUN_ALL_BATCHES and not IS_CHILD_BATCH_RUN:
    run_all_batches_sequentially()
    sys.exit(0)

if not (0 <= FIELD_CENTER_INDEX < len(FIELD_CENTERS_DEG)):
    raise ValueError(
        f"FIELD_CENTER_INDEX={FIELD_CENTER_INDEX} 越界；"
        f"请确保它在 [0, {len(FIELD_CENTERS_DEG) - 1}] 内。"
    )

FIELD_CENTER_RA_DEG, FIELD_CENTER_DEC_DEG = FIELD_CENTERS_DEG[FIELD_CENTER_INDEX]
print(
    f"[Sky center] batch={FIELD_CENTER_INDEX}  RA={FIELD_CENTER_RA_DEG:.6f} deg  DEC={FIELD_CENTER_DEC_DEG:.6f} deg"
)

# 1.6) 可选：热弹效应，四个导星探测器的周期性径向外扩位移。
# - 每 10 秒为一个周期；
# - 周期内最后 3 秒，对当前 batch 对应的导星探测器施加一次固定的“向外”位移；
# - 其余时间恢复为 0。
# 位移量以角秒给出，脚本内部会自动转换为像素。
#
# 环境变量用法示例：
#   ET_ENABLE_PERIODIC_RADIAL_EXPANSION=1 \
#   ET_RADIAL_EXPANSION_PERIOD_S=10 \
#   ET_RADIAL_EXPANSION_ACTIVE_LAST_S=3 \
#   ET_RADIAL_EXPANSION_ARCSEC=0.0054 \
#   python3 et_sim_guide_v2_jitter_filtstar_motion.py
#
# 变量含义：
# - ET_ENABLE_PERIODIC_RADIAL_EXPANSION：总开关，1 开 0 关；
# - ET_RADIAL_EXPANSION_PERIOD_S：一个完整周期的长度；
# - ET_RADIAL_EXPANSION_ACTIVE_LAST_S：每个周期末尾“激活”的秒数；
# - ET_RADIAL_EXPANSION_ARCSEC：相对位移幅度，单位 arcsec。
ENABLE_PERIODIC_RADIAL_EXPANSION = False
RADIAL_EXPANSION_PERIOD_S = 10.0
RADIAL_EXPANSION_ACTIVE_LAST_S = 3.0
RADIAL_EXPANSION_ARCSEC = 0.0054  # 幅度（0.0054 arcsec ≈ 0.0017 pix）

# 整幅视场中心定义：
# - 设为 None: 自动使用 4 个 FIELD_CENTERS_DEG 的几何均值作为“整幅视场中心”；
# - 设为数值: 显式指定整幅视场中心 (RA, DEC) [deg]。
RADIAL_EXPANSION_FOV_CENTER_RA_DEG = None
RADIAL_EXPANSION_FOV_CENTER_DEC_DEG = None

# 可选：手动覆盖 4 片探测器外扩方向（按 FIELD_CENTERS_DEG 顺序）。
# - 每项为 (dx, dy) 或 None。
# - None: 使用“相对整幅视场中心”的自动方向。
# - (dx, dy): 使用你指定的方向和符号（会自动归一化，幅值由 RADIAL_EXPANSION_ARCSEC 控制）。
# 例如把第 0 片设为朝 +x 方向外扩：MANUAL_DETECTOR_DIRECTION_DXDY[0] = (1.0, 0.0)
MANUAL_DETECTOR_DIRECTION_DXDY = [
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (0.0, 1.0),
]


def compute_field_center_radial_unit_vectors(
    field_centers_deg,
    fov_center_ra_deg=None,
    fov_center_dec_deg=None,
):
    """Compute outward unit vectors for each guide detector center in tangent-plane coords."""

    centers = np.asarray(field_centers_deg, dtype=float)
    if fov_center_ra_deg is None:
        center_ra_deg = float(np.mean(centers[:, 0]))
    else:
        center_ra_deg = float(fov_center_ra_deg)

    if fov_center_dec_deg is None:
        center_dec_deg = float(np.mean(centers[:, 1]))
    else:
        center_dec_deg = float(fov_center_dec_deg)

    cos_dec = float(np.cos(np.deg2rad(center_dec_deg)))

    radial_unit_vectors = []
    for ra_deg, dec_deg in centers:
        dx_deg = (float(ra_deg) - center_ra_deg) * cos_dec
        dy_deg = float(dec_deg) - center_dec_deg
        radius = float(np.hypot(dx_deg, dy_deg))
        if radius == 0.0:
            radial_unit_vectors.append((0.0, 0.0))
        else:
            radial_unit_vectors.append((dx_deg / radius, dy_deg / radius))

    return radial_unit_vectors


def _normalize_direction_vector(dx, dy):
    norm = float(np.hypot(float(dx), float(dy)))
    if norm == 0.0:
        raise ValueError("手动方向向量 (dx, dy) 不能是 (0, 0)。")
    return float(dx) / norm, float(dy) / norm


def resolve_detector_expansion_unit_vector(
    detector_index,
    auto_unit_vectors,
    manual_directions,
):
    if len(manual_directions) != len(auto_unit_vectors):
        raise ValueError(
            "MANUAL_DETECTOR_DIRECTION_DXDY 的长度必须与 FIELD_CENTERS_DEG 一致。"
        )

    manual_item = manual_directions[detector_index]
    if manual_item is None:
        ux, uy = auto_unit_vectors[detector_index]
        return float(ux), float(uy), "auto_fov_center"

    if len(manual_item) != 2:
        raise ValueError(
            "MANUAL_DETECTOR_DIRECTION_DXDY 的每一项必须是 None 或 (dx, dy)。"
        )

    ux, uy = _normalize_direction_vector(manual_item[0], manual_item[1])
    return ux, uy, "manual_dxdy"


FIELD_CENTER_RADIAL_UNIT_VECTORS = compute_field_center_radial_unit_vectors(
    FIELD_CENTERS_DEG,
    fov_center_ra_deg=RADIAL_EXPANSION_FOV_CENTER_RA_DEG,
    fov_center_dec_deg=RADIAL_EXPANSION_FOV_CENTER_DEC_DEG,
)
FIELD_CENTER_RADIAL_UNIT_X_DEG, FIELD_CENTER_RADIAL_UNIT_Y_DEG, _RADIAL_UNIT_SOURCE = (
    resolve_detector_expansion_unit_vector(
        detector_index=FIELD_CENTER_INDEX,
        auto_unit_vectors=FIELD_CENTER_RADIAL_UNIT_VECTORS,
        manual_directions=MANUAL_DETECTOR_DIRECTION_DXDY,
    )
)
print(
    "[Periodic radial expansion] "
    f"source={_RADIAL_UNIT_SOURCE} "
    f"batch={FIELD_CENTER_INDEX} radial_unit=("
    f"{FIELD_CENTER_RADIAL_UNIT_X_DEG:+.6f}, {FIELD_CENTER_RADIAL_UNIT_Y_DEG:+.6f})"
)


def build_periodic_radial_expansion_pix(
    time_s,
    pixel_scale_arcsec_per_pix,
    period_s,
    active_last_s,
    radial_arcsec,
    radial_unit_x,
    radial_unit_y,
):
    """Return per-frame x/y pixel offsets for the requested periodic radial expansion."""

    time_s = np.asarray(time_s, dtype=float)
    x_offset_pix = np.zeros_like(time_s, dtype=float)
    y_offset_pix = np.zeros_like(time_s, dtype=float)

    if period_s <= 0.0 or active_last_s <= 0.0 or radial_arcsec == 0.0:
        return x_offset_pix, y_offset_pix

    active_last_s = min(float(active_last_s), float(period_s))
    phase_s = np.mod(time_s, float(period_s))
    active_mask = phase_s >= (float(period_s) - active_last_s)
    offset_pix = float(radial_arcsec) / float(pixel_scale_arcsec_per_pix)

    x_offset_pix[active_mask] = offset_pix * float(radial_unit_x)
    y_offset_pix[active_mask] = offset_pix * float(radial_unit_y)
    return x_offset_pix, y_offset_pix

# 2) drift-jitter 分界：以“单个 raw frame 的有效积分时间”作为时间尺度。
#    - CCD：积分窗口 = Exposure Duration * Simulation Cadence Mult
#    - CMOS：积分窗口 = (Exposure + Readout) * Simulation Cadence Mult
#      因为当前模型假设 CMOS 在读出期间仍持续积光。
#    解释：把短于该积分窗口的运动平均进 PSF（jitter）；更慢的变化保留为帧间质心漂移（drift）。
#    因此主探测器 10s 与导星探测器 250ms 会自动得到不同 split_hz。
USE_CADENCE_SPLIT = True

# 2.1) 可选：手动覆盖 split_hz（一般不需要）
# - None：自动 split_hz = 1 / raw_frame_integration_s（推荐）
# - 数值：强制使用该 split_hz
SPLIT_HZ_OVERRIDE = None

# 3) 导星星表查询深度：
# - 先从 Gaia G=GAIA_GMAG_LIM 开始查；
# - 若该视场内星数不足目标值，则逐步放宽到更暗的星；
# - 只把最亮的若干颗送进仿真，避免逐星 PSF 叠加过慢。
GAIA_GMAG_LIM = 12.5
GAIA_GMAG_LIM_STEP = 0.5
GAIA_GMAG_LIM_MAX = 14
GUIDE_STAR_QUERY_BACKEND = "et_focalplane"
GUIDE_GAIA_CATALOG_DIR = "/home/liyang/ETguider/gaia_dr3_19mag"
ET_FOCALPLANE_ROOT = "/home/liyang/ET7/et_focalplane"
ET_FOCALPLANE_DATA_DIR = str(Path(ET_FOCALPLANE_ROOT) / "data")
ET_FOCALPLANE_SRC_DIR = str(Path(ET_FOCALPLANE_ROOT) / "src")
GUIDE_QUERY_TARGET_EPOCH = 2000.0
GUIDE_QUERY_CROP_TO_SIM_FRAME = True
GUIDE_QUERY_CROP_MARGIN_PIX = 0.0

# 3.1) 批次级 field polar angle（单位：度）。
# DVA / thermal drift 在当前裁剪 ET workflow 中按“探测器级共享方向”近似处理，
# 这里显式固定为导星探测器对应的 14 deg PSF 方向，并在图像渲染与 truth 导出中复用。
FIELD_POLAR_ANGLE_DEG = GUIDE_PSF_FIELD_ANGLE_DEG
FIELD_POLAR_ANGLE_RAD = float(np.deg2rad(FIELD_POLAR_ANGLE_DEG))

# 3.2) 星数上限（重要的性能开关）
# photsim6ft 的 `generate_background_starlight_frames` 会对每颗星逐个生成 PSF 并累加，
# 星数一旦上千，每帧耗时会非常夸张，GPU 上还可能触发 watchdog timeout。
# 为避免大视场下逐星 PSF 叠加过慢，这里默认只保留最亮的若干颗星（包含 target 星）。
MAX_SIM_STARS = None #200 #***

# 3.2.1) 导星星表选择规则：
# - 亮星区间(G < GAIA_GMAG_LIM_MIN)，取 top N_bright；
# - 暗星区间(G >= GAIA_GMAG_LIM_MIN)，取 top N_faint；
# - 合并两组，target星(索引0)始终保留。
# - N_bright, N_faint 可参数化。
GUIDE_TOP_BRIGHT = 350  # G < GAIA_GMAG_LIM_MIN 区间取多少颗
GUIDE_TOP_FAINT = 0   # G >= GAIA_GMAG_LIM_MIN 区间取多少颗
GUIDE_MIN_CATALOG_STARS = GUIDE_TOP_BRIGHT + GUIDE_TOP_FAINT  # 用于查询深度

# 3.2.2) 导星“亮端”星等限制（Kepler Mag）。
# 设为 10.0 表示仅保留 Gaia G Mag >= 10 的星（不选更亮的星）。
# 设为 None 可关闭该限制。
# 也可用环境变量 ET_GAIA_GMAG_LIM_MIN 覆盖（例如 10 或 None）。
GAIA_GMAG_LIM_MIN = 99 #10.5  # None  #10.0

# 3.3) 是否对整幅星场施加静态亚像元偏移。
# - False：默认关闭，便于把星表真值与像面真值严格对齐，适合导星精度评估。
# - True：为整幅星场加入同一个 (dx, dy) 常量偏移，模拟天空相对像元网格的静态相位。
#   若不手动指定 offset，脚本会在 [-0.5, 0.5] pix 内随机采样，并写入 run_meta.json。
APPLY_STATIC_FIELD_OFFSET = False
STATIC_FIELD_OFFSET_X_PIX = None
STATIC_FIELD_OFFSET_Y_PIX = None

# 3.5) 计算设备：你希望使用 GPU。
# - 设为 "cuda" 会尝试使用 GPU；若当前 PyTorch/驱动不可用，会自动回退到 CPU。
# - 也可手动设为 "cpu" 强制走 CPU。
REQUESTED_COMPUTE_DEVICE = "cuda"

# 4) 产物输出根目录（你可指定到任意磁盘/目录）：
#    你希望“所有产品”都落在一个确定位置。这里的 OUTPUT_ROOT 就是那个最上层目录。
#    脚本会在其下创建更细的子目录结构（你说子目录怎么组织无所谓）。

OUTPUT_ROOT = "/home/liyang/ET7/Results/Transit/Simulation/"


# 本次运行的名字（目录名的一部分），建议你按项目/日期/参数命名，便于检索。
OUTPUT_RUN_NAME = "guide_full_10p5mag" #"guide_det_v1_noise_psf_6s"

# 是否启用流式落盘：每帧（或极小批次）生成后立刻写入 .npz，避免内存累计。
STREAM_SAVE_FRAMES = True


# 每次生成并保存多少帧。
# - 1：最安全（逐帧保存），几乎不可能因“堆帧”而爆内存/显存。
# - >1：更快，但会短暂占用更多内存（尤其是 subpixel 阶段）。
FRAME_BATCH_SIZE = 1

# 5) 随机序列一致性：跨不同 FIELD_CENTER_INDEX 仍保持同一套随机序列
#    目标：保证 drift/jitter、MD 随机游走、以及所有依赖 RNG 的 time series 在不同天区中心下完全一致。
GLOBAL_SEED = 12345

# 建议保持 True：每个 batch 在生成 dynamic params 前重置种子，避免“星场查询/随机偏移”消耗 RNG 影响 motion。
RESEED_BEFORE_DYNAMIC_PARAMS = True


def set_global_seed(seed: int) -> None:
    """设置所有常见 RNG 的随机种子，保证跨批次可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"环境变量 {name}={value!r} 不是合法布尔值。")


def env_int(name: str, default=None):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


_GLOBAL_SEED_ENV = os.environ.get("GLOBAL_SEED")
GLOBAL_SEED = env_int("GLOBAL_SEED", GLOBAL_SEED)
if _GLOBAL_SEED_ENV not in (None, ""):
    print(f"[Seed] Using GLOBAL_SEED={GLOBAL_SEED} (from env)")




def env_float_or_none(name: str, default=None):
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    if value == "":
        return default
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def efficiency_to_percent_quantity(value: float, name: str):
    factor = float(value)
    if factor < 0.0:
        raise ValueError(f"{name} 不能为负数：{value}")
    if factor <= 1.0:
        factor *= 100.0
    return factor * u.percent


def resolve_field_offsets(
    apply_offset: bool,
    offset_x_pix,
    offset_y_pix,
) -> tuple[float, float]:
    """Resolve a single static field offset and reuse it across adaptive Gaia retries."""

    if apply_offset:
        resolved_x = (
            float(offset_x_pix)
            if offset_x_pix is not None
            else float(np.random.uniform(-0.5, 0.5))
        )
        resolved_y = (
            float(offset_y_pix)
            if offset_y_pix is not None
            else float(np.random.uniform(-0.5, 0.5))
        )
        return resolved_x, resolved_y

    return (
        0.0 if offset_x_pix is None else float(offset_x_pix),
        0.0 if offset_y_pix is None else float(offset_y_pix),
    )


def _slice_star_data_by_indices(star_data: dict, keep_indices) -> dict:
    """Return a copy of star_data keeping only per-star entries at keep_indices."""

    keep = np.asarray(keep_indices, dtype=int)
    n_stars = int(len(np.asarray(star_data["x0"])))
    sliced = {}

    for key, value in star_data.items():
        if isinstance(value, str):
            sliced[key] = value
            continue

        if np.isscalar(value):
            sliced[key] = value
            continue

        try:
            arr = np.asarray(value)
        except Exception:
            sliced[key] = value
            continue

        if arr.ndim >= 1 and arr.shape[0] == n_stars:
            sliced[key] = arr[keep].copy()
        else:
            sliced[key] = value

    return sliced


def enrich_star_data_truth_columns(star_data: dict) -> dict:
    """Add detector-absolute truth columns when the et_focalplane backend provides them."""

    enriched = dict(star_data)
    field_offset_x = float(enriched.get("field_offset_x_pix", 0.0))
    field_offset_y = float(enriched.get("field_offset_y_pix", 0.0))

    if "detector_xpix" in enriched:
        enriched["detector_xpix_shifted"] = (
            np.asarray(enriched["detector_xpix"], dtype=float) + field_offset_x
        )
    if "detector_ypix" in enriched:
        enriched["detector_ypix_shifted"] = (
            np.asarray(enriched["detector_ypix"], dtype=float) + field_offset_y
        )
    if "target_center_xpix" in enriched:
        enriched["target_center_xpix_shifted"] = (
            float(enriched["target_center_xpix"]) + field_offset_x
        )
    if "target_center_ypix" in enriched:
        enriched["target_center_ypix_shifted"] = (
            float(enriched["target_center_ypix"]) + field_offset_y
        )

    return enriched


# def select_guide_star_indices(
#     star_df,
#     min_catalog_stars: int,
#     max_sim_stars: int | None,
#     min_gaia_gmag: float | None = None,
# ):
#     """Build dual selections for simulation and guiding.

#     Rules:
#     1) Guide candidates are selected only from faint stars (Gaia G >= min_gaia_gmag).
#     2) Simulation uses all bright stars (Gaia G < min_gaia_gmag) plus selected guide candidates.
#     3) Keep target star (index 0) as the first simulation star whenever present.
#     """

#     if "Gaia G Mag" not in star_df.colnames:
#         raise ValueError("星表缺少 'Gaia G Mag' 列，无法按 Gaia G 星等做筛选。")

#     mags = np.asarray(star_df["Gaia G Mag"], dtype=float)
#     if mags.size == 0:
#         raise ValueError("空星表无法用于导星仿真。")

#     all_idx = np.arange(mags.size, dtype=int)
#     candidate_idx = all_idx
#     bright_idx = np.asarray([], dtype=int)
#     if min_gaia_gmag is not None:
#         min_g = float(min_gaia_gmag)
#         candidate_idx = all_idx[mags[all_idx] >= min_g]
#         bright_idx = all_idx[mags[all_idx] < min_g]

#     if candidate_idx.size == 0:
#         raise ValueError(
#             f"按 Gaia G Mag >= {float(min_gaia_gmag):.3f} 过滤后无可用星。"
#         )

#     target_included = bool(np.any(candidate_idx == 0))
#     keep = [0] if target_included else []
#     for idx in candidate_idx[np.argsort(mags[candidate_idx])].tolist():
#         if target_included and idx == 0:
#             continue
#         keep.append(idx)

#     n_available = int(len(keep))
#     n_filtered_bright = int(mags.size - candidate_idx.size)

#     if max_sim_stars is not None and max_sim_stars > 0 and len(keep) > max_sim_stars:
#         keep = keep[: int(max_sim_stars)]

#     guide_keep_arr = np.asarray(keep, dtype=int)

#     sim_keep = []
#     if mags.size > 0:
#         sim_keep.append(0)

#     if bright_idx.size > 0:
#         for idx in bright_idx[np.argsort(mags[bright_idx])].tolist():
#             if idx == 0:
#                 continue
#             sim_keep.append(int(idx))

#     for idx in guide_keep_arr.tolist():
#         if idx in sim_keep:
#             continue
#         sim_keep.append(int(idx))

#     sim_keep_arr = np.asarray(sim_keep, dtype=int)
#     guide_mask_in_sim = np.isin(sim_keep_arr, guide_keep_arr)

#     return {
#         "keep_indices": sim_keep_arr,
#         "guide_keep_indices": guide_keep_arr,
#         "guide_mask_in_sim": guide_mask_in_sim,
#         "mags_all": mags,
#         "n_available": int(n_available),
#         "n_selected": int(len(guide_keep_arr)),
#         "n_sim_selected": int(len(sim_keep_arr)),
#         "n_guide_selected": int(len(guide_keep_arr)),
#         "n_bright_added_to_sim": int(np.sum(np.isin(sim_keep_arr, bright_idx))),
#         "n_filtered_bright": int(n_filtered_bright),
#         "min_gaia_gmag": (
#             None if min_gaia_gmag is None else float(min_gaia_gmag)
#         ),
#         "target_included": bool(target_included),
#         "selected_brightest_mag": float(np.min(mags[guide_keep_arr])),
#         "selected_faintest_mag": float(np.max(mags[guide_keep_arr])),
#         "sim_brightest_mag": float(np.min(mags[sim_keep_arr])),
#         "sim_faintest_mag": float(np.max(mags[sim_keep_arr])),
#         "meets_min_count": bool(n_available >= int(min_catalog_stars)),
#     }


# def query_guide_star_field_adaptive(
#     *,
#     ra,
#     dec,
#     px_cnt,
#     px_scale,
#     apply_offset: bool,
#     plot: bool,
#     offset_x_pix,
#     offset_y_pix,
#     base_gaia_gmag_lim: float,
#     min_catalog_stars: int,
#     gaia_gmag_step: float,
#     gaia_gmag_lim_max: float,
#     query_backend: str,
#     catalog_dir: str | None,
#     et_focalplane_data_dir: str | None,
#     et_focalplane_src_dir: str | None,
#     target_epoch: float,
#     crop_to_simulation_frame: bool,
#     crop_margin_pix: float,
# ):
#     """Query Gaia repeatedly, relaxing the magnitude limit until enough stars are found."""

#     resolved_offset_x_pix, resolved_offset_y_pix = resolve_field_offsets(
#         apply_offset=apply_offset,
#         offset_x_pix=offset_x_pix,
#         offset_y_pix=offset_y_pix,
#     )

#     tried_limits = []
#     gaia_gmag_lim = float(base_gaia_gmag_lim)
#     star_data = None
#     backend = query_backend.strip().lower()

#     while True:
#         if backend == "et_focalplane":
#             star_data = mk_real_field_stars_et_focalplane(
#                 target_ra=ra,
#                 target_dec=dec,
#                 catalog_dir=catalog_dir or GUIDE_GAIA_CATALOG_DIR,
#                 registry_data_dir=et_focalplane_data_dir or ET_FOCALPLANE_DATA_DIR,
#                 px_cnt=px_cnt,
#                 apply_offset=apply_offset,
#                 mag_lim=gaia_gmag_lim,
#                 plot=False,
#                 offset_x_pix=resolved_offset_x_pix,
#                 offset_y_pix=resolved_offset_y_pix,
#                 target_epoch=target_epoch,
#                 crop_to_simulation_frame=crop_to_simulation_frame,
#                 crop_margin_pix=crop_margin_pix,
#                 et_focalplane_src=et_focalplane_src_dir or ET_FOCALPLANE_SRC_DIR,
#             )
#         elif backend == "legacy_box_gaia":
#             star_data = mk_real_field_stars_2(
#                 ra=ra,
#                 dec=dec,
#                 px_cnt=px_cnt,
#                 px_scale=px_scale,
#                 apply_offset=apply_offset,
#                 mag_lim=gaia_gmag_lim,
#                 plot=False,
#                 offset_x_pix=resolved_offset_x_pix,
#                 offset_y_pix=resolved_offset_y_pix,
#             )
#         else:
#             raise ValueError(
#                 f"Unsupported GUIDE_STAR_QUERY_BACKEND={query_backend!r}. "
#                 "Use 'et_focalplane' or 'legacy_box_gaia'."
#             )

#         n_found = int(len(np.asarray(star_data["x0"])))
#         tried_limits.append(
#             {
#                 "gaia_gmag_lim": float(gaia_gmag_lim),
#                 "n_stars_found": n_found,
#             }
#         )

#         if n_found >= int(min_catalog_stars) or gaia_gmag_lim >= float(
#             gaia_gmag_lim_max
#         ):
#             break

#         gaia_gmag_lim = min(
#             float(gaia_gmag_lim_max),
#             float(gaia_gmag_lim + gaia_gmag_step),
#         )

#     if plot:
#         if backend == "et_focalplane":
#             star_data = mk_real_field_stars_et_focalplane(
#                 target_ra=ra,
#                 target_dec=dec,
#                 catalog_dir=catalog_dir or GUIDE_GAIA_CATALOG_DIR,
#                 registry_data_dir=et_focalplane_data_dir or ET_FOCALPLANE_DATA_DIR,
#                 px_cnt=px_cnt,
#                 apply_offset=apply_offset,
#                 mag_lim=gaia_gmag_lim,
#                 plot=True,
#                 offset_x_pix=resolved_offset_x_pix,
#                 offset_y_pix=resolved_offset_y_pix,
#                 target_epoch=target_epoch,
#                 crop_to_simulation_frame=crop_to_simulation_frame,
#                 crop_margin_pix=crop_margin_pix,
#                 et_focalplane_src=et_focalplane_src_dir or ET_FOCALPLANE_SRC_DIR,
#             )
#         else:
#             star_data = mk_real_field_stars_2(
#                 ra=ra,
#                 dec=dec,
#                 px_cnt=px_cnt,
#                 px_scale=px_scale,
#                 apply_offset=apply_offset,
#                 mag_lim=gaia_gmag_lim,
#                 plot=True,
#                 offset_x_pix=resolved_offset_x_pix,
#                 offset_y_pix=resolved_offset_y_pix,
#             )

#     return star_data, {
#         "query_backend": backend,
#         "query_detector_id": str(star_data.get("detector_id", "")),
#         "query_target_epoch": float(target_epoch),
#         "query_base_gaia_gmag_lim": float(base_gaia_gmag_lim),
#         "query_final_gaia_gmag_lim": float(gaia_gmag_lim),
#         "query_gaia_gmag_step": float(gaia_gmag_step),
#         "query_gaia_gmag_lim_max": float(gaia_gmag_lim_max),
#         "query_crop_to_simulation_frame": bool(crop_to_simulation_frame),
#         "query_crop_margin_pix": float(crop_margin_pix),
#         "query_history": tried_limits,
#         "query_reached_min_star_target": bool(
#             len(np.asarray(star_data["x0"])) >= int(min_catalog_stars)
#         ),
#         "resolved_field_offset_x_pix": float(resolved_offset_x_pix),
#         "resolved_field_offset_y_pix": float(resolved_offset_y_pix),
#     }


def build_variant_settings(**overrides):
    settings = dict(
        enable_stellar_photon_noise=False,
        enable_background_light=True,
        enable_scattered_light=True,
        enable_dark_current=True,
        enable_readout_noise=True,
        enable_gain=True,
        enable_whole_pixel_gain_normal_distribution=False,
        enable_whole_pixel_gain_sinusoidal_modulation=False,
        enable_target_star=True,
        enable_background_stars=True,
        enable_jitter=False,
        enable_dva_drift=False,
        enable_pointing_drift=False,
        enable_psf_breathing=False,
        enable_inter_pixel_response=False,
        enable_intra_pixel_response=False,
        enable_pixel_phase_response=False,
    )
    settings.update(overrides)
    return settings


EFFECT_PROFILES = {
    "full": dict(
        name="full",
        description="full",
        variant_settings={},
        include_pointing_drift=True,
        include_dva=True,
        include_thermal=True,
        include_momentum_dump=True,
        uses_jitter_integrated_psf=True,
    ),
    "v1_noise_psf": dict(
        name="v1_noise_psf",
        description="v1_noise_psf",
        variant_settings=build_variant_settings(),
        include_pointing_drift=False,
        include_dva=False,
        include_thermal=False,
        include_momentum_dump=False,
        uses_jitter_integrated_psf=False,
    ),
    "v2_point_drift_jitter": dict(
        name="v2_point_drift_jitter",
        description="v2_point_drift_jitter",
        variant_settings=build_variant_settings(
            enable_jitter=True,
            enable_pointing_drift=True,
        ),
        include_pointing_drift=True,
        include_dva=False,
        include_thermal=False,
        include_momentum_dump=False,
        uses_jitter_integrated_psf=True,
    ),
    # "v2a_jitter": dict(
    #     name="v2a_jitter",
    #     description="v2a_jitter",
    #     variant_settings=build_variant_settings(
    #         enable_jitter=True,
    #         enable_pointing_drift=False,
    #     ),
    #     include_pointing_drift=False,
    #     include_dva=False,
    #     include_thermal=False,
    #     include_momentum_dump=False,
    #     uses_jitter_integrated_psf=True,
    # ),
    "v3_dva": dict(
        name="v3_dva",
        description="v3_dva",
        variant_settings=build_variant_settings(
            enable_dva_drift=True,
        ),
        include_pointing_drift=False,
        include_dva=True,
        include_thermal=False,
        include_momentum_dump=False,
        uses_jitter_integrated_psf=False,
    ),
    "v4_thermal": dict(
        name="v4_thermal",
        description="v4_thermal",
        variant_settings=build_variant_settings(
            enable_pointing_drift=True,
        ),
        include_pointing_drift=False,
        include_dva=False,
        include_thermal=True,
        include_momentum_dump=False,
        uses_jitter_integrated_psf=False,
    ),
    "v5_prv_subpixel": dict(
        name="v5_prv_subpixel",
        description="v5_prv_subpixel",
        variant_settings=build_variant_settings(
            enable_inter_pixel_response=True,
            enable_intra_pixel_response=True,
            enable_pixel_phase_response=True,
        ),
        include_pointing_drift=False,
        include_dva=False,
        include_thermal=False,
        include_momentum_dump=False,
        uses_jitter_integrated_psf=False,
    ),
    # 自定义：关闭暗电流、背景光、杂散光、读出噪声
    "v0_onlypsf": dict(
        name="v0_onlypsf",  
        description="v0_onlypsf", #no_dark_bg_scatter_readout
        variant_settings=build_variant_settings(
            enable_stellar_photon_noise=True,
            enable_background_light=False,
            enable_scattered_light=False,
            enable_dark_current=False,
            enable_readout_noise=False,
            enable_gain=False,
            enable_target_star=True,
            enable_background_stars=True,
        ),
        include_pointing_drift=False,
        include_dva=False,
        include_thermal=False,
        include_momentum_dump=False,
        uses_jitter_integrated_psf=False,
    ),
}


EFFECT_PROFILE_ALIASES = {
    "full": "full",
    "version0": "v0_onlypsf",
    "v0": "v0_onlypsf",
    "v0_onlypsf": "v0_onlypsf",
    "version1": "v1_noise_psf",
    "v1": "v1_noise_psf",
    "v1_noise_psf": "v1_noise_psf",
    "version2": "v2_point_drift_jitter",
    "v2": "v2_point_drift_jitter",
    "v2_point_drift_jitter": "v2_point_drift_jitter",
    # "v2a": "v2a_jitter",
    # "v2a_jitter": "v2a_jitter",
    "version3": "v3_dva",
    "v3": "v3_dva",
    "v3_dva": "v3_dva",
    "version4": "v4_thermal",
    "v4": "v4_thermal",
    "v4_thermal": "v4_thermal",
    "version5": "v5_prv_subpixel",
    "v5": "v5_prv_subpixel",
    "v5_prv_subpixel": "v5_prv_subpixel",
}


requested_effect_profile = env_str("ET_EFFECT_PROFILE", "v2_point_drift_jitter").lower()
effect_profile_key = EFFECT_PROFILE_ALIASES.get(
    requested_effect_profile, requested_effect_profile
)
if effect_profile_key not in EFFECT_PROFILES:
    valid_profiles = ", ".join(sorted(EFFECT_PROFILES))
    raise ValueError(
        f"未知 ET_EFFECT_PROFILE={requested_effect_profile!r}。"
        f" 可选值: {valid_profiles}"
    )

ACTIVE_EFFECT_PROFILE = EFFECT_PROFILES[effect_profile_key]
PROFILE_TARGET_FRAMES = env_int("ET_PROFILE_TARGET_FRAMES", None)
ENABLE_FRAME_TO_FRAME_DRIFT = env_flag(
    "ET_ENABLE_FRAME_TO_FRAME_DRIFT",
    False,
)
_ENABLE_FRAME_TO_FRAME_DRIFT_ENV = os.environ.get("ET_ENABLE_FRAME_TO_FRAME_DRIFT")

RUNTIME_EFFECT_COMPONENTS = {
    "pointing_drift": bool(ENABLE_FRAME_TO_FRAME_DRIFT),
    "jitter": bool(ACTIVE_EFFECT_PROFILE["variant_settings"].get("enable_jitter", True)),
    "dva": bool(ACTIVE_EFFECT_PROFILE["include_dva"]),
    "thermal": bool(ACTIVE_EFFECT_PROFILE["include_thermal"]),
    "momentum_dump": bool(ACTIVE_EFFECT_PROFILE["include_momentum_dump"]),
    "psf_breathing": bool(
        ACTIVE_EFFECT_PROFILE["variant_settings"].get("enable_psf_breathing", True)
    ),
    "inter_pixel_response": bool(
        ACTIVE_EFFECT_PROFILE["variant_settings"].get(
            "enable_inter_pixel_response", True
        )
    ),
    "intra_pixel_response": bool(
        ACTIVE_EFFECT_PROFILE["variant_settings"].get(
            "enable_intra_pixel_response", True
        )
    ),
    "pixel_phase_response": bool(
        ACTIVE_EFFECT_PROFILE["variant_settings"].get(
            "enable_pixel_phase_response", True
        )
    ),
}

OUTPUT_ROOT = env_str("ET_OUTPUT_ROOT_OVERRIDE", OUTPUT_ROOT)
OUTPUT_RUN_NAME = env_str("ET_OUTPUT_RUN_NAME_OVERRIDE", OUTPUT_RUN_NAME)
MAX_SIM_STARS = env_int("ET_MAX_SIM_STARS", MAX_SIM_STARS)
GAIA_GMAG_LIM_MIN = env_float_or_none(
    "ET_GAIA_GMAG_LIM_MIN", GAIA_GMAG_LIM_MIN
)
ENABLE_PERIODIC_RADIAL_EXPANSION = env_flag(
    "ET_ENABLE_PERIODIC_RADIAL_EXPANSION", ENABLE_PERIODIC_RADIAL_EXPANSION
)
RADIAL_EXPANSION_PERIOD_S = float(
    env_float_or_none("ET_RADIAL_EXPANSION_PERIOD_S", RADIAL_EXPANSION_PERIOD_S)
)
RADIAL_EXPANSION_ACTIVE_LAST_S = float(
    env_float_or_none(
        "ET_RADIAL_EXPANSION_ACTIVE_LAST_S", RADIAL_EXPANSION_ACTIVE_LAST_S
    )
)
RADIAL_EXPANSION_ARCSEC = float(
    env_float_or_none("ET_RADIAL_EXPANSION_ARCSEC", RADIAL_EXPANSION_ARCSEC)
)

print(
    "[Effect profile] "
    f"{ACTIVE_EFFECT_PROFILE['name']}  "
    f"(pointing={RUNTIME_EFFECT_COMPONENTS['pointing_drift']}, "
    f"jitter={RUNTIME_EFFECT_COMPONENTS['jitter']}, "
    f"dva={RUNTIME_EFFECT_COMPONENTS['dva']}, "
    f"thermal={RUNTIME_EFFECT_COMPONENTS['thermal']}, "
    f"prv={RUNTIME_EFFECT_COMPONENTS['inter_pixel_response'] or RUNTIME_EFFECT_COMPONENTS['intra_pixel_response'] or RUNTIME_EFFECT_COMPONENTS['pixel_phase_response']})"
)
if _ENABLE_FRAME_TO_FRAME_DRIFT_ENV is None:
    print("[Runtime default] ET_ENABLE_FRAME_TO_FRAME_DRIFT=False for final_4 jitter-only runs.")
else:
    print(
        "[Runtime override] "
        f"ET_ENABLE_FRAME_TO_FRAME_DRIFT={ENABLE_FRAME_TO_FRAME_DRIFT}"
    )


def get_output_dirs():
    """构造并创建本次运行的输出目录。

    返回：
    - run_dir：本次运行根目录
    - batch_dir：当前 FIELD_CENTER_INDEX 的目录
    - frames_dir：逐帧（或逐小批次）保存 .npz 的目录
    - cache_dir：缓存目录（例如 jitter 缓存）
    """
    run_dir = opj(OUTPUT_ROOT, OUTPUT_RUN_NAME)
    batch_dir = opj(
        run_dir,
        f"batch{FIELD_CENTER_INDEX}_ra{FIELD_CENTER_RA_DEG:.4f}_dec{FIELD_CENTER_DEC_DEG:.4f}",
    )
    frames_dir = opj(batch_dir, "frames")
    cache_dir = opj(run_dir, "cache")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    return run_dir, batch_dir, frames_dir, cache_dir


def pick_compute_device(requested: str) -> str:
    requested = (requested or "").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # 兜底：未知字符串时，按“有 GPU 用 GPU”处理
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---- global init: deterministic RNG + output dirs ----
set_global_seed(GLOBAL_SEED)
RUN_DIR, BATCH_DIR, FRAMES_DIR, CACHE_DIR = get_output_dirs()
COMPUTE_DEVICE = pick_compute_device(REQUESTED_COMPUTE_DEVICE)

# 先落一个最小 metadata，后面读取完 xlsx / 推导参数后会再补充写入。
meta_path = opj(BATCH_DIR, "run_meta.json")
if not os.path.isfile(meta_path):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "field_center_index": FIELD_CENTER_INDEX,
                "field_center_ra_deg": FIELD_CENTER_RA_DEG,
                "field_center_dec_deg": FIELD_CENTER_DEC_DEG,
                "global_seed": GLOBAL_SEED,
                "requested_compute_device": REQUESTED_COMPUTE_DEVICE,
                "compute_device": COMPUTE_DEVICE,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_pickle(pklf):
    with open(pklf, "rb") as f:
        dd = pickle.load(f)

    return dd


def prepare_photsim7_config_xlsx(config_xlsx_path: str, cache_dir: str):
    """Create a temporary config sheet accepted by Photsim7's unit validation."""

    import pandas as pd

    source_path = Path(config_xlsx_path)
    config_df = pd.read_excel(source_path)
    if len(config_df.columns) < 2:
        raise ValueError(f"配置文件列数不足，无法定位参数名列: {source_path}")

    parameter_col = config_df.columns[1]
    if "Value" not in config_df.columns or "Unit" not in config_df.columns:
        raise ValueError(
            f"配置文件缺少 Value/Unit 列，无法为 Photsim7 修正 Readout Noise 单位: {source_path}"
        )

    readout_mask = config_df[parameter_col].astype(str).str.strip().eq("Readout Noise")
    if not readout_mask.any():
        raise ValueError(f"配置文件中没有 Readout Noise 行: {source_path}")

    config_df.loc[readout_mask, "Value"] = float(GUIDE_READOUT_NOISE_E_PER_PIX)
    config_df.loc[readout_mask, "Unit"] = "electron/pix"

    psf_field_id_mask = config_df[parameter_col].astype(str).str.strip().eq("PSF Field ID")
    if psf_field_id_mask.any():
        config_df.loc[psf_field_id_mask, "Value"] = str(int(GUIDE_PSF_FIELD_ID))
        config_df.loc[psf_field_id_mask, "Unit"] = None

    # Keep simulator policy explicit: hidden random global offsets must stay disabled.
    forced_zero_items = {}
    for offset_name in ("Telescope FOV Max Offset", "Target Max Offset"):
        offset_mask = (
            config_df[parameter_col].astype(str).str.strip().eq(offset_name)
        )
        if not offset_mask.any():
            forced_zero_items[offset_name] = {
                "present": False,
                "forced_zero": False,
                "original_values": [],
            }
            print(f"[Config sanitize] {offset_name} not found in {source_path}")
            continue

        original_values = []
        for value in config_df.loc[offset_mask, "Value"].tolist():
            if pd.isna(value):
                original_values.append(None)
            else:
                original_values.append(str(value))

        config_df.loc[offset_mask, "Value"] = 0
        forced_zero_items[offset_name] = {
            "present": True,
            "forced_zero": True,
            "original_values": original_values,
        }
        if any(v not in (None, "0", "0.0") for v in original_values):
            print(
                f"[Config sanitize] {offset_name} {original_values} -> 0"
            )

    output_dir = Path(cache_dir) / "photsim7_config"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_path.stem}_photsim7_units.xlsx"
    config_df.to_excel(output_path, index=False)
    return str(output_path), forced_zero_items


plt.close("all")
# %matplotlib inline

"""
Working with this configuration:
------------------------------------------------------------------------------------------
|                         Last run date: 2025-06-05 02:05 PM EDT                         |
------------------------------------------------------------------------------------------
| Python | Jupyterlab | Numpy | Scipy  | Astropy | Matplotlib | Tensorflow |   Pytorch   |
------------------------------------------------------------------------------------------
| 3.12.9 |   4.3.6    | 2.2.4 | 1.15.2 |  7.0.1  |   3.10.1   |    N/A     | 2.6.0+cu126 |
------------------------------------------------------------------------------------------
"""

# config_xlsx_full_path = 'et_inputs_D2_PSF(241006)_Ji(0.24as)_Dr(0.03as_TESS_to_ET)_MD(limit_0.15as_Cyc_3d)_B(TESS_to_ET).xlsx'
config_xlsx_candidates = []

config_xlsx_override = os.environ.get("ET_CONFIG_XLSX_PATH")
if config_xlsx_override:
    config_xlsx_candidates.append(config_xlsx_override)

config_xlsx_candidates.extend(
    [
        "/home/liyang/ET7/ET-fsgsim/config/et_inputs_transit_guider.xlsx",
    ]
)

config_xlsx_full_path = None
for candidate in config_xlsx_candidates:
    if candidate and os.path.isfile(candidate):
        config_xlsx_full_path = candidate
        break

if config_xlsx_full_path is None:
    searched = ", ".join(config_xlsx_candidates)
    raise FileNotFoundError(
        f"No transit guider xlsx found. Checked: {searched}"
    )

config_manager_xlsx_path, config_forced_zero_items = prepare_photsim7_config_xlsx(
    config_xlsx_full_path,
    CACHE_DIR,
)
print(f"[Config] Photsim7-compatible config sheet={config_manager_xlsx_path}")

config_manager = ConfigurationManager(filepath=config_manager_xlsx_path)

def resolve_psf_bundle_name() -> tuple[str, str]:
    field_specific_enabled = env_flag("ET_ENABLE_FIELD_SPECIFIC_PSF", False)

    psf_bundle_name_env = os.environ.get("ET_PSF_BUNDLE_NAME_OVERRIDE")
    if psf_bundle_name_env is None:
        # Backward-compatible alias for older launch scripts.
        psf_bundle_name_env = os.environ.get("PSF_BUNDLE_NAME_OVERRIDE")

    if psf_bundle_name_env is not None and psf_bundle_name_env.strip():
        return psf_bundle_name_env.strip(), "env_override"

    if field_specific_enabled:
        return (
            FIELD_SPECIFIC_PSF_BUNDLE_NAMES.get(
                FIELD_CENTER_INDEX,
                PSF_BUNDLE_NAME_DEFAULT.strip(),
            ),
            "field_specific",
        )

    return PSF_BUNDLE_NAME_DEFAULT.strip(), "default"


_psf_bundle_name, _psf_bundle_source = resolve_psf_bundle_name()
config_manager.parameters["PSF Bundle Name"] = _psf_bundle_name
print(
    f"[Config] PSF Bundle Name={config_manager.parameters['PSF Bundle Name']}"
    f"  source={_psf_bundle_source}"
)

_optical_eff_override = env_float_or_none("ET_GUIDE_OPTICAL_EFFICIENCY", None)
_et_qe_override = env_float_or_none(
    "ET_GUIDE_QUANTUM_EFFICIENCY",
    GUIDE_ET_QUANTUM_EFFICIENCY_DEFAULT,
)
_optical_eff_missing_from_input = any(
    "Optical Efficiency Ratio" in item for item in config_manager.missing_log
)
if _optical_eff_override is not None:
    config_manager.parameters["Optical Efficiency Ratio"] = efficiency_to_percent_quantity(
        _optical_eff_override,
        "ET_GUIDE_OPTICAL_EFFICIENCY",
    )
elif _optical_eff_missing_from_input:
    config_manager.parameters["Optical Efficiency Ratio"] = efficiency_to_percent_quantity(
        GUIDE_OPTICAL_EFFICIENCY_RATIO_DEFAULT,
        "GUIDE_OPTICAL_EFFICIENCY_RATIO_DEFAULT",
    )
config_manager.parameters["ET Quantum Efficiency"] = efficiency_to_percent_quantity(
    _et_qe_override,
    "ET_GUIDE_QUANTUM_EFFICIENCY",
)
print(
    "[Config] ET throughput/QE "
    f"throughput={config_manager.parameters['Optical Efficiency Ratio']} "
    f"qe={config_manager.parameters['ET Quantum Efficiency']}"
)

# ---------------------------------------------------------------------------
# 导星探测器参数覆盖：以主探测器 xlsx 为基础，仅在脚本层改写差异项。
# ---------------------------------------------------------------------------
config_manager.parameters["Simulation Name"] = "ET_guide_detector"
config_manager.parameters["Pixel Scale"] = GUIDE_PIXEL_SCALE_ARCSEC_PER_PIX * (
    u.arcsec / u.pix
)
config_manager.parameters["Pixel Width"] = GUIDE_PIXEL_WIDTH_UM * u.um
config_manager.parameters["Exposure Duration"] = GUIDE_EXPOSURE_S * u.s
config_manager.parameters["Readout Duration"] = GUIDE_READOUT_S * u.s
config_manager.parameters["Observing Duration"] = GUIDE_OBSERVING_DURATION_S * u.s
config_manager.parameters["PSF Field ID"] = int(GUIDE_PSF_FIELD_ID)
config_manager.parameters["Readout Noise"] = GUIDE_READOUT_NOISE_E_PER_PIX * (
    u.electron / u.pix
)
# config_manager.parameters["Background Flux"] = GUIDE_BACKGROUND_FLUX_E_PER_S_PER_PIX * (
#     u.electron / u.s / u.pix
# )
config_manager.parameters["Scattered Light"] = (
    GUIDE_SCATTERED_LIGHT_E_PER_S_PER_PIX * u.electron / u.s / u.pix
)
config_manager.parameters["Dark Current"] = GUIDE_DARK_CURRENT_E_PER_S_PER_PIX * (
    u.electron / u.s / u.pix
)
config_manager.parameters["Inter-PRV (RMS)"] = GUIDE_INTER_PRV_RMS_PERCENT * u.percent
config_manager.parameters["Enable ADC Digitization"] = True
config_manager.parameters["Full Well Electrons"] = GUIDE_FULL_WELL_ELECTRONS * u.electron
config_manager.parameters["Gain Electrons Per ADU"] = (
    GUIDE_GAIN_ELECTRONS_PER_ADU * u.electron / u.adu
)
config_manager.parameters["ADC Bit Depth"] = GUIDE_ADC_BIT_DEPTH
config_manager.parameters["ADC Min Value"] = 0.0
config_manager.parameters["ADC Round Values"] = True
config_manager.parameters["Bias Level ADU"] = GUIDE_BIAS_LEVEL_ADU * u.adu
config_manager.parameters["Column Noise Sigma ADU"] = GUIDE_COLUMN_NOISE_SIGMA_ADU * u.adu
config_manager.parameters["Enable Cosmic Rays"] = GUIDE_ENABLE_COSMIC_RAYS
config_manager.parameters["Enable Whole Pixel Gain Normal Distribution"] = False
config_manager.parameters["Enable Whole Pixel Gain Sinusoidal Modulation"] = False
config_manager.parameters["Subtract Nonstellar Mean"] = GUIDE_SUBTRACT_NONSTELLAR_MEAN

FRAME_TIMING = resolve_detector_frame_timing(config_manager.parameters)
RAW_FRAME_INTEGRATION_S = float(FRAME_TIMING["raw_frame_integration_s"])
RAW_FRAME_SAMPLING_INTERVAL_S = float(FRAME_TIMING["raw_frame_sampling_interval_s"])
DETECTOR_TYPE = str(FRAME_TIMING["detector_type"])

if PROFILE_TARGET_FRAMES is not None and PROFILE_TARGET_FRAMES > 0:
    config_manager.parameters["Observing Duration"] = (
        PROFILE_TARGET_FRAMES * RAW_FRAME_SAMPLING_INTERVAL_S
    ) * u.s

if not ACTIVE_EFFECT_PROFILE["uses_jitter_integrated_psf"]:
    config_manager.parameters["Use Jitter-Integrated PSF"] = False

_n_jit_models_override = env_int("ET_N_JITTER_INTEGRATED_PSF_MODELS", None)
if _n_jit_models_override is not None:
    if _n_jit_models_override <= 0:
        raise ValueError("ET_N_JITTER_INTEGRATED_PSF_MODELS must be positive.")
    config_manager.parameters["N Jitter-Integrated PSF Models"] = int(
        _n_jit_models_override
    )
    print(
        "[Config override] N Jitter-Integrated PSF Models="
        f"{config_manager.parameters['N Jitter-Integrated PSF Models']}"
    )

_n_jit_frames_override = env_int("ET_N_JITTER_FRAMES_PER_MODEL", None)
if _n_jit_frames_override is not None:
    if _n_jit_frames_override <= 0:
        raise ValueError("ET_N_JITTER_FRAMES_PER_MODEL must be positive.")
    config_manager.parameters["N Jitter Frames Per Model"] = int(
        _n_jit_frames_override
    )
    print(
        "[Config override] N Jitter Frames Per Model="
        f"{config_manager.parameters['N Jitter Frames Per Model']}"
    )

# ---------------------------------------------------------------------------
# 关键自检：确认 xlsx 的 observing duration / cadence / 帧数
# ---------------------------------------------------------------------------
# 你在终端里看到的进度条总数（例如 x/360、x/10000）通常来自后面
# `for start in trange(0, n_coadd_frames, batch_size)` 的 `n_coadd_frames`。
# 而 `n_coadd_frames` 本质上由 xlsx 的 Observing Duration 和 cadence 决定。
_obs_s = config_manager.parameters["Observing Duration"].to(u.s).value
_exp_s = config_manager.parameters["Exposure Duration"].to(u.s).value
_rd_s = config_manager.parameters["Readout Duration"].to(u.s).value
_mult = float(config_manager.parameters["Simulation Cadence Mult"])
_frame_interval_s = RAW_FRAME_SAMPLING_INTERVAL_S
_derived_n_frames = (
    int(np.floor(_obs_s / _frame_interval_s)) if _frame_interval_s > 0 else 0
)
print(
    f"[Config check] xlsx={config_xlsx_full_path}  "
    f"obs={_obs_s:.3f}s  exp={_exp_s:.3f}s  rd={_rd_s:.3f}s  mult={_mult:g}  "
    f"detector={DETECTOR_TYPE}  integration={RAW_FRAME_INTEGRATION_S:.6f}s  "
    f"frame_interval={_frame_interval_s:.6f}s -> n_frames≈{_derived_n_frames}"
)

# ============================================================================
# 单帧尺寸修改：用导星探测器视场自动换算像素边长，并覆盖配置里的 Detector Width/Height。
# ============================================================================
# 导星仿真按正方形视场运行；Photsim7 会同时读取 Detector Width 和 Detector Height。
# 因此两个方向都需要覆盖，避免旧 xlsx 里的 101 pix height 泄漏到输出数组。
# - mk_real_field_stars_2 查询范围（px_cnt）
# - Detector 构造
# - PSF warp
# - 生成图像数组
# 都同步变成裁剪后的大小。
pixel_scale = config_manager.parameters["Pixel Scale"].to(u.arcsec / u.pix)

# 角尺度换算：Npix = (FOV[deg] * 3600 [arcsec/deg]) / (pixel_scale [arcsec/pix])
fov_arcsec = FOV_DEG * 3600.0
n_pixels_fov = int(np.ceil(fov_arcsec / pixel_scale.value))

# 取奇数像素边长：让几何中心落在像素中心附近，便于后续中心/质心处理。
if (n_pixels_fov % 2) == 0:
    n_pixels_fov += 1

print(
    f"[Frame size] Using FOV={FOV_DEG:.2f} deg and pixel_scale={pixel_scale.value:.4f} arcsec/pix -> "
    f"Detector Width/Height = {n_pixels_fov} pix (forced odd)."
)

config_manager.parameters["Detector Width"] = n_pixels_fov * u.pix
config_manager.parameters["Detector Height"] = n_pixels_fov * u.pix

# 补充写入本批次 metadata（可用于你后续做“同一套 motion、不同天区中心”的对照检查）
try:
    with open(meta_path, "r", encoding="utf-8") as f:
        _meta = json.load(f)
except Exception:
    _meta = {}

_readout_noise_e_per_pix = float(
    config_manager.parameters["Readout Noise"].to(u.electron / u.pix).value
)
_gain_electrons_per_adu = float(
    config_manager.parameters["Gain Electrons Per ADU"].to(u.electron / u.adu).value
)
_readout_noise_adu_per_pix = _readout_noise_e_per_pix / _gain_electrons_per_adu
_enable_adc_digitization = bool(config_manager.parameters["Enable ADC Digitization"])

_meta.update(
    {
        "photsim_core": "photsim7",
        "photsim7_root": str(PHOTSIM7_ROOT),
        "photsim7_data_dir": str(PHOTSIM7_DATA_DIR),
        "instrument": "guide_detector",
        "config_xlsx_full_path": config_xlsx_full_path,
        "config_manager_xlsx_path": config_manager_xlsx_path,
        "config_forced_zero_items": config_forced_zero_items,
        "effect_profile": ACTIVE_EFFECT_PROFILE["name"],
        "effect_profile_requested": requested_effect_profile,
        "effect_profile_target_frames": (
            None if PROFILE_TARGET_FRAMES is None else int(PROFILE_TARGET_FRAMES)
        ),
        "optical_efficiency_ratio_percent": float(
            config_manager.parameters["Optical Efficiency Ratio"].to(u.percent).value
        ),
        "et_quantum_efficiency_percent": float(
            config_manager.parameters["ET Quantum Efficiency"].to(u.percent).value
        ),
        "effect_profile_variant_settings": dict(
            ACTIVE_EFFECT_PROFILE["variant_settings"]
        ),
        "effect_profile_components": dict(RUNTIME_EFFECT_COMPONENTS),
        "enable_frame_to_frame_drift": bool(ENABLE_FRAME_TO_FRAME_DRIFT),
        "psf_bundle_name": str(config_manager.parameters["PSF Bundle Name"]),
        "psf_bundle_source": str(_psf_bundle_source),
        "psf_field_angle_deg": float(GUIDE_PSF_FIELD_ANGLE_DEG),
        "psf_field_id": int(GUIDE_PSF_FIELD_ID),
        "fov_deg": float(FOV_DEG),
        "pixel_scale_arcsec_per_pix": float(pixel_scale.value),
        "detector_width_pix": int(n_pixels_fov),
        "detector_height_pix": int(n_pixels_fov),
        "pixel_width_um": float(GUIDE_PIXEL_WIDTH_UM),
        "exposure_duration_s": float(
            config_manager.parameters["Exposure Duration"].to(u.s).value
        ),
        "readout_duration_s": float(
            config_manager.parameters["Readout Duration"].to(u.s).value
        ),
        "observing_duration_s": float(
            config_manager.parameters["Observing Duration"].to(u.s).value
        ),
        # "background_surface_brightness_mag_per_arcsec2": float(
        #     GUIDE_BACKGROUND_SURFACE_BRIGHTNESS_MAG_PER_ARCSEC2
        # ),
        "background_flux_e_per_s_per_pix": float(
            config_manager.parameters["Background Flux"]
            .to(u.electron / u.s / u.pix)
            .value
        ),
        "dark_current_e_per_s_per_pix": float(
            config_manager.parameters["Dark Current"].to(u.electron / u.s / u.pix).value
        ),
        "scattered_light_e_per_s_per_pix": float(
            config_manager.parameters["Scattered Light"]
            .to(u.electron / u.s / u.pix)
            .value
        ),
        "dark_plus_scattered_e_per_s_per_pix": float(
            config_manager.parameters["Dark Current"].to(u.electron / u.s / u.pix).value
            + config_manager.parameters["Scattered Light"]
            .to(u.electron / u.s / u.pix)
            .value
        ),
        "readout_noise_adu_per_pix": float(_readout_noise_adu_per_pix),
        "readout_noise_e_per_pix": float(_readout_noise_e_per_pix),
        "gain_adu_per_electron": float(1.0 / _gain_electrons_per_adu),
        "gain_electrons_per_adu": float(_gain_electrons_per_adu),
        "enable_adc_digitization": _enable_adc_digitization,
        "full_well_electrons": float(
            config_manager.parameters["Full Well Electrons"].to(u.electron).value
        ),
        "adc_bit_depth": int(config_manager.parameters["ADC Bit Depth"]),
        "adc_min_value": float(config_manager.parameters["ADC Min Value"]),
        "adc_round_values": bool(config_manager.parameters["ADC Round Values"]),
        "bias_level_adu": float(
            config_manager.parameters["Bias Level ADU"].to(u.adu).value
        ),
        "column_noise_sigma_adu": float(
            config_manager.parameters["Column Noise Sigma ADU"].to(u.adu).value
        ),
        "enable_cosmic_rays": bool(config_manager.parameters["Enable Cosmic Rays"]),
        "subtract_nonstellar_mean": bool(
            config_manager.parameters["Subtract Nonstellar Mean"]
        ),
        "image_unit": "adu_dn" if _enable_adc_digitization else "electron",
        "inter_prv_rms_percent": float(
            config_manager.parameters["Inter-PRV (RMS)"].to(u.percent).value
        ),
        "intra_prv_rms_percent": float(
            config_manager.parameters["Intra-PRV (RMS)"].to(u.percent).value
        ),
        "gaia_gmag_lim": float(GAIA_GMAG_LIM),
        "apply_static_field_offset": bool(APPLY_STATIC_FIELD_OFFSET),
        "requested_field_offset_x_pix": (
            None
            if STATIC_FIELD_OFFSET_X_PIX is None
            else float(STATIC_FIELD_OFFSET_X_PIX)
        ),
        "requested_field_offset_y_pix": (
            None
            if STATIC_FIELD_OFFSET_Y_PIX is None
            else float(STATIC_FIELD_OFFSET_Y_PIX)
        ),
        "stream_save_frames": bool(STREAM_SAVE_FRAMES),
        "frame_batch_size": int(FRAME_BATCH_SIZE),
        "frame_truth_schema_version": 2,
        "frame_truth_coordinate_convention": (
            "centered_pix uses detector-centered zero point with +y downward; "
            "image_pix uses zero-based detector array coordinates"
        ),
        "frame_truth_absolute_coordinate_fields": [
            "truth_abs_x_image_pix",
            "truth_abs_y_image_pix",
            "truth_abs_x_detector_pix",
            "truth_abs_y_detector_pix",
        ],
        "frame_truth_source": "photsim6ft_frame_truth_builder",
        "field_polar_angle_rad": float(FIELD_POLAR_ANGLE_RAD),
        "jitter_truth_mode": "shared_per_frame_mean",
        "enable_periodic_radial_expansion": bool(ENABLE_PERIODIC_RADIAL_EXPANSION),
        "radial_expansion_period_s": float(RADIAL_EXPANSION_PERIOD_S),
        "radial_expansion_active_last_s": float(RADIAL_EXPANSION_ACTIVE_LAST_S),
        "radial_expansion_arcsec": float(RADIAL_EXPANSION_ARCSEC),
        "radial_expansion_unit_source": str(_RADIAL_UNIT_SOURCE),
        "radial_expansion_unit_x": float(FIELD_CENTER_RADIAL_UNIT_X_DEG),
        "radial_expansion_unit_y": float(FIELD_CENTER_RADIAL_UNIT_Y_DEG),
        "radial_expansion_fov_center_ra_deg": (
            None
            if RADIAL_EXPANSION_FOV_CENTER_RA_DEG is None
            else float(RADIAL_EXPANSION_FOV_CENTER_RA_DEG)
        ),
        "radial_expansion_fov_center_dec_deg": (
            None
            if RADIAL_EXPANSION_FOV_CENTER_DEC_DEG is None
            else float(RADIAL_EXPANSION_FOV_CENTER_DEC_DEG)
        ),
    }
)

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(_meta, f, ensure_ascii=False, indent=2)

# config_manager.parameters['Observing Duration'] = 7 * u.day

# config_manager.parameters

variant_manager = VariantManager()  # Initialize the manager

# Add variant 0 - active effect profile
variant_manager.add_variant(
    description=ACTIVE_EFFECT_PROFILE["description"],
    optimal_aperture=config_manager.parameters["Optimal Aperture Algorithim"],
    **ACTIVE_EFFECT_PROFILE["variant_settings"],
)

"""Example extra variant"""
# variant_manager.add_variant(
#     description='1',
#     optimal_aperture=config_manager.parameters['Optimal Aperture Algorithim'],
#     enable_stellar_photon_noise=False,
# )

# accepted_settings = {
#     'description': 'None',
#     'enable_stellar_photon_noise': True,
#     'enable_background_light': True,
#     'enable_scattered_light': True,
#     'enable_dark_current': True,
#     'enable_readout_noise': True,
#     'enable_gain': True,
#     'enable_target_star': True,
#     'enable_background_stars': True,
#     'enable_jitter': True,
#     'enable_dva_drift': True,
#     'enable_pointing_drift': True,
#     'enable_psf_breathing': True,
#     'enable_inter_pixel_response': True,
#     'enable_intra_pixel_response': True,
#     'enable_pixel_phase_response': True,
#     'enable_flat_field_correction': False,
#     'use_oa_from_variant': 0,
#     'optimal_aperture': 'Kepler',
#     'enable_coadding': True,
# }

# variant_manager.add_variant(
#     description='1',
#     optimal_aperture=config_manager.parameters['Optimal Aperture Algorithim'],
#     enable_dva_drift=False,
#     enable_pointing_drift=False,
#     use_oa_from_variant=1,
# )

# variant_manager.add_variant(
#     description='2',
#     optimal_aperture=config_manager.parameters['Optimal Aperture Algorithim'],
#     enable_jitter=False,
#     use_oa_from_variant=2,
# )

# variant_manager.add_variant(
#     description='3',
#     optimal_aperture=config_manager.parameters['Optimal Aperture Algorithim'],
#     enable_psf_breathing=False,
#     use_oa_from_variant=3,
# )

# ---------------------------------------------------------------------------------------------

print(
    "--------------------------------------------------\nUser-defined variant catalog\n\n"
)
# Display the user-defined variant catalog
# display(variant_manager.variants_dataframe)
print(
    "\n--------------------------------------------------\nUser-defined + optimal aperture required variant catalog\n\n"
)
# Display the user-defined + optimal aperture required variants catalog
# display(variant_manager.all_variants_dataframe)

del variant_manager.all_variants[1]
del variant_manager.all_variants[2]

variant_manager._update_all_variants_dataframe()

print(
    "--------------------------------------------------\nUser-defined variant catalog\n\n"
)
# Display the user-defined variant catalog
# display(variant_manager.variants_dataframe)
print(
    "\n--------------------------------------------------\nUser-defined + optimal aperture required variant catalog\n\n"
)
# Display the user-defined + optimal aperture required variants catalog
# display(variant_manager.all_variants_dataframe)

# Load and parse motion power spectrum data into a Dataframe (DF)
power_spec_config = load_power_spectrum_config(config_xlsx_full_path)

# Build the power spectrum dictionary from the DF
comps = create_power_spec_components(power_spec_config)

# display(power_spec_config)  # Show DF

comps  # Show dict

# Build the motion power spectrum
jit_spec_mask, jitter_freq_mag, drift_spec_mask, drift_freq_mag = gen_psd_motion(
    comps, plot=False
)

et_freqs, et_amps, jit_freq_range = gen_power_spec(comps)

mask2 = (et_freqs >= 1 / (5 * u.hr.to(u.s))) & (et_freqs <= 1 / (30 * u.min.to(u.s)))
et_amps[mask2] = 0.00014079 * 1.8


# -----------------------------------------------------------------------------
# Drift/Jitter 分界（脚本层实现）：split_hz = 1 / raw-frame integration time
# -----------------------------------------------------------------------------
# 这里统一使用：
# - ET PSD：只保留 <= split_hz 的低频部分，作为“帧间漂移/慢变化”
# - TESS PSD：只保留 >  split_hz 的高频部分，作为“曝光内抖动/快变化”
# 这样可以避免把本应留下帧间质心差异的运动误当作 jitter 进入 PSF 卷积。
raw_frame_integration_s = RAW_FRAME_INTEGRATION_S
raw_frame_sampling_interval_s = RAW_FRAME_SAMPLING_INTERVAL_S

if USE_CADENCE_SPLIT:
    split_hz = 1.0 / float(raw_frame_integration_s)
else:
    # 兼容旧逻辑（历史脚本用 0.1Hz）：不建议用于 1.6s cadence 的导星评估。
    split_hz = 0.1

print(
    f"[Motion split] detector={DETECTOR_TYPE}  integration={raw_frame_integration_s:.3f}s  "
    f"frame_interval={raw_frame_sampling_interval_s:.3f}s -> split_hz={split_hz:.6f} Hz "
    f"(drift<=split, jitter>split)."
)

if SPLIT_HZ_OVERRIDE is not None:
    split_hz = float(SPLIT_HZ_OVERRIDE)
    print(
        f"[Motion split] SPLIT_HZ_OVERRIDE is set -> split_hz={split_hz:.6f} Hz "
        f"(drift<=split, jitter>split)."
    )

# 只保留 ET 的低频部分（<= split_hz），高频部分由 TESS jitter 负责。
et_amps = et_amps.copy()
et_amps[et_freqs > split_hz] = 0.0

et_amp_spec = [et_freqs, et_amps]

pixel_scale = config_manager.parameters["Pixel Scale"]

plot_psd(comps, et_freqs, et_amps)
plt.plot(et_freqs, et_amps * pixel_scale.value, ":c", label="All")
plt.legend()

len(jit_spec_mask), jit_spec_mask.sum(), len(drift_spec_mask), drift_spec_mask.sum()

TESS_PSD_PATH = env_str(
    "ET_TESS_PSD_PATH",
    "/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz_within_1s_scaled_to_0p6arcsec.pkl",
    #"/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz_within_1s_scaled_to_0p12arcsec.pkl",
    #"/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz.pkl",
)
if not os.path.isfile(TESS_PSD_PATH):
    raise FileNotFoundError(f"TESS jitter PSD not found: {TESS_PSD_PATH}")
print(f"[Jitter PSD] {TESS_PSD_PATH}")
TESS_PSD_CACHE_LABEL = Path(TESS_PSD_PATH).stem
tess_psd = load_pickle(TESS_PSD_PATH)

tess_jitter_mult = 1  # Use the jitter amplitude encoded by TESS_PSD_PATH directly.

# ============================================================================
# JI-PSF jitter 序列生成（只用于“曝光内模糊”）：只取 > split_hz 的高频部分
# ============================================================================
# 重要：photsim6 的 jitter-integrated PSF 是“把 PSF 在一条 jitter 轨迹上平移多帧，再求平均”。
# 因此 xy_jitter_pix 的每个条目应该代表“一次曝光内”的快抖动轨迹。
# 我们把 <= split_hz 的慢变化都剥离到 drift（每帧中心偏移）里，避免把慢漂移错误地卷进 PSF。


def _angles_to_xy_pix(
    theta_x_arcsec,
    theta_y_arcsec,
    theta_z_arcsec,
    field_angle_deg,
    x_axis_angle_deg,
    plate_scale,
):
    """把姿态角(arcsec)转换为焦平面上的 x/y 像素位移。

    这里复用 photsim6ft 的思路：
    - 用一个代表全幅焦平面坐标系的位置 (x_pix, y_pix) 来施加旋转
    - displacement = old - new

    注意：即使我们只仿真裁剪图（n_pixels 变小），旋转导致的位移仍由“真实 field_angle”决定，
    与你裁剪多少像素无关。所以这里仍然用 23.5°/9000pix 的全幅映射来取参考点。
    """
    from photsim6.data_generators import PixelSpaceSimulator

    x_center = 0.0
    y_center = 0.0
    et_pss = PixelSpaceSimulator(
        plate_scale=plate_scale, x_center=x_center, y_center=y_center
    )

    fa_theta_rad = x_axis_angle_deg / 180.0 * np.pi
    x_fa = np.cos(fa_theta_rad) * field_angle_deg
    y_fa = np.sin(fa_theta_rad) * field_angle_deg

    x_pix = np.interp(x_fa, [0, 23.5 / 2], [0, 9000])
    y_pix = np.interp(y_fa, [0, 23.5 / 2], [0, 9000])

    x_pix_new, y_pix_new = et_pss.apply_spacecraft_rotations(
        theta_x_arcsec * u.arcsec,
        theta_y_arcsec * u.arcsec,
        theta_z_arcsec * u.arcsec,
        x_pix,
        y_pix,
    )

    dx_pix = x_pix - x_pix_new
    dy_pix = y_pix - y_pix_new
    return dx_pix, dy_pix


def generate_tess_centroid_xy_motion_split(
    time,
    tess_psd,
    split_hz,
    field_angle_deg=10.0,
    x_axis_angle_deg=45.0,
    tess_mult=1.0,
    highpass=True,
    plot=False,
):
    """从 TESS PSD 生成 x/y 像素位移，并按 split_hz 做高/低频分割。

    - highpass=True  -> 只取 > split_hz 的高频部分（jitter，供 JI-PSF 使用）
    - highpass=False -> 只取 <= split_hz 的低频部分（drift，供每帧中心偏移使用）

    这里显式调用 `generate_tess_jitter()` 而不是用包内 `generate_tess_centroid_jitter()`：
    因为后者内部把 x/y 的 frequency_min 硬编码成 0.1Hz，不适配你要的 0.625Hz 分界。
    """
    from photsim6.data_generators import generate_tess_jitter

    def _resample_to_time_grid(values, src_time_s, dst_time_s):
        """把 generate_tess_jitter() 的输出重采样到给定的 time 网格上。

        背景：generate_tess_jitter 内部用 `N_time = int(duration * f_samp) + 1`。
        当 duration*f_samp 因浮点误差略小于整数时，会少 1 个采样点（例如 9999 而不是 10000），
        导致后续与 `time` 的长度不一致并触发 broadcast 错误。
        """
        values = np.asarray(values, dtype=float)
        dst_time_s = np.asarray(dst_time_s, dtype=float)
        if values.shape[0] == dst_time_s.shape[0]:
            return values

        src_time_s = np.asarray(src_time_s, dtype=float)
        if src_time_s.shape[0] != values.shape[0]:
            src_time_s = np.linspace(dst_time_s[0], dst_time_s[-1], num=values.shape[0])
        if src_time_s[0] > src_time_s[-1]:
            src_time_s = src_time_s[::-1]
            values = values[::-1]

        return np.interp(dst_time_s, src_time_s, values)

    ztime_s = (time - time[0]).to(u.s).value
    if ztime_s.size >= 2:
        dt = float(ztime_s[1] - ztime_s[0])
    else:
        dt = float(raw_frame_sampling_interval_s)
    f_samp = 1.0 / dt
    # 用 (t[-1] + dt) 而不是 t[-1]，避免 duration*f_samp 因浮点误差落到整数的左侧。
    duration = float(ztime_s[-1] + dt)

    if highpass:
        fmin = float(split_hz)
        fmax = np.inf
    else:
        fmin = 0.0
        fmax = float(split_hz)

    x_tess_freqs, x_tess_psd = tess_psd["x"]
    y_tess_freqs, y_tess_psd = tess_psd["y"]
    z_tess_freqs, z_tess_psd = tess_psd["z"]

    theta_x_as, t_x = generate_tess_jitter(
        x_tess_freqs.value,
        x_tess_psd.value,
        frequency_min=fmin,
        frequency_max=fmax,
        duration=duration,
        base_f_samp=f_samp,
        supersample_factor=1,
    )
    theta_y_as, t_y = generate_tess_jitter(
        y_tess_freqs.value,
        y_tess_psd.value,
        frequency_min=fmin,
        frequency_max=fmax,
        duration=duration,
        base_f_samp=f_samp,
        supersample_factor=1,
    )
    theta_z_as, t_z = generate_tess_jitter(
        z_tess_freqs.value,
        z_tess_psd.value,
        frequency_min=fmin,
        frequency_max=fmax,
        duration=duration,
        base_f_samp=f_samp,
        supersample_factor=1,
    )

    # 兜底：确保输出与 time 的采样点数严格一致
    theta_x_as = _resample_to_time_grid(theta_x_as, t_x, ztime_s)
    theta_y_as = _resample_to_time_grid(theta_y_as, t_y, ztime_s)
    theta_z_as = _resample_to_time_grid(theta_z_as, t_z, ztime_s)

    # 与旧脚本一致：仅放大 x/y 的抖动幅度（tess_mult），z 轴不放大。
    theta_x_as = theta_x_as * tess_mult
    theta_y_as = theta_y_as * tess_mult

    if plot:
        plt.figure(figsize=(11, 4))
        plt.plot(ztime_s, theta_x_as, ".-", label="theta_x [arcsec]")
        plt.plot(ztime_s, theta_y_as, ".-", label="theta_y [arcsec]")
        plt.plot(ztime_s, theta_z_as, ".-", label="theta_z [arcsec]")
        plt.title(
            f"TESS {'high' if highpass else 'low'}-freq motion "
            f"(f in [{fmin}, {fmax}])"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    dx_pix, dy_pix = _angles_to_xy_pix(
        theta_x_arcsec=theta_x_as,
        theta_y_arcsec=theta_y_as,
        theta_z_arcsec=theta_z_as,
        field_angle_deg=field_angle_deg,
        x_axis_angle_deg=x_axis_angle_deg,
        plate_scale=pixel_scale,
    )

    return dx_pix, dy_pix


def generate_et_xy_motion_lowfreq(time, et_amp_spec, split_hz):
    """从 ET PSD 生成低频(<=split_hz)的 x/y 像素位移，用作 drift。"""
    ztime_s = (time - time[0]).to(u.s).value
    et_f, et_amp = et_amp_spec

    # 低频漂移：<= split_hz
    mask = (et_f <= split_hz) & (et_amp > 0)
    if mask.sum() == 0:
        return np.zeros_like(ztime_s, dtype=float), np.zeros_like(ztime_s, dtype=float)

    xy_phases = [np.random.uniform(0, 2 * np.pi, mask.sum()) for _ in range(2)]
    x_pix = psd_to_motion(et_f[mask], et_amp[mask], xy_phases[0], ztime_s)
    y_pix = psd_to_motion(et_f[mask], et_amp[mask], xy_phases[1], ztime_s)
    return x_pix, y_pix


def generate_jitter_xy_highfreq_for_exposure(
    time, et_amp_spec, tess_psd, split_hz, tess_mult=1.0, plot=False
):
    """生成一次曝光内的高频 jitter 轨迹（>split_hz），供 JI-PSF 模糊使用。

    这里选择：
    - ET 高频部分已被上面 et_amps[et_freqs>split]=0 剔除（避免与 TESS jitter 叠加）。
    - jitter 统一来自 TESS PSD 的高频部分 (> split_hz)。
    """
    x_jit_tess_pix, y_jit_tess_pix = generate_tess_centroid_xy_motion_split(
        time=time,
        tess_psd=tess_psd,
        split_hz=split_hz,
        field_angle_deg=10.0,
        x_axis_angle_deg=45.0,
        tess_mult=tess_mult,
        highpass=True,
        plot=plot,
    )
    return x_jit_tess_pix, y_jit_tess_pix


resume = True

use_jitter_integrated_psf = bool(ACTIVE_EFFECT_PROFILE["uses_jitter_integrated_psf"])

# Get key params for jitter generation and the filename
n_jitter_frames_per_model = int(config_manager.parameters["N Jitter Frames Per Model"])
n_jit_psf = config_manager.parameters["N Jitter-Integrated PSF Models"]

jitter_mag_str = str(np.around(comps["Jitter: All"]["mag"], 7))
jitter_frq_str = str(comps["Jitter: All"]["freqs"]).replace(" ", "")

# Define filename and save/load directory
# jit_fn = f'xy_jitter_pix_N({n_jit_psf})_AFJ({jitter_mag_str}_{jitter_frq_str})_exp{raw_frame_integration_s:.2f}s.npy'

# 文件名里显式写入 split_hz，避免你切换 cadence 后误读旧缓存。
jit_fn = (
    f"xy_{tess_jitter_mult}X_jitter_pix_N({n_jit_psf})_"
    f"AFJ({TESS_PSD_CACHE_LABEL})_{DETECTOR_TYPE}_"
    f"exp{raw_frame_integration_s:.6f}s_"
    f"nsub{n_jitter_frames_per_model}_"
    f"split{split_hz:.6f}Hz.npy"
)

jit_fp = opj(CACHE_DIR, "jitter", "et", "JI-PSF_data", jit_fn)

os.makedirs(os.path.dirname(jit_fp), exist_ok=True)

# -------------------------------------------------------------------

if use_jitter_integrated_psf:
    if os.path.isfile(jit_fp) and resume:
        print("Loaded jitter")
        xy_jitter_pix = np.load(jit_fp)
    else:
        xy_jitter_pix = []

        for i in trange(
            config_manager.parameters["N Jitter-Integrated PSF Models"]
        ):  # For each JI-PSF

            et_time = (
                np.linspace(
                    0.0,
                    raw_frame_integration_s,
                    num=n_jitter_frames_per_model,
                    endpoint=False,
                    dtype=np.float64,
                )
                * u.s
            )

            # 只生成 > split_hz 的高频抖动：用于“曝光内 PSF 模糊”
            x_jit_pix, y_jit_pix = generate_jitter_xy_highfreq_for_exposure(
                time=et_time,
                et_amp_spec=et_amp_spec,
                tess_psd=tess_psd,
                split_hz=split_hz,
                tess_mult=tess_jitter_mult,
                plot=False,
            )

            xy_jitter_pix.append([x_jit_pix, y_jit_pix])

        print("Saved jitter")
        np.save(jit_fp, xy_jitter_pix)
else:
    xy_jitter_pix = np.zeros((1, 2, 1), dtype=np.float32)
    print(
        '[Effect profile] Skip jitter cache and force "Use Jitter-Integrated PSF" to FALSE.'
    )

jit_fp


def generate_tess_roll_xy_drift(
    time, tess_psd, split_hz, field_angle=10.0, x_axis_angle=45
):

    from photsim6.data_generators import PixelSpaceSimulator

    x_center = 0.0
    y_center = 0.0

    et_pss = PixelSpaceSimulator(
        plate_scale=pixel_scale,
        x_center=x_center,
        y_center=y_center,
    )
    # -----------------------------------------------------------------------------------
    # TESS 低频(<=split_hz) roll motion -> x, y motion（作为 drift 的一部分）

    def _resample_to_time_grid(values, src_time_s, dst_time_s):
        values = np.asarray(values, dtype=float)
        dst_time_s = np.asarray(dst_time_s, dtype=float)
        if values.shape[0] == dst_time_s.shape[0]:
            return values

        src_time_s = np.asarray(src_time_s, dtype=float)
        if src_time_s.shape[0] != values.shape[0]:
            src_time_s = np.linspace(dst_time_s[0], dst_time_s[-1], num=values.shape[0])
        if src_time_s[0] > src_time_s[-1]:
            src_time_s = src_time_s[::-1]
            values = values[::-1]
        return np.interp(dst_time_s, src_time_s, values)

    ztime_s = (time - time[0]).to(u.s).value
    if ztime_s.size >= 2:
        dt = float(ztime_s[1] - ztime_s[0])
    else:
        dt = float(raw_frame_sampling_interval_s)
    f_samp = 1.0 / dt
    duration = float(ztime_s[-1] + dt)

    from photsim6.data_generators import generate_tess_jitter

    z_tess_freqs, z_tess_psd = tess_psd["z"]
    theta_z_arcsec, t_z = generate_tess_jitter(
        z_tess_freqs.value,
        z_tess_psd.value,
        frequency_min=0.0,
        frequency_max=float(split_hz),
        duration=duration,
        base_f_samp=f_samp,
        supersample_factor=1,
    )

    theta_z_arcsec = _resample_to_time_grid(theta_z_arcsec, t_z, ztime_s)

    fa_theta_rad = x_axis_angle / 180 * np.pi
    x_fa = np.cos(fa_theta_rad) * field_angle
    y_fa = np.sin(fa_theta_rad) * field_angle

    x_pix = np.interp(x_fa, [0, 23.5 / 2], [0, 9000])
    y_pix = np.interp(y_fa, [0, 23.5 / 2], [0, 9000])

    x_pix_new, y_pix_new = et_pss.apply_spacecraft_rotations(
        0 * u.arcsec,
        0 * u.arcsec,
        theta_z_arcsec * u.arcsec,
        x_pix,
        y_pix,
    )

    x_roll_pix = x_pix - x_pix_new
    y_roll_pix = y_pix - y_pix_new

    return x_roll_pix, y_roll_pix, theta_z_arcsec


time_example = np.arange(0, 2 * u.day.to(u.s), 10) * u.s

# theta_z = get_roll_angles(time_example)
x_drift_tess_roll_pix, y_drift_tess_roll_pix, z_drift_theta_as = (
    generate_tess_roll_xy_drift(time_example, tess_psd, split_hz=split_hz)
)

print(
    f"Generated {len(z_drift_theta_as)} roll angles for times {time_example[0]} to {time_example[-1]} s"
)
print("First few roll angles [arcsec]:", z_drift_theta_as[:5])

from photsim6.plot import centroid_motion_and_component_plot_with_r95

r_stat_circles = [95]
r_stat_colors = None

fig = centroid_motion_and_component_plot_with_r95(
    t=time_example.to(u.s),
    x=x_drift_tess_roll_pix * pixel_scale.value,
    y=y_drift_tess_roll_pix * pixel_scale.value,
    title="",
    name="Drift",
    xy_unit="arcsec",
    time_unit="day",
    figsize=(16, 6),
    r_stat_circles=r_stat_circles,
    r_stat_colors=r_stat_colors,
)

# Define dynamic param configs (DPC)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Build the main dict that holds all DPC
dynamic_param_config = {}
dynamic_param_config["motion"] = []

# ---------------------------------
# Momentum dump of spacecraft reaction wheels

if ACTIVE_EFFECT_PROFILE["include_momentum_dump"] and (
    config_manager.parameters["Momentum Dump Model"].lower() != "none"
):  # Was MD requested?

    # Get user requested MD settings
    jump_cadence = config_manager.parameters["Momentum Dump Cycle"]
    jump_length = config_manager.parameters["Momentum Dump R(68%)"]
    jump_circ_rad = config_manager.parameters["Momentum Dump R(68%)"]

    obs_duration = config_manager.parameters["Observing Duration"]
    if obs_duration.to(u.s).value < jump_cadence.to(u.s).value:
        print(
            "[Momentum dump] skip: observing duration "
            f"{obs_duration.to(u.s).value:.3f}s < cycle {jump_cadence.to(u.s).value:.3f}s."
        )
    else:
        # Create the MD DPC item
        dynamic_param_config["motion"] += [  # Its motion, so add to the motion catagory
            dict(
                component_name="momentum_dump_jumps",
                model_name=config_manager.parameters["Momentum Dump Model"],
                model_params=dict(
                    period_step=jump_cadence,
                    circle_radius=jump_circ_rad.to(u.arcsec).value / pixel_scale.value,
                    r_step_avg=jump_length.to(u.arcsec).value / pixel_scale.value,
                    stay_inside=True,
                    random_r_step=True,
                    step_func=True,
                ),
            ),
        ]

# ---------------------------------
# DVA motion

dva_model_data = load_pickle(
    opj(BASE_DATA_DIR, "DVA", "et", "ET_DVA_effect_models_slim_v231117.pkl")
)  # Load DVA model (created from JWST code)

if ACTIVE_EFFECT_PROFILE["include_dva"]:
    dynamic_param_config["motion"] += [
        dict(
            component_name="dva_drift",
            model_name="dva_model",  # Built-in DVA model
            model_params=dict(
                dva_model=dva_model_data,
                psf_field_angle=12.0,  # Field angle (FA) of 12 deg is where mots stars will be observed in ET
                pixel_scale=pixel_scale.value,
                t0=0.0,
            ),
        ),
    ]

# ---------------------------------
# Thermal drift due to thermal defocus
"""
Based on TESS lens temperatures and comverted to ET motion using a 
ET lens temp to centorid motion model provided by ET engineers.
"""

if ACTIVE_EFFECT_PROFILE["include_thermal"]:
    dynamic_param_config["motion"] += [
        dict(
            component_name="thermal_drift",
            model_name="et_tess_thermal_drift_model",
            model_params=dict(
                field_angle=10.0,  # Cant change until more data from ET engineers..
            ),
        ),
    ]

# ---------------------------------
# Other drifts (from power spectrum)
"""
This eventually needs to be better implemented. Currently, it uses a custom model
which is defined in the function below (uses global variables! Ah!). 
"""


def generate_fft_hf_drift(time):

    ztime_s = (time - time[0]).to(u.s).value

    def _ensure_len_on_ztime(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 0:
            return np.full(ztime_s.shape, float(arr), dtype=float)
        if arr.shape[0] == ztime_s.shape[0]:
            return arr
        src_t = np.linspace(ztime_s[0], ztime_s[-1], num=arr.shape[0])
        return np.interp(ztime_s, src_t, arr)

    # =====================================================================
    # 选项A（脚本层分界）：生成每帧中心漂移 drift（<= split_hz）
    # =====================================================================
    # 这里返回的 x/y 会被 image.py 叠加到每帧 PSF 中心位置上（frame-to-frame motion）。
    # 与之对应的“曝光内高频 jitter”已经通过 xy_jitter_pix 进入 JI-PSF 模糊。

    # 1) ET 低频 drift（<= split_hz）
    x_drift_et_pix, y_drift_et_pix = generate_et_xy_motion_lowfreq(
        time=time, et_amp_spec=et_amp_spec, split_hz=split_hz
    )
    x_drift_et_pix = _ensure_len_on_ztime(x_drift_et_pix)
    y_drift_et_pix = _ensure_len_on_ztime(y_drift_et_pix)

    # 2) TESS 低频 roll drift（<= split_hz）
    x_drift_tess_roll_pix, y_drift_tess_roll_pix, _ = generate_tess_roll_xy_drift(
        time=time,
        tess_psd=tess_psd,
        split_hz=split_hz,
        field_angle=10.0,
        x_axis_angle=45,
    )
    x_drift_tess_roll_pix = _ensure_len_on_ztime(x_drift_tess_roll_pix)
    y_drift_tess_roll_pix = _ensure_len_on_ztime(y_drift_tess_roll_pix)

    # 3) 合并得到总 drift（不包含 DVA/thermal/MD 等，它们在其他 motion 组件里单独加）
    x_drift_pix = x_drift_et_pix + x_drift_tess_roll_pix
    y_drift_pix = y_drift_et_pix + y_drift_tess_roll_pix

    if ENABLE_PERIODIC_RADIAL_EXPANSION:
        x_periodic_pix, y_periodic_pix = build_periodic_radial_expansion_pix(
            time_s=ztime_s,
            pixel_scale_arcsec_per_pix=pixel_scale.value,
            period_s=RADIAL_EXPANSION_PERIOD_S,
            active_last_s=RADIAL_EXPANSION_ACTIVE_LAST_S,
            radial_arcsec=RADIAL_EXPANSION_ARCSEC,
            radial_unit_x=FIELD_CENTER_RADIAL_UNIT_X_DEG,
            radial_unit_y=FIELD_CENTER_RADIAL_UNIT_Y_DEG,
        )
        x_drift_pix = x_drift_pix + x_periodic_pix
        y_drift_pix = y_drift_pix + y_periodic_pix

    return x_drift_pix, y_drift_pix


if ENABLE_FRAME_TO_FRAME_DRIFT:
    dynamic_param_config["motion"] += [
        dict(
            component_name="fft_hf_drift",
            model_name="user_input_xy_function",  # Tells the generator to use a custom model function
            model_params=dict(
                func=generate_fft_hf_drift,
            ),
        ),
    ]
else:
    print("[Motion] Skip frame-to-frame drift registration; jitter-only centroid input.")

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# PSF breathing
"""
This eventually needs to be better implemented. Currently, it uses a custom model
which is defined in the function imported below. This function is similar to the 
thermal_drift DP in that it uses TESS lens temperatures and an ET temp to PSF breathing
model (from engineers) to generate PSF breathing in the simulation.
"""
from photsim6.data_generators import et_time_to_breathing

dynamic_param_config["psf_scale"] = []

dynamic_param_config["psf_scale"] += [
    # dict(
    #     component_name='psf_scale',
    #     model_name='user_input_r_function',
    #     model_params=dict(
    #         func=et_time_to_breathing, # ET thermal PSF breathing derived from TESS temperatures and generated using ET field angle of ONLY 10 deg!
    #     ),
    # ),
    dict(
        component_name="psf_scale",
        model_name="sine_1d",
        model_params=dict(
            amp=3.0 / 100,
            period=3.0 * u.day,
            offset=1.0,
        ),
    ),
]

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# Display the motion-only DPC dict, eventually provided to the simulator
dynamic_param_config["motion"], dynamic_param_config["psf_scale"]

inject_transits = False
# inject_transits = True

if inject_transits:

    n_exop = 20  # How many transits to be injected in a batch of stars.
    n_stars = 500  # How many stars will you simulate per batch? Typically simulate 500 stars per batch.

    # Build the dict that will be provided to the simulator to generate 4 earth-like transits per light curve, regardless of observing length
    transit_data = dict(
        star_ids=np.sort(
            np.random.permutation(n_stars)[:n_exop]
        ),  # In the batch, which star IDs will have transits injected?
        star_mags=np.full(
            n_exop, 13.0
        ),  # What should be the magnitude of each star with transits?
        to_img_gen=[],  # The relative flux profile for each transit. If not provided, it will be autmatically generated for you.
    )

else:
    transit_data = None

transit_data

config_manager.parameters["Telescope Count"] = 1

simulator = Simulator(
    sim_config=config_manager.parameters,
    dynamic_param_config=dynamic_param_config,
    variant_manager=variant_manager,
    n_raw_frames_per_coadd=1,
    xy_jitter_pix=xy_jitter_pix,
    compute_device=COMPUTE_DEVICE,
    float_precision=32,
    mag_range=[7, 17], # [7,17] #***
    # store_images=True,
    start_ray=False,
    transit_data=transit_data,
    ray_cluster_address=None,
    ray_namespace=None,
    mag_type="ET",
    # ray_cluster_address='ray://100.125.174.11:10001',
    # ray_namespace="persistent_actors",
)

simulator.time_manager.summary

# Confirm the readout noise for coadded frames (RN * sqrt(n_raw_frames_per_coadd))
simulator.sim_config["Sim Readout Noise"]


from photsim6.psf.model import PSFModelManager

# from photsim6.psf.utils import generate_random_field_id
# from photsim6.lc_processing import generate_variant_light_curves
from photsim6.image import ImageProcessing, generate_images_for_star

# from photsim6.aperture import build_all_variant_oa
# # from photsim6.analysis import measure_centroids
from photsim6.field import Stars, generate_field_stars

# from photsim6.instrumentation import Observatory

actor_config = simulator.build_actor_config()

# 进一步补充本批次 metadata：帧数/时间采样等
try:
    with open(meta_path, "r", encoding="utf-8") as f:
        _meta = json.load(f)
except Exception:
    _meta = {}

_meta.update(
    {
        "requested_compute_device": REQUESTED_COMPUTE_DEVICE,
        "compute_device": COMPUTE_DEVICE,
        "detector_type": DETECTOR_TYPE,
        "telescope_count": int(config_manager.parameters["Telescope Count"]),
        "n_coadd_frames": int(actor_config.get("n_coadd_frames")),
        "sim_frame_duration_s": float(
            simulator.time_manager.sim_frame_duration.to(u.s).value
        ),
        "sim_exposure_s": float(simulator.time_manager.sim_exposure.to(u.s).value),
        "sim_readout_s": float(simulator.time_manager.sim_readout.to(u.s).value),
        "raw_frame_integration_s": float(raw_frame_integration_s),
        "raw_frame_sampling_interval_s": float(raw_frame_sampling_interval_s),
        "motion_split_hz": float(split_hz),
        "motion_split_reference": "raw_frame_integration_s",
        "n_jitter_frames_per_model": int(n_jitter_frames_per_model),
        "use_jitter_integrated_psf": bool(use_jitter_integrated_psf),
        "tess_psd_path": str(TESS_PSD_PATH),
        "tess_psd_cache_label": str(TESS_PSD_CACHE_LABEL),
        "enable_frame_to_frame_drift": bool(ENABLE_FRAME_TO_FRAME_DRIFT),
        "registered_motion_components": [
            str(component.get("component_name", ""))
            for component in dynamic_param_config["motion"]
        ],
        "enable_periodic_radial_expansion": bool(ENABLE_PERIODIC_RADIAL_EXPANSION),
        "radial_expansion_period_s": float(RADIAL_EXPANSION_PERIOD_S),
        "radial_expansion_active_last_s": float(RADIAL_EXPANSION_ACTIVE_LAST_S),
        "radial_expansion_arcsec": float(RADIAL_EXPANSION_ARCSEC),
        "radial_expansion_unit_source": str(_RADIAL_UNIT_SOURCE),
        "radial_expansion_unit_x": float(FIELD_CENTER_RADIAL_UNIT_X_DEG),
        "radial_expansion_unit_y": float(FIELD_CENTER_RADIAL_UNIT_Y_DEG),
    }
)

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(_meta, f, ensure_ascii=False, indent=2)

psf_model_manager = PSFModelManager(
    config=actor_config,
    warp_frame_batch_size=10,
    xy_jitter_pix=actor_config["xy_jitter_pix"],
    intialize=True,
    build_jit_int_models=use_jitter_integrated_psf,
)

psf_field_id = GUIDE_PSF_FIELD_ID  # 14 deg guide-detector PSF

psf_model = psf_model_manager.models[psf_field_id]
# star_data['id'] = star_id

# simulator.observatory = simulator.Observatory(simulator.sim_config, psf_model.compute_device, psf_model.float_precision)
simulator.observatory.store_variant_detectors(simulator.variant_manager)

# ============================================================================

# ====== 仅用两步法选星，彻底移除adaptive查询 ======
from photsim6.field import mk_real_field_stars_et_focalplane
from astropy import units as u
import numpy as np
import pandas as pd

pixel_scale = config_manager.parameters["Pixel Scale"]
n_pixels = int(config_manager.parameters["Detector Width"].to(u.pix).value)
target_ra = float(FIELD_CENTER_RA_DEG)
target_dec = float(FIELD_CENTER_DEC_DEG)

# 直接查 G<14 的所有星
star_data = mk_real_field_stars_et_focalplane(
    target_ra=target_ra * u.deg,
    target_dec=target_dec * u.deg,
    catalog_dir=GUIDE_GAIA_CATALOG_DIR,
    registry_data_dir=ET_FOCALPLANE_DATA_DIR,
    px_cnt=n_pixels,
    apply_offset=APPLY_STATIC_FIELD_OFFSET,
    mag_lim=14.0,
    plot=True,
    offset_x_pix=STATIC_FIELD_OFFSET_X_PIX,
    offset_y_pix=STATIC_FIELD_OFFSET_Y_PIX,
    target_epoch=2000.0,
    crop_to_simulation_frame=True,
    crop_margin_pix=0.0,
    et_focalplane_src=ET_FOCALPLANE_SRC_DIR,
)
star_data = enrich_star_data_truth_columns(star_data)

try:
    with open(meta_path, "r", encoding="utf-8") as f:
        _meta = json.load(f)
except Exception:
    _meta = {}

_meta.update(
    {
        "field_offset_x_pix": float(star_data.get("field_offset_x_pix", 0.0)),
        "field_offset_y_pix": float(star_data.get("field_offset_y_pix", 0.0)),
        "guide_query_target_center_xpix": float(
            star_data.get("target_center_xpix", np.nan)
        ),
        "guide_query_target_center_ypix": float(
            star_data.get("target_center_ypix", np.nan)
        ),
        "guide_query_target_center_xpix_shifted": float(
            star_data.get("target_center_xpix_shifted", np.nan)
        ),
        "guide_query_target_center_ypix_shifted": float(
            star_data.get("target_center_ypix_shifted", np.nan)
        ),
        # 已移除所有 guide_query_meta 相关内容，彻底清理遗留参数
    }
)

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(_meta, f, ensure_ascii=False, indent=2)

n_stars = len(star_data["x0"])
# Build Stars catalog
stars = Stars()
optical_eff_dpct = (
    simulator.sim_config["Optical Efficiency Ratio"].to(u.percent).value / 100
)
et_quantum_efficiency_dpct = 0.7
if "ET Quantum Efficiency" in simulator.sim_config:
    et_quantum_efficiency_dpct = (
        simulator.sim_config["ET Quantum Efficiency"].to(u.percent).value / 100
    )
stars.build_catalog(
    star_data,
    simulator.time_manager.sim_exposure,
    optical_eff_dpct,
    simulator.relative_aperture_area,
    mag_type=simulator.mag_type,
    et_quantum_efficiency=et_quantum_efficiency_dpct,
)

field_id = psf_field_id  # Which PSF ID to use? Keep catalog/ray generation consistent with the selected PSF.
star_id = 0  # Bogus/doesnt matterprint(np.sum(mask), column_sum)
transit_rel_flux = None
# Assign field ID and FOV angle
star_data["field_id"] = field_id
stars.catalog["Field ID"] = field_id
stars.catalog["Star ID"] = star_id

target_star_angle = float(FIELD_POLAR_ANGLE_RAD)
star_data["fov_theta"] = target_star_angle
stars.catalog["FOV theta"] = target_star_angle
if "source_id" in star_data:
    stars.catalog["Source ID"] = np.asarray(star_data["source_id"], dtype=np.int64)
if "gaia_g_mag" in star_data:
    stars.catalog["Gaia G Mag"] = np.asarray(star_data["gaia_g_mag"], dtype=float)
if "detector_xpix_shifted" in star_data:
    stars.catalog["Detector Xpix Shifted"] = np.asarray(
        star_data["detector_xpix_shifted"], dtype=float
    )
if "detector_ypix_shifted" in star_data:
    stars.catalog["Detector Ypix Shifted"] = np.asarray(
        star_data["detector_ypix_shifted"], dtype=float
    )
simulator.star_data = star_data

star_df = stars.catalog
print(n_stars)
# star_df


# =================== 两步分组选星逻辑（仅保留新逻辑） ===================
stars.catalog = star_df
if "Gaia G Mag" not in star_df.colnames:
    raise ValueError("星表缺少 'Gaia G Mag' 列，无法按 Gaia G 星等做筛选。")
mags = np.asarray(star_df["Gaia G Mag"], dtype=float)
n_available_catalog_stars = int(mags.size)
all_idx = np.arange(mags.size, dtype=int)
bright_mask = mags < GAIA_GMAG_LIM_MIN
bright_idx = all_idx[bright_mask]
faint_mask = mags >= GAIA_GMAG_LIM_MIN
faint_idx = all_idx[faint_mask]
keep = [0] if 0 in all_idx else []
bright_sorted = bright_idx[np.argsort(mags[bright_idx])]
bright_top = [idx for idx in bright_sorted if idx != 0][:GUIDE_TOP_BRIGHT]
faint_sorted = faint_idx[np.argsort(mags[faint_idx])]
faint_top = [idx for idx in faint_sorted if idx != 0][:GUIDE_TOP_FAINT]
for idx in bright_top + faint_top:
    if idx not in keep:
        keep.append(idx)
keep = np.array(keep, dtype=int)
star_df = star_df[keep]
star_data = _slice_star_data_by_indices(star_data, keep)
n_selected_catalog_stars = int(len(star_df))
selected_faintest_gaia_g_mag = (
    float(np.max(np.asarray(star_df["Gaia G Mag"], dtype=float)))
    if n_selected_catalog_stars > 0
    else None
)
selected_faintest_kepler_mag = None
if n_selected_catalog_stars > 0 and ("Kepler Mag" in star_df.colnames):
    try:
        selected_faintest_kepler_mag = float(
            np.max(np.asarray(star_df["Kepler Mag"], dtype=float))
        )
    except Exception:
        selected_faintest_kepler_mag = None
guide_mask_in_sim = np.zeros(len(star_df), dtype=bool)
for i, idx in enumerate(keep):
    if idx in faint_top or (idx == 0 and 0 in faint_top):
        guide_mask_in_sim[i] = True
star_df["Guide Candidate"] = guide_mask_in_sim
star_df["Guide Candidate Flag"] = guide_mask_in_sim.astype(np.int8)
stars.catalog = star_df
simulator.star_data = star_data
print(
    f"[Stars] Two-step selection: bright(G<{GAIA_GMAG_LIM_MIN}) top{GUIDE_TOP_BRIGHT} + faint(G>={GAIA_GMAG_LIM_MIN}) top{GUIDE_TOP_FAINT}; "
    f"total used for simulation = {len(star_df)}; "
    f"guide candidates (faint group) = {np.sum(guide_mask_in_sim)}; "
    f"brightest G = {np.min(mags[keep]):.3f}; faintest G = {np.max(mags[keep]):.3f}."
)
if 0 not in keep:
    print("[Stars] WARNING: target star(index=0)未被选入仿真星表！")
truth_index = np.arange(len(star_df), dtype=np.int64)
star_df["Truth Index"] = truth_index
star_df["Star ID"] = truth_index
star_df["FOV theta"] = np.full(len(star_df), target_star_angle, dtype=np.float64)
stars.catalog = star_df
guide_star_df = star_df[np.asarray(star_df["Guide Candidate"], dtype=bool)]
try:
    guide_star_df.write(opj(BATCH_DIR, "guide_candidates.ecsv"), overwrite=True)
except Exception:
    guide_star_df.to_pandas().to_csv(
        opj(BATCH_DIR, "guide_candidates.csv"), index=False
    )

try:
    with open(meta_path, "r", encoding="utf-8") as f:
        _meta = json.load(f)
except Exception:
    _meta = {}

_meta.update(
    {
        "n_truth_stars": int(len(star_df)),
        "field_polar_angle_rad": float(target_star_angle),
        "guide_catalog_min_star_target": int(GUIDE_MIN_CATALOG_STARS),
        "guide_catalog_available_star_count": int(n_available_catalog_stars),
        "guide_catalog_selected_star_count": int(n_selected_catalog_stars),
        "guide_catalog_selected_faintest_kepler_mag": selected_faintest_kepler_mag,
        "guide_catalog_selected_faintest_gaia_g_mag": selected_faintest_gaia_g_mag,
        "guide_catalog_meets_min_star_target": bool(
            n_available_catalog_stars >= int(GUIDE_MIN_CATALOG_STARS)
        ),
    }
)

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(_meta, f, ensure_ascii=False, indent=2)


def store_new_dynamic_params(self):

    self.data_generation_manager.store_variant_dynamic_parameters(
        self.time,
        self.n_raw_frames,
        self.dynamic_param_config,
        self.variant_manager,
        self.observatory,
        sim_config=self.sim_config,
        target_star_data=self.star_data,
        will_convole_jitter=self.convole_jitter,
        compute_device=self.compute_device,
        torch_dtype=self.torch_dtype,
    )


# 为保证跨不同天区中心时 motion/MD/thermal 等随机序列完全一致：
# 在生成动态参数前再次重置 RNG（星场生成的 random offset 不会影响到这里）。
if RESEED_BEFORE_DYNAMIC_PARAMS:
    set_global_seed(GLOBAL_SEED)

store_new_dynamic_params(simulator)

psf_model.build()

psf_model.is_built = True

"""
- 9 min run cost for 1k detfield_id with 6 scope and 1 variant and max_bkg_star_mag=15.0
- 9 min run cost for 1k det with 1 scope and 1 variant and max_bkg_star_mag=18.0

-----------

Part of the reason it takes so long is that the PSF of each star is evaluated across all subpixels of the entire image. This is 
highly inefficient since 99.999% of the star flux is within a 5 pixel radius circle of the centroid. Photsim was not designed to 
run in this way.
"""

# ---------------------------------------------------------------------------
# 输出：星表与图像（流式保存）
# ---------------------------------------------------------------------------

# 保存星表（本批天区中心），方便你后续做质心/星匹配/导航星筛选。
try:
    star_df.write(opj(BATCH_DIR, "stars.ecsv"), overwrite=True)
except Exception:
    # 少数环境可能没有 ECSV writer；兜底保存为 CSV。
    star_df.to_pandas().to_csv(opj(BATCH_DIR, "stars.csv"), index=False)

n_coadd_frames = int(actor_config["n_coadd_frames"])
batch_size = max(1, int(FRAME_BATCH_SIZE))

# coadd 帧时间轴（单位：秒）
cadence_s = float(simulator.time_manager.sim_frame_duration.to(u.s).value)

print(f"[Output] run_dir={RUN_DIR}")
print(f"[Output] batch_dir={BATCH_DIR}")
print(f"[Output] frames_dir={FRAMES_DIR}")
print(f"[Frames] n_coadd_frames={n_coadd_frames}  batch_size={batch_size}")

from photsim6.plot import plot_images

preview_saved = False

if STREAM_SAVE_FRAMES:

    for start in trange(0, n_coadd_frames, batch_size):
        stop = min(start + batch_size, n_coadd_frames)
        batch_range = range(start, stop)

        # 目前 Telescope Count 被固定为 1；这里写成循环以防你后续改回多镜筒。
        for scope_id, telescope in enumerate(simulator.observatory.telescopes):

            image_generator = ImageProcessing(
                config=actor_config,
                coadd_slice_range=batch_range,
                stars=star_df,
                telescope=telescope,
                psf_model=psf_model,
                variant_manager=simulator.variant_manager,
                cloning_info=simulator.image_gen_process_config["cloning_info"],
                groups_per_step=simulator.image_gen_process_config["groups_per_step"],
                transit_rel_flux=transit_rel_flux,
            )

            variant_data_slice = image_generator.process_all_steps()

            variant_ids = np.array(sorted(variant_data_slice.keys()), dtype=int)
            batch_images = torch.stack(
                [variant_data_slice[int(vid)] for vid in variant_ids.tolist()],
                dim=0,
            )
            truth_payload = image_generator.build_truth_payload(variant_ids)

            batch_time_s = (
                np.arange(start, stop, dtype=np.float64) * cadence_s
            ).astype(np.float64)

            out_npz = opj(
                FRAMES_DIR, f"scope{scope_id}_coadd_{start:06d}_{stop-1:06d}.npz"
            )
            np.savez_compressed(
                out_npz,
                images=batch_images.cpu().numpy().astype(np.float32),
                variant_ids=variant_ids,
                coadd_start=np.int64(start),
                coadd_stop=np.int64(stop),
                time_s=batch_time_s,
                cadence_s=np.float64(cadence_s),
                unit="electron_or_adu",  # photsim6 的单位链可能随配置而变，这里先不强行假定。
                **truth_payload,
            )

            # 保存一个预览 PNG（第一批次的第一帧、variant=0）
            if not preview_saved:
                try:
                    first_frame = batch_images[0, 0].cpu().numpy()
                    if first_frame.size == 0:
                        raise ValueError("empty preview frame")

                    fig, ax = plt.subplots(figsize=(10, 10))
                    show_frame = np.log10(np.clip(first_frame, 1e-6, None))
                    ax.imshow(show_frame, origin="lower", cmap="gray")
                    ax.set_title("Preview Frame (log10 scale)")
                    ax.set_xlabel("x [pix]")
                    ax.set_ylabel("y [pix]")
                    fig.tight_layout()
                    fig.savefig(
                        opj(
                            BATCH_DIR,
                            f"preview_batch{FIELD_CENTER_INDEX}_"
                            f"ra{FIELD_CENTER_RA_DEG:.4f}_dec{FIELD_CENTER_DEC_DEG:.4f}_"
                            f"frame{n_pixels}pix_FOV{FOV_DEG:.1f}deg_psf{field_id}.png",
                        )
                    )
                except Exception as exc:
                    print(f"[Preview] skipped preview image: {exc}")
                finally:
                    plt.close("all")
                    preview_saved = True

            # 主动释放引用，避免 Python/torch 缓存导致内存缓慢上涨
            del variant_data_slice
            del batch_images
            del image_generator

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

else:
    # 兼容：如果你真的想一次性生成（不建议 10k 帧），保留原始接口。
    images = generate_images_for_star(
        star_df,
        config=actor_config,
        psf_model=psf_model,
        variant_manager=simulator.variant_manager,
        observatory=simulator.observatory,
        image_cloning_info=simulator.image_gen_process_config["cloning_info"],
        image_groups_per_step=simulator.image_gen_process_config["groups_per_step"],
        transit_rel_flux=transit_rel_flux,
    )

    images_npz = opj(BATCH_DIR, "images_fullcube.npz")
    np.savez_compressed(images_npz, images=images.cpu().numpy())
