import os
import runpy
import sys
from pathlib import Path

print("启动参数：", sys.argv)
print("环境变量：", os.environ)

ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Non-terminal simulation inputs.
# Edit these values directly in this file when you want to run without shell envs.
# -----------------------------------------------------------------------------
FULL3_EFFECT_PROFILE = "full_with_jit"
#"full_no_jitter_no_drift" #"full"
FULL3_RUN_ALL_BATCHES = True
FULL3_GLOBAL_SEED = 12345
FULL3_ENABLE_FRAME_TO_FRAME_DRIFT = False
FULL3_ENABLE_FIELD_SPECIFIC_PSF = False # 用final_5.py的时候才开启
FULL3_TESS_PSD_PATH = "/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz.pkl"
#"/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz_within_1s_scaled_to_0p12arcsec_py311.pkl"
#"/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz_within_1s_scaled_to_0p36arcsec.pkl"
#"/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz.pkl"
#"/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz_within_1s_scaled_to_0p6arcsec.pkl"


def _set_env_from_local(name: str, value) -> None:
    os.environ.setdefault(name, str(value))


_set_env_from_local("ET_EFFECT_PROFILE", FULL3_EFFECT_PROFILE)
_set_env_from_local("ET_RUN_ALL_BATCHES", 1 if FULL3_RUN_ALL_BATCHES else 0)
_set_env_from_local("GLOBAL_SEED", FULL3_GLOBAL_SEED)
_set_env_from_local(
    "ET_ENABLE_FRAME_TO_FRAME_DRIFT",
    1 if FULL3_ENABLE_FRAME_TO_FRAME_DRIFT else 0,
)
_set_env_from_local(
    "ET_ENABLE_FIELD_SPECIFIC_PSF",
    1 if FULL3_ENABLE_FIELD_SPECIFIC_PSF else 0,
)
_set_env_from_local("ET_TESS_PSD_PATH", FULL3_TESS_PSD_PATH)

# os.environ.setdefault("ET_PSF_BUNDLE_NAME_OVERRIDE", "241006/D280mm-focus-guide6p5um")
# os.environ.setdefault("ET_OUTPUT_RUN_NAME_OVERRIDE", "guide_full_")
# os.environ.setdefault("ET_PROFILE_TARGET_FRAMES", "1")

runpy.run_path(str(ROOT / "et_sim_transit_guide_final_4.py"), run_name="__main__")


