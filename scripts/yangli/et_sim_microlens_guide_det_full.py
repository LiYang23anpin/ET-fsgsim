import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ET_FSGSIM_ROOT = ROOT.parent.parent
CLUSTER_ET7_ROOT = ET_FSGSIM_ROOT.parent

MICROLENS_GUIDE_OPTICAL_EFFICIENCY = 0.52
MICROLENS_GUIDE_QUANTUM_EFFICIENCY = 0.5
MICROLENS_SCRIPT_NAME = "et_sim_microlens_guide_det_v1_noise_psf_ly5.py"


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

os.environ.setdefault("PHOTSIM7_ROOT", str(CLUSTER_ET7_ROOT / "Photsim7"))
os.environ.setdefault("ET_DATA_DIR", str(CLUSTER_ET7_ROOT / "Photsim7-data"))
os.environ.setdefault(
	"ET_CONFIG_XLSX", str(ET_FSGSIM_ROOT / "config" / "et_inputs_microlens_guider.xlsx")
)
os.environ.setdefault("ET_FOCALPLANE_ROOT", str(CLUSTER_ET7_ROOT / "et_focalplane"))
os.environ.setdefault(
	"GUIDE_GAIA_CATALOG_DIR",
	first_complete_gaia_catalog_dir(
		CLUSTER_ET7_ROOT / "gaia_dr3_19mag",
		Path.home() / "ET7" / "gaia_dr3_19mag",
		Path.home() / "ETguider" / "gaia_dr3_19mag",
	),
)
os.environ.setdefault(
	"ET_OUTPUT_ROOT_OVERRIDE",
	str(CLUSTER_ET7_ROOT / "Results" / "FSG_guide_sims" / "Micro_simulation"),
)
os.environ.setdefault(
	"ET_MICROLENS_JITTER_PSD_PATH",
	"/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz_within_1s_scaled_to_0p12arcsec_py311.pkl",
)
os.environ.setdefault("ET_ENABLE_FRAME_TO_FRAME_DRIFT", "0")
os.environ.setdefault("ET_GUIDE_OPTICAL_EFFICIENCY", str(MICROLENS_GUIDE_OPTICAL_EFFICIENCY))
os.environ.setdefault("ET_GUIDE_QUANTUM_EFFICIENCY", str(MICROLENS_GUIDE_QUANTUM_EFFICIENCY))

os.environ.setdefault("ET_EFFECT_PROFILE", "full_with_jit")
os.environ.setdefault("ET_RUN_ALL_BATCHES", "1")
os.environ.setdefault("ET_PROFILE_TARGET_FRAMES", "50")
# os.environ.setdefault("ET_OUTPUT_RUN_NAME_OVERRIDE", "microlens_guide_photsim7_100f")

runpy.run_path(str(ROOT / MICROLENS_SCRIPT_NAME), run_name="__main__")
