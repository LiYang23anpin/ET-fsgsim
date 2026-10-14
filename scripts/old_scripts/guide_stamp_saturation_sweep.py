#!/usr/bin/env python3
"""Generate 31x31 single-star guide stamps and check full-well saturation.

This script uses the Photsim7 PSF loading/stamp-rendering path, then applies the
requested detector terms for a single cadence:

stellar Poisson noise, sky-background Poisson noise, scattered-light Poisson
noise, dark-current Poisson noise, full-well clipping, readout noise, gain,
bias, and 12-bit ADC digitization.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from astropy import units as u
from astropy.table import QTable
from openpyxl import load_workbook


PHOTSIM7_ROOT = Path(os.environ.get("PHOTSIM7_ROOT", "/home/liyang/ET7/Photsim7"))
PHOTSIM7_SRC = PHOTSIM7_ROOT / "photsim7"
os.environ.setdefault("ET_DATA_DIR", "/home/liyang/ET7/Photsim7-data")


def ensure_photsim7_package_stub() -> None:
    """Import selected Photsim7 modules without executing photsim7/__init__.py."""

    if not PHOTSIM7_SRC.exists():
        raise FileNotFoundError(f"Cannot find Photsim7 source directory: {PHOTSIM7_SRC}")

    if str(PHOTSIM7_ROOT) not in sys.path:
        sys.path.insert(0, str(PHOTSIM7_ROOT))

    if "photsim7" in sys.modules:
        return

    pkg = types.ModuleType("photsim7")
    pkg.__path__ = [str(PHOTSIM7_SRC)]
    pkg.__package__ = "photsim7"
    sys.modules["photsim7"] = pkg


ensure_photsim7_package_stub()

from photsim7.background import sky_surface_brightness_to_background_flux
from photsim7.cosmic_rays import apply_adc_digitization, clip_full_well_electrons, electrons_to_adu
from photsim7.full_frame_renderer import SingleCadenceFullFrameRenderer
from photsim7.kepler_utils import et_mag_to_total_flux
from photsim7.psf.model import PSFModelManager


STAMP_SIZE_PIX = 31
SUBPIXELS_PER_PIXEL_DIM = 1
GUIDE_PIXEL_SCALE_ARCSEC_PER_PIX = 3.146
GUIDE_EXPOSURE_S = 0.25
GUIDE_PSF_BUNDLE_NAME = "241006/D280mm-focus-guide6p5um"
GUIDE_PSF_FIELD_ID = 7
BACKGROUND_SURFACE_BRIGHTNESS_MAG_ARCSEC2 = 22.0
TRANSIT_BACKGROUND_MODE = "surface_brightness"
TRANSIT_BACKGROUND_FLUX_E_PER_S_PER_PIX = np.nan
TRANSIT_SCATTERED_LIGHT_E_PER_S_PER_PIX = 2.11
TRANSIT_DARK_CURRENT_E_PER_S_PER_PIX = 0.7
MICROLENS_CONFIG_XLSX = "/home/liyang/ET7/Photsim7-data/et_inputs_microlens_guider.xlsx"
MICROLENS_PSF_BUNDLE_NAME = "microlens/RebinnedPSF_LE_BEST_GD1_Field1_multiband_Defocus0"
MICROLENS_FOCAL_LENGTH_MM = 5160.0

# The microlens guide script states that only dark+scattered was specified, and
# that splitting it is equivalent when both are independent Poisson processes.
DARK_PLUS_SCATTERED_E_PER_S_PER_PIX = 10.0
DARK_CURRENT_E_PER_S_PER_PIX = DARK_PLUS_SCATTERED_E_PER_S_PER_PIX / 2.0
SCATTERED_LIGHT_E_PER_S_PER_PIX = DARK_PLUS_SCATTERED_E_PER_S_PER_PIX / 2.0

READOUT_NOISE_E_PER_PIX = 5.0
FULL_WELL_ELECTRONS = 1800.0
ADC_BIT_DEPTH = 12
BIAS_LEVEL_ADU = 135.0
GAIN_ELECTRONS_PER_ADU = 0.45
QUANTUM_EFFICIENCY = 1.0
OPTICAL_THROUGHPUT = 1.0
SEED = 20260522


@dataclass(frozen=True)
class SweepConfig:
    instrument: str
    output_dir: str
    config_xlsx: str | None
    focal_length_mm: float | None
    stamp_size_pix: int
    subpixels_per_pixel_dim: int
    pixel_scale_arcsec_per_pix: float
    exposure_s: float
    psf_bundle_name: str
    psf_field_id: int
    background_mode: str
    background_surface_brightness_mag_arcsec2: float | None
    background_flux_e_per_s_per_pix: float
    scattered_light_e_per_s_per_pix: float
    dark_current_e_per_s_per_pix: float
    readout_noise_e_per_pix: float
    full_well_electrons: float
    adc_bit_depth: int
    bias_level_adu: float
    gain_electrons_per_adu: float
    xlsx_parameters: dict[str, object]


def _coerce_float(value: object, parameter: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Parameter {parameter!r} must be numeric, got {value!r}.") from exc


def _coerce_int(value: object, parameter: str) -> int:
    return int(round(_coerce_float(value, parameter)))


def load_general_inputs_xlsx(path: str | Path) -> dict[str, object]:
    workbook = load_workbook(path, data_only=True)
    if "General Inputs" not in workbook.sheetnames:
        raise ValueError(f"Cannot find 'General Inputs' sheet in {path}.")

    sheet = workbook["General Inputs"]
    parameters: dict[str, object] = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if len(row) < 3:
            continue
        parameter = row[1]
        value = row[2]
        if parameter is None:
            continue
        parameters[str(parameter)] = value
    return parameters


def resolve_psf_bundle_file(bundle_name: str) -> str:
    return str(Path(os.environ["ET_DATA_DIR"]) / "psf" / "et" / bundle_name / "sim_psf_images.pkl")


def build_sweep_config(args: argparse.Namespace) -> SweepConfig:
    if args.instrument == "transit":
        output_dir = args.output_dir or "/home/liyang/ET7/Results/FSG_guide_sims/single_star_saturation_sweep"
        return SweepConfig(
            instrument="transit_guide_detector",
            output_dir=output_dir,
            config_xlsx=None,
            focal_length_mm=None,
            stamp_size_pix=STAMP_SIZE_PIX,
            subpixels_per_pixel_dim=SUBPIXELS_PER_PIXEL_DIM,
            pixel_scale_arcsec_per_pix=GUIDE_PIXEL_SCALE_ARCSEC_PER_PIX,
            exposure_s=GUIDE_EXPOSURE_S,
            psf_bundle_name=GUIDE_PSF_BUNDLE_NAME,
            psf_field_id=GUIDE_PSF_FIELD_ID,
            background_mode=TRANSIT_BACKGROUND_MODE,
            background_surface_brightness_mag_arcsec2=BACKGROUND_SURFACE_BRIGHTNESS_MAG_ARCSEC2,
            background_flux_e_per_s_per_pix=TRANSIT_BACKGROUND_FLUX_E_PER_S_PER_PIX,
            scattered_light_e_per_s_per_pix=TRANSIT_SCATTERED_LIGHT_E_PER_S_PER_PIX,
            dark_current_e_per_s_per_pix=TRANSIT_DARK_CURRENT_E_PER_S_PER_PIX,
            readout_noise_e_per_pix=READOUT_NOISE_E_PER_PIX,
            full_well_electrons=FULL_WELL_ELECTRONS,
            adc_bit_depth=ADC_BIT_DEPTH,
            bias_level_adu=BIAS_LEVEL_ADU,
            gain_electrons_per_adu=GAIN_ELECTRONS_PER_ADU,
            xlsx_parameters={},
        )

    config_xlsx = args.config_xlsx or MICROLENS_CONFIG_XLSX
    xlsx_parameters = load_general_inputs_xlsx(config_xlsx)
    output_dir = args.output_dir or "/home/liyang/ET7/Results/FSG_guide_sims/microlens_single_star_saturation_sweep"
    background_mode = str(xlsx_parameters.get("Sky Background Mode", "electron_rate")).strip().lower()
    pixel_scale_arcsec_per_pix = _coerce_float(xlsx_parameters.get("Pixel Scale"), "Pixel Scale")
    background_surface_brightness = _coerce_float(
        xlsx_parameters.get("Sky Background Surface Brightness", 21.0),
        "Sky Background Surface Brightness",
    )
    if background_mode == "surface_brightness":
        background_flux = float(
            sky_surface_brightness_to_background_flux(
                background_surface_brightness,
                pixel_scale_arcsec_per_pix * u.arcsec / u.pix,
            ).to(u.electron / u.s / u.pix).value
        )
    else:
        background_mode = "electron_rate"
        background_flux = _coerce_float(xlsx_parameters.get("Background Flux"), "Background Flux")

    return SweepConfig(
        instrument="microlens_guide_detector",
        output_dir=output_dir,
        config_xlsx=str(config_xlsx),
        focal_length_mm=float(args.focal_length_mm),
        stamp_size_pix=STAMP_SIZE_PIX,
        subpixels_per_pixel_dim=_coerce_int(
            xlsx_parameters.get("Subpixels Per Pixel Dim", SUBPIXELS_PER_PIXEL_DIM),
            "Subpixels Per Pixel Dim",
        ),
        pixel_scale_arcsec_per_pix=pixel_scale_arcsec_per_pix,
        exposure_s=float(args.exposure_s),
        psf_bundle_name=str(xlsx_parameters.get("PSF Bundle Name") or MICROLENS_PSF_BUNDLE_NAME),
        psf_field_id=_coerce_int(xlsx_parameters.get("PSF Field ID", 0), "PSF Field ID"),
        background_mode=background_mode,
        background_surface_brightness_mag_arcsec2=background_surface_brightness,
        background_flux_e_per_s_per_pix=background_flux,
        scattered_light_e_per_s_per_pix=_coerce_float(xlsx_parameters.get("Scattered Light"), "Scattered Light"),
        dark_current_e_per_s_per_pix=_coerce_float(xlsx_parameters.get("Dark Current"), "Dark Current"),
        readout_noise_e_per_pix=_coerce_float(xlsx_parameters.get("Readout Noise"), "Readout Noise"),
        full_well_electrons=FULL_WELL_ELECTRONS,
        adc_bit_depth=_coerce_int(xlsx_parameters.get("ADC Bit Depth", ADC_BIT_DEPTH), "ADC Bit Depth"),
        bias_level_adu=BIAS_LEVEL_ADU,
        gain_electrons_per_adu=_coerce_float(
            xlsx_parameters.get("Gain Electrons Per ADU", GAIN_ELECTRONS_PER_ADU),
            "Gain Electrons Per ADU",
        ),
        xlsx_parameters=xlsx_parameters,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ET guide single-star stamps for 1800 e full-well saturation."
    )
    parser.add_argument(
        "--instrument",
        default="transit",
        choices=("transit", "microlens"),
        help="Guide detector configuration to use.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for .npy stamp products and CSV/JSON summary.",
    )
    parser.add_argument(
        "--config-xlsx",
        default=None,
        help="Microlens guider xlsx parameter file. Used only with --instrument microlens.",
    )
    parser.add_argument(
        "--focal-length-mm",
        type=float,
        default=MICROLENS_FOCAL_LENGTH_MM,
        help="Microlens guider focal length used for report metadata.",
    )
    parser.add_argument(
        "--exposure-s",
        type=float,
        default=GUIDE_EXPOSURE_S,
        help="Exposure duration in seconds. For this report the microlens guider uses 0.25 s.",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=500,
        help="Number of noisy realizations per magnitude for saturation probability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Base random seed.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used for the PSF render step.",
    )
    parser.add_argument(
        "--float-precision",
        type=int,
        default=32,
        choices=(32, 64),
        help="Floating-point precision passed to Photsim7 PSF models.",
    )
    parser.add_argument(
        "--quantum-efficiency",
        type=float,
        default=QUANTUM_EFFICIENCY,
        help="Multiplier applied to stellar and sky-background photo-electrons.",
    )
    parser.add_argument(
        "--optical-throughput",
        type=float,
        default=OPTICAL_THROUGHPUT,
        help="Multiplier applied to stellar and sky-background photo-electrons.",
    )
    return parser.parse_args()


def magnitude_grid() -> np.ndarray:
    return np.round(np.arange(14.0, 8.999, -0.5), 1)


def make_psf_model_manager(config_values: SweepConfig, device: str, float_precision: int) -> PSFModelManager:
    config = {
        "sim_config": {},
        "bundle_name": config_values.psf_bundle_name,
        "pixel_scale": config_values.pixel_scale_arcsec_per_pix * u.arcsec / u.pix,
        "n_pixels": config_values.stamp_size_pix,
        "n_rows": config_values.stamp_size_pix,
        "n_cols": config_values.stamp_size_pix,
        "detector_shape_pix": (config_values.stamp_size_pix, config_values.stamp_size_pix),
        "n_subpixels": config_values.subpixels_per_pixel_dim,
        "relative_aperture_area": 1.0,
        "integrate_jitter": False,
        "n_jitter_integrated_psf_models": 1,
        "n_jitter_frames": 1,
        "xy_jitter_pix": None,
        "compute_device": device,
        "float_precision": float_precision,
        "n_variants": 1,
    }
    psf_model_manager = PSFModelManager(
        config=config,
        warp_frame_batch_size=1,
        xy_jitter_pix=None,
        intialize=False,
        build_jit_int_models=True,
        field_ids=[config_values.psf_field_id],
        pad_to_detector_shape=False,
    )
    psf_model_manager.load_bundle_data()
    if len(np.atleast_1d(psf_model_manager.model_angles)) <= 1:
        psf_model_manager.model_probabilities = np.ones(len(psf_model_manager.model_images), dtype=float)
    else:
        psf_model_manager.measure_model_probabilitiy()
    psf_model_manager.build_all_models(build_jit_int_models=True)
    return psf_model_manager


def make_renderer(
    config_values: SweepConfig,
    psf_model_manager: PSFModelManager,
    photon_count_e: float,
    device: str,
    float_precision: int,
):
    center_pix = (config_values.stamp_size_pix - 1) / 2.0
    stars = QTable(
        {
            "Detector Xpix Shifted": [center_pix],
            "Detector Ypix Shifted": [center_pix],
            "Photon Count": [float(photon_count_e)] * u.electron,
            "Field ID": [config_values.psf_field_id],
        }
    )
    sim_config = {
        "Detector Width": config_values.stamp_size_pix * u.pix,
        "Detector Height": config_values.stamp_size_pix * u.pix,
        "Subpixels Per Pixel Dim": config_values.subpixels_per_pixel_dim,
    }
    return SingleCadenceFullFrameRenderer(
        sim_config=sim_config,
        stars=stars,
        psf_model_manager=psf_model_manager,
        frame_exposure=config_values.exposure_s * u.s,
        detector_response_sampler=None,
        compute_device=device,
        float_precision=float_precision,
    )


def render_expected_star(
    config_values: SweepConfig,
    psf_model_manager,
    mag: float,
    device: str,
    float_precision: int,
    photon_efficiency: float = 1.0,
) -> np.ndarray:
    photon_rate_e_per_s = float(et_mag_to_total_flux(float(mag)))
    photon_count_e = photon_rate_e_per_s * config_values.exposure_s * float(photon_efficiency)
    renderer = make_renderer(config_values, psf_model_manager, photon_count_e, device, float_precision)
    expected = renderer.render_expected_stellar_image(progress=False)
    if isinstance(expected, torch.Tensor):
        expected = expected.detach().cpu().numpy()
    return np.asarray(expected, dtype=np.float64)


def simulate_realizations(
    config_values: SweepConfig,
    expected_star_e: np.ndarray,
    rng: np.random.Generator,
    n_realizations: int,
    background_counts_e: float,
    scattered_counts_e: float,
    dark_counts_e: float,
) -> dict:
    expected_star_e = np.asarray(expected_star_e, dtype=np.float64)
    shape = (int(n_realizations),) + expected_star_e.shape

    star_e = rng.poisson(np.clip(expected_star_e, 0.0, None), size=shape).astype(np.float64)
    background_e = rng.poisson(background_counts_e, size=shape).astype(np.float64)
    scattered_e = rng.poisson(scattered_counts_e, size=shape).astype(np.float64)
    dark_e = rng.poisson(dark_counts_e, size=shape).astype(np.float64)

    charge_before_clip_e = star_e + background_e + scattered_e + dark_e
    saturated_mask = charge_before_clip_e >= config_values.full_well_electrons
    saturated_pixels_per_frame = saturated_mask.reshape(n_realizations, -1).sum(axis=1)

    clipped_charge_e = clip_full_well_electrons(
        charge_before_clip_e,
        full_well_electrons=config_values.full_well_electrons,
    )
    readout_noise_e = rng.normal(
        loc=0.0,
        scale=config_values.readout_noise_e_per_pix,
        size=shape,
    )
    after_readout_e = clipped_charge_e + readout_noise_e
    adu = electrons_to_adu(after_readout_e, gain_electrons_per_adu=config_values.gain_electrons_per_adu)
    adu = adu + config_values.bias_level_adu
    adc = apply_adc_digitization(
        adu,
        enabled=True,
        bit_depth=config_values.adc_bit_depth,
        min_value=0.0,
        round_values=True,
    )

    return {
        "charge_before_clip_e": charge_before_clip_e,
        "saturated_mask": saturated_mask,
        "saturated_pixels_per_frame": saturated_pixels_per_frame,
        "clipped_charge_e": clipped_charge_e,
        "after_readout_e": after_readout_e,
        "adu_float": adu,
        "adc": adc,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_values = build_sweep_config(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false.")
    if args.n_realizations <= 0:
        raise ValueError("--n-realizations must be positive.")
    if args.quantum_efficiency < 0.0:
        raise ValueError("--quantum-efficiency must be non-negative.")
    if args.optical_throughput < 0.0:
        raise ValueError("--optical-throughput must be non-negative.")

    photon_efficiency = float(args.quantum_efficiency) * float(args.optical_throughput)

    output_dir = Path(config_values.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if config_values.background_mode == "electron_rate":
        background_flux_e_per_s_per_pix = float(config_values.background_flux_e_per_s_per_pix)
    elif config_values.background_mode == "surface_brightness":
        background_flux = sky_surface_brightness_to_background_flux(
            config_values.background_surface_brightness_mag_arcsec2,
            config_values.pixel_scale_arcsec_per_pix * u.arcsec / u.pix,
        )
        background_flux_e_per_s_per_pix = float(background_flux.to(u.electron / u.s / u.pix).value)
    else:
        raise ValueError(f"Unsupported background mode: {config_values.background_mode!r}")
    background_counts_e = background_flux_e_per_s_per_pix * config_values.exposure_s * photon_efficiency
    scattered_counts_e = config_values.scattered_light_e_per_s_per_pix * config_values.exposure_s
    dark_counts_e = config_values.dark_current_e_per_s_per_pix * config_values.exposure_s

    psf_model_manager = make_psf_model_manager(config_values, args.device, args.float_precision)
    summary_rows: list[dict] = []
    representative_stamps = {}

    for mag in magnitude_grid():
        expected_star_e = render_expected_star(
            config_values,
            psf_model_manager,
            float(mag),
            args.device,
            args.float_precision,
            photon_efficiency=photon_efficiency,
        )
        sim = simulate_realizations(
            config_values=config_values,
            expected_star_e=expected_star_e,
            rng=rng,
            n_realizations=args.n_realizations,
            background_counts_e=background_counts_e,
            scattered_counts_e=scattered_counts_e,
            dark_counts_e=dark_counts_e,
        )
        max_before_clip = sim["charge_before_clip_e"].reshape(args.n_realizations, -1).max(axis=1)
        max_adc = sim["adc"].reshape(args.n_realizations, -1).max(axis=1)
        n_saturated_frames = int(np.count_nonzero(sim["saturated_pixels_per_frame"] > 0))

        representative_stamps[f"mag_{mag:.1f}_expected_star_e"] = expected_star_e.astype(np.float32)
        representative_stamps[f"mag_{mag:.1f}_charge_before_clip_e"] = sim[
            "charge_before_clip_e"
        ][0].astype(np.float32)
        representative_stamps[f"mag_{mag:.1f}_adc_adu"] = sim["adc"][0]

        summary_rows.append(
            {
                "mag": f"{mag:.1f}",
                "photon_efficiency": f"{photon_efficiency:.6f}",
                "base_photon_rate_e_per_s": f"{float(et_mag_to_total_flux(float(mag))):.6f}",
                "base_photon_count_e_0p25s": f"{float(et_mag_to_total_flux(float(mag))) * config_values.exposure_s:.6f}",
                "photon_rate_e_per_s": f"{float(et_mag_to_total_flux(float(mag))) * photon_efficiency:.6f}",
                "photon_count_e_0p25s": f"{float(et_mag_to_total_flux(float(mag))) * config_values.exposure_s * photon_efficiency:.6f}",
                "expected_peak_star_e": f"{float(np.max(expected_star_e)):.6f}",
                "expected_total_star_e": f"{float(np.sum(expected_star_e)):.6f}",
                "max_charge_e_sample0": f"{float(max_before_clip[0]):.6f}",
                "max_charge_e_median": f"{float(np.median(max_before_clip)):.6f}",
                "max_charge_e_p95": f"{float(np.percentile(max_before_clip, 95)):.6f}",
                "max_charge_e_max": f"{float(np.max(max_before_clip)):.6f}",
                "saturated_frames": n_saturated_frames,
                "saturation_fraction": f"{n_saturated_frames / args.n_realizations:.6f}",
                "max_saturated_pixels_in_frame": int(np.max(sim["saturated_pixels_per_frame"])),
                "max_adc_adu_sample0": int(max_adc[0]),
            }
        )

    write_csv(output_dir / "saturation_summary.csv", summary_rows)
    np.savez_compressed(output_dir / "single_star_stamps_31x31.npz", **representative_stamps)

    saturated = [row for row in summary_rows if float(row["saturation_fraction"]) > 0.0]
    first_saturated_mag = saturated[0]["mag"] if saturated else None
    always_saturated = [row for row in summary_rows if float(row["saturation_fraction"]) >= 1.0]
    always_saturated_mag = always_saturated[0]["mag"] if always_saturated else None

    metadata = {
        "instrument": config_values.instrument,
        "config_xlsx": config_values.config_xlsx,
        "stamp_size_pix": STAMP_SIZE_PIX,
        "subpixels_per_pixel_dim": config_values.subpixels_per_pixel_dim,
        "psf_bundle_name": config_values.psf_bundle_name,
        "psf_bundle_file": resolve_psf_bundle_file(config_values.psf_bundle_name),
        "psf_field_id": config_values.psf_field_id,
        "star_center_pix_zero_indexed": [(STAMP_SIZE_PIX - 1) / 2.0, (STAMP_SIZE_PIX - 1) / 2.0],
        "exposure_s": config_values.exposure_s,
        "focal_length_mm": config_values.focal_length_mm,
        "pixel_scale_arcsec_per_pix": config_values.pixel_scale_arcsec_per_pix,
        "background_mode": config_values.background_mode,
        "background_surface_brightness_mag_arcsec2": config_values.background_surface_brightness_mag_arcsec2,
        "quantum_efficiency": float(args.quantum_efficiency),
        "optical_throughput": float(args.optical_throughput),
        "photon_efficiency": float(photon_efficiency),
        "photon_efficiency_applied_to": ["stellar", "sky_background"],
        "background_flux_e_per_s_per_pix_before_efficiency": background_flux_e_per_s_per_pix,
        "background_flux_e_per_s_per_pix": background_flux_e_per_s_per_pix * photon_efficiency,
        "background_counts_e_per_pix": background_counts_e,
        "scattered_light_e_per_s_per_pix": config_values.scattered_light_e_per_s_per_pix,
        "scattered_counts_e_per_pix": scattered_counts_e,
        "dark_current_e_per_s_per_pix": config_values.dark_current_e_per_s_per_pix,
        "dark_counts_e_per_pix": dark_counts_e,
        "readout_noise_e_per_pix": config_values.readout_noise_e_per_pix,
        "full_well_electrons": config_values.full_well_electrons,
        "adc_bit_depth": config_values.adc_bit_depth,
        "bias_level_adu": config_values.bias_level_adu,
        "gain_electrons_per_adu": config_values.gain_electrons_per_adu,
        "xlsx_parameters_used_for_microlens": config_values.xlsx_parameters,
        "n_realizations_per_mag": args.n_realizations,
        "seed": args.seed,
        "first_mag_with_any_saturated_realization": first_saturated_mag,
        "first_mag_with_all_realizations_saturated": always_saturated_mag,
        "products": {
            "csv": str(output_dir / "saturation_summary.csv"),
            "stamps_npz": str(output_dir / "single_star_stamps_31x31.npz"),
        },
    }
    with (output_dir / "saturation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Output directory: {output_dir}")
    print(f"Instrument: {config_values.instrument}")
    print(f"PSF bundle file: {resolve_psf_bundle_file(config_values.psf_bundle_name)}")
    print(
        "Assumed dark+scattered = "
        f"{config_values.dark_current_e_per_s_per_pix + config_values.scattered_light_e_per_s_per_pix:.3f} e/s/pix "
        f"({config_values.dark_current_e_per_s_per_pix:.3f} dark + "
        f"{config_values.scattered_light_e_per_s_per_pix:.3f} scattered)."
    )
    print(
        "Background: "
        f"{background_flux_e_per_s_per_pix * photon_efficiency:.6f} e/s/pix after photon efficiency, "
        f"{background_counts_e:.6f} e/pix/frame."
    )
    print(
        "Photon efficiency: "
        f"QE={args.quantum_efficiency:.6f}, throughput={args.optical_throughput:.6f}, "
        f"product={photon_efficiency:.6f}."
    )
    print("mag  expected_peak_e  sat_fraction  max_charge_e_max  max_sat_pix")
    for row in summary_rows:
        print(
            f"{row['mag']:>4}  "
            f"{float(row['expected_peak_star_e']):15.3f}  "
            f"{float(row['saturation_fraction']):12.3f}  "
            f"{float(row['max_charge_e_max']):16.3f}  "
            f"{int(row['max_saturated_pixels_in_frame']):11d}"
        )
    if first_saturated_mag is None:
        print("Result: no saturated pixels in this magnitude grid.")
    else:
        print(f"Result: first grid magnitude with any saturated realization is mag={first_saturated_mag}.")
    if always_saturated_mag is not None:
        print(f"Result: all realizations saturated starting at mag={always_saturated_mag}.")


if __name__ == "__main__":
    main()