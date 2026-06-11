# JI-PSF OOM analysis and mitigation notes

## Scope

This note summarizes the observed CUDA OOM related to `xy_jitter_pix`, `N Jitter-Integrated PSF Models`, and `N Jitter Frames Per Model` in transit guider simulation. The goal is to preserve a clear decision record for later selective merge into main.

## Symptom

Representative traceback:

- `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.69 GiB`
- Error location: `Photsim7/photsim7/psf/model.py`, inside `PSFModel.generate_images()` during JI-PSF build.

## Parameters and meanings

- `xy_jitter_pix`: exposure-internal high-frequency jitter trajectory in pixel units, shape approximately `(n_models, 2, n_frames_per_model)`.
- `N Jitter-Integrated PSF Models`: number of prebuilt jitter-blurred PSF models (`n_models`).
- `N Jitter Frames Per Model`: sub-samples per exposure used to average one jitter-blurred PSF (`n_frames_per_model`).

Total jitter samples scale as:

- `total_samples = n_models * n_frames_per_model`

Memory-critical tensor in original build path scales mainly with:

- `n_frames_per_model * H * W`

where `H=W=2049` for current guider frame size.

## Why OOM happened

In the original JI-PSF path, for each jitter model, the code allocates all subframe warps at once and then averages:

- generate all `n_frames_per_model` warped PSF frames
- `torch.clamp(...)`
- normalize and `torch.mean(...)`

With float32 and `2049x2049`:

- one full `images` tensor at `n_frames_per_model=300` is about `300 * 2049 * 2049 * 4 bytes ~= 4.69 GiB`

This matches the OOM message exactly.

## PSD comparison findings

Compared files:

- old PSD: `/home/liyang/Photosim6/data/tess_data/fPSD_extended_to_10Hz_as_tess_xyz_psd.pkl`
- new PSD: `/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz_amp12x.pkl`

Observed:

- Frequency sampling count is the same in both files (per axis): `2,592,005`.
- Frequency range is the same: from about `3.858e-6 Hz` to `10 Hz`.
- `f > 4 Hz` point count is also the same.

Conclusion:

- OOM was not triggered by PSD sampling count increase.
- OOM is dominated by JI-PSF build tensor size and current parameter scale (`100/300`) at `2049x2049`.

## Historical near-zero behavior

Previously, near-zero jitter was treated with tolerance check using `np.isclose(..., atol=1e-4)` logic in Photsim7 jitter gating paths. That could skip JI-PSF build for very small jitter amplitudes.

## Code locations involved

Inside this repository:

- `scripts/yangli/et_sim_transit_guide_final_3.py`
  - reads config and allows env overrides:
    - `ET_N_JITTER_INTEGRATED_PSF_MODELS`
    - `ET_N_JITTER_FRAMES_PER_MODEL`
  - builds `xy_jitter_pix`
  - passes jitter data into `PSFModelManager`

Outside this repository (local sibling path):

- `/home/liyang/ET7/Photsim7/photsim7/psf/model.py`
  - `PSFModel.build()`
  - `PSFModel.generate_images()`

Note: the Photsim7 file above is not tracked in this repo and is therefore not part of this PR.

## Practical mitigation choices

1. Keep physical JI-PSF and reduce scale for routine runs:

- suggested start: `20 / 100`
- safer fallback: `20 / 64` or `10 / 64`

2. Keep high fidelity but optimize implementation in Photsim7:

- chunked accumulation in JI-PSF build to reduce peak memory.
- this is a code-level optimization in Photsim7 repo/path, not in ET-fsgsim.

3. Reintroduce configurable near-zero threshold (if scientifically acceptable):

- skip JI-PSF only when `max(|xy_jitter_pix|) <= threshold_pix`.
- threshold should be explicitly documented and configurable.

## Suggested run-time overrides

Examples for this repo script:

```bash
ET_N_JITTER_INTEGRATED_PSF_MODELS=20 \
ET_N_JITTER_FRAMES_PER_MODEL=100 \
python ./scripts/yangli/et_sim_transit_full_multi_3.py \
  --indices 0 --parent-run-folder guide_100frame_et7_psd12x --seed-base 12345
```

If GPU pressure remains high:

```bash
ET_N_JITTER_INTEGRATED_PSF_MODELS=20 \
ET_N_JITTER_FRAMES_PER_MODEL=64 \
python ./scripts/yangli/et_sim_transit_full_multi_3.py \
  --indices 0 --parent-run-folder guide_100frame_et7_psd12x --seed-base 12345
```

## Decision record for selective merge

- This PR intentionally contains documentation only.
- No source code behavior changes are included here.
- Future code PRs can selectively implement one mitigation path above.
