# Guider Script QE and Optical Efficiency Update

## Summary

- Added script-level defaults for the transit guider simulation:
  - optical throughput fallback: `0.5`
  - ET quantum efficiency default: `0.7`
- The script now preserves the xlsx `Optical Efficiency Ratio` when that parameter is present in the input sheet.
- The script only falls back to `0.5` when the xlsx input omits `Optical Efficiency Ratio`.

## Script Inputs

- xlsx source:
  - `config/et_inputs_transit_guider.xlsx`
- new environment overrides:
  - `ET_GUIDE_OPTICAL_EFFICIENCY`
  - `ET_GUIDE_QUANTUM_EFFICIENCY`

## Runtime Behavior

- `Optical Efficiency Ratio` is kept from the xlsx whenever it exists.
- If the xlsx omits that field, the script injects a fallback of `0.5`.
- `ET Quantum Efficiency` is always set in the runtime config and defaults to `0.7`.
- Both values are written into `run_meta.json` for later audit.

## Notes

- This repository currently sees the guider script as a new file under `scripts/yangli/`.
- As a result, the corresponding PR will add the full script file rather than a small tracked-file diff.