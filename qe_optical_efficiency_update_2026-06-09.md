- 已完成两部分本地修改，并把包侧改动发成了线上 PR。

- Photsim7 这边已经改好并开出 PR：

ET 星等链路不再在 field.py 里写死 QE。
ET 模式现在会同时应用 Optical Efficiency Ratio 和新的 ET Quantum Efficiency 配置，配置入口补在了 configurator.py、simulator.py、ray_cluster.py。
回归测试更新在 test_photsim6_merge_features.py。
本地可读修改记录在 et_qe_optical_efficiency_update_2026-06-09.md。
已创建 PR：https://github.com/TutuchanXD/Photsim7/pull/35

- 运行脚本这边也已经本地改好：

et_sim_transit_guide_final_3.py 现在会优先保留 xlsx 里的 Optical Efficiency Ratio。
只有当 xlsx 缺少 Optical Efficiency Ratio 时，才回退到默认值 0.5。
ET Quantum Efficiency 默认值设为 0.7，并支持环境变量覆盖。
两个值都会写进 run_meta.json。
本地可读修改记录在本文件qe_optical_efficiency_update_2026-06-09.md。


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