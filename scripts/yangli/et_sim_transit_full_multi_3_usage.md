# et_sim_transit_full_multi_3.py 使用说明

## 0. 示例：
"ET_GUIDE_OPTICAL_EFFICIENCY=0.5 ET_GUIDE_QUANTUM_EFFICIENCY=0.7 python ./scripts/yangli/et_sim_transit_full_multi_3.py --indices 1 --parent-run-folder guide_60frm_Opt50_QE70_mag99"


## 1. 调用链

当前脚本的实际调用链是：

1. `et_sim_transit_full_multi_3.py`
2. `et_sim_transit_full_3.py`
3. `et_sim_transit_guide_final_4.py`

其中：

- `et_sim_transit_full_multi_3.py` 负责按 `--indices` 批量生成多个 run，并为每个 run 注入 `ET_OUTPUT_RUN_NAME_OVERRIDE` 和 `GLOBAL_SEED`。
- `et_sim_transit_full_3.py` 负责给下游补齐可直接编辑的本地变量，并映射成环境变量：`ET_EFFECT_PROFILE=full`、`ET_RUN_ALL_BATCHES=1`、`GLOBAL_SEED=12345`、`ET_ENABLE_FRAME_TO_FRAME_DRIFT=0`、`ET_TESS_PSD_PATH=...0p6arcsec.pkl`。
- `et_sim_transit_guide_final_4.py` 才是真正执行仿真、查询星表、写出图像和元数据的主脚本。

## 2. 最简洁启动命令

如果你已经在可用环境里（需要至少有 `matplotlib`、`astropy`、`torch`），在脚本目录下最简洁的启动命令是：

```bash
python et_sim_transit_full_multi_3.py
```

在当前机器上，实际可用且已验证通过的解释器是：

```bash
/home/liyang/anaconda3/envs/py311/bin/python et_sim_transit_full_multi_3.py
```

常用显式写法：

```bash
/home/liyang/anaconda3/envs/py311/bin/python et_sim_transit_full_multi_3.py \
  --indices 1-3 \
  --parent-run-folder guide_1min_10p5m_100star \
  --seed-base 12345
```

可调参数写法：
"""
ET_DATA_DIR=/home/liyang/ET7/Photsim7-data ET_PHOTSIM7_ROOT=/home/liyang/ET7/Photsim7 ET_PHOTSIM7_DATA_DIR=/home/liyang/ET7/Photsim7-data ET_OUTPUT_ROOT_OVERRIDE=/home/liyang/ET7/Results/Transit/Simulation ET_TESS_PSD_PATH=/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz.pkl ET_GUIDE_OPTICAL_EFFICIENCY=0.5 ET_GUIDE_QUANTUM_EFFICIENCY=0.65 python et_sim_transit_full_multi_3.py --indices 1 --parent-run-folder guide_100frm_qe045_QE65
"""

(python为/home/liyang/anaconda3/envs/py311/bin/python)
这条和你的意图一致的点是：
一定用 py311。
输出根路径明确还是 /home/liyang/ET7/Results/Transit/Simulation。
目录名就是 guide_100frm_qe045_QE65。
PSD 明确锁定为 /home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz.pkl。
QE 和光透率明确锁定为 0.65 和 0.45。
另外，当前脚本里 et_sim_transit_guide_final_3.py:167 是 25 秒，在 0.25 秒 cadence 下默认就是 100 帧，所以你的 guide_100frm 命名现在是匹配的，不需要再额外加 ET_PROFILE_TARGET_FRAMES=100。




## 3. 命令行参数

`et_sim_transit_full_multi_3.py` 只暴露这 3 个命令行参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--indices` | `"1"` | 运行序号，支持 `2-12`、`2,5,9`、`2-5,9,11-12` 这样的写法 |
| `--parent-run-folder` | `guide_1min_10p5m_100star` | 输出根目录下的父目录名 |
| `--seed-base` | `12345` | 每个 run 的基础随机种子，实际 run 种子为 `seed_base + index` |

注意：脚本 help 文案里把 `--indices` 说明写成了“默认 2-99”，但代码实际默认值是 `1`。

## 4. 所有可从命令行注入的环境变量

下面按调用层级列出这条链路里所有实际会被读取、且可以在 shell 前缀中传入的环境变量。

### 4.1 顶层包装脚本：et_sim_transit_full_multi_3.py

这个脚本本身几乎不读环境变量，但会主动设置或覆盖以下变量：

| 变量名 | 默认/实际值 | 作用 | 备注 |
| --- | --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | 强制写死为 `1` | 绑定 GPU 设备 | 在这个包装脚本里是硬覆盖，外部同名环境变量会被它改写 |
| `ET_OUTPUT_RUN_NAME_OVERRIDE` | 每个 run 自动设为 `<parent-run-folder>/transit_full_<idx>` | 传给下游作为输出 run 名 | 由脚本内部注入，外部预设会被覆盖 |
| `GLOBAL_SEED` | 每个 run 自动设为 `seed_base + index` | 传给下游作为随机种子 | 由脚本内部注入，外部预设会被覆盖 |

### 4.2 中间层：et_sim_transit_full_3.py

这个脚本通过 `os.environ.setdefault(...)` 给下游提供默认值；如果你在 shell 里先传入同名变量，则会保留你的值。

| 变量名 | 默认值 | 作用 |
| --- | --- | --- |
| `FULL3_EFFECT_PROFILE` | `full` | 选择 effect profile，直接改这个 Python 变量即可 |
| `FULL3_RUN_ALL_BATCHES` | `True` | 是否自动顺序跑完 4 个 `FIELD_CENTERS_DEG` 批次 |
| `FULL3_GLOBAL_SEED` | `12345` | 全局随机种子 |
| `FULL3_ENABLE_FRAME_TO_FRAME_DRIFT` | `False` | 是否启用帧间 drift；`False` 时只保留帧内 jitter |
| `FULL3_TESS_PSD_PATH` | `/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz_within_1s_scaled_to_0p6arcsec.pkl` | 帧内 jitter 用的 PSD 文件 |

说明：`et_sim_transit_full_3.py` 会把这些本地变量映射成下游环境变量，因此你可以只改这个文件，不必每次在终端前面手敲一长串 env。

### 4.3 主仿真脚本：et_sim_transit_guide_final_4.py

#### 4.3.1 运行环境与路径

| 变量名 | 默认值 | 作用 |
| --- | --- | --- |
| `ET_PHOTSIM7_ROOT` | 若未传，则尝试 `/home/liyang/ET7/Photsim7` | Photsim7 源码根目录 |
| `ET_PHOTSIM7_DATA_DIR` | 若未传，则尝试 `/home/liyang/ET7/Photsim7-data` | Photsim7 数据目录 |
| `ET_CONFIG_XLSX_PATH` | 若未传，则使用 `/home/liyang/ET7/ET-fsgsim/config/et_inputs_transit_guider.xlsx` | 配置表路径 |
| `ET_PSF_BUNDLE_NAME_OVERRIDE` | 默认不设 | 覆盖 PSF bundle 名称 |
| `PSF_BUNDLE_NAME_OVERRIDE` | 默认不设 | `ET_PSF_BUNDLE_NAME_OVERRIDE` 的旧别名 |

补充说明：脚本会把 `ET_PHOTSIM7_DATA_DIR` 解析后的结果写入内部环境变量 `ET_DATA_DIR`，但这不是当前调用链里一个稳定的对外输入接口，因为它会在脚本启动后被覆盖。

#### 4.3.2 批处理与随机性

| 变量名 | 默认值 | 作用 |
| --- | --- | --- |
| `ET_RUN_ALL_BATCHES` | 脚本内部缺省逻辑是 `True`；通过当前调用链时，上游默认把它设成 `1` | 是否在父进程里顺序启动全部 batch |
| `ET_CHILD_BATCH_RUN` | `False` | 内部子进程标记，正常手工启动通常不用设置 |
| `ET_FIELD_CENTER_INDEX` | `0` | 选择 `FIELD_CENTERS_DEG` 中的 batch 索引 |
| `GLOBAL_SEED` | `12345` | 全局随机种子；在顶层包装脚本调用时会被自动改写为 `seed_base + index` |

#### 4.3.3 仿真规模与输出

| 变量名 | 默认值 | 作用 |
| --- | --- | --- |
| `ET_EFFECT_PROFILE` | 直接运行 `et_sim_transit_guide_final_4.py` 时默认解析为 `v2_point_drift_jitter`；通过当前调用链时，上游默认把它设成 `full` | 选择 effect profile，可用值含 `full`、`v0_onlypsf`、`v1_noise_psf`、`v2_point_drift_jitter`、`v3_dva`、`v4_thermal`、`v5_prv_subpixel` 及其别名 |
| `ET_PROFILE_TARGET_FRAMES` | `None` | 目标帧数上限；常用于冒烟测试或缩短运行时间 |
| `ET_OUTPUT_ROOT_OVERRIDE` | `/home/liyang/ET7/Results/Transit/Simulation/` | 输出根目录 |
| `ET_OUTPUT_RUN_NAME_OVERRIDE` | `guide_full_10p5mag` | 输出 run 名；通过当前包装脚本调用时会被自动改写 |
| `ET_ENABLE_FRAME_TO_FRAME_DRIFT` | `0`（由 `FULL3_ENABLE_FRAME_TO_FRAME_DRIFT=False` 映射） | 是否保留帧间 drift |
| `ET_TESS_PSD_PATH` | `/home/liyang/ET7/Photsim7-data/pds/ET_psd3-2_extended_to_10Hz_within_1s_scaled_to_0p6arcsec.pkl` | 帧内 jitter 的 PSD 输入 |

#### 4.5.4 仿真参数

| 变量名 | 默认值 | 作用 |
| --- | --- | --- |
| `ET_MAX_SIM_STARS` | `None` | 限制参与仿真的星数上限，主要用于控时/控显存 |
| `ET_GAIA_GMAG_LIM_MIN` | `11` | 亮/暗星分组阈值，控制导星候选星选择 |
| 'ET_GUIDE_OPTICAL_EFFICIENCY' | 0.5 | 光学透过率 |
| "ET_GUIDE_QUANTUM_EFFICIENCY" | 0.7 | 量子效率 |

#### 4.3.4 周期性径向外扩相关变量

| 变量名 | 默认值 | 作用 |
| --- | --- | --- |
| `ET_ENABLE_PERIODIC_RADIAL_EXPANSION` | `False` | 周期性径向外扩总开关 |
| `ET_RADIAL_EXPANSION_PERIOD_S` | `10.0` | 一个完整周期时长，单位秒 |
| `ET_RADIAL_EXPANSION_ACTIVE_LAST_S` | `3.0` | 每个周期末尾激活时长，单位秒 |
| `ET_RADIAL_EXPANSION_ARCSEC` | `0.0054` | 外扩幅度，单位角秒 |

示例：

```bash
ET_ENABLE_PERIODIC_RADIAL_EXPANSION=1 \
ET_RADIAL_EXPANSION_PERIOD_S=10 \
ET_RADIAL_EXPANSION_ACTIVE_LAST_S=3 \
ET_RADIAL_EXPANSION_ARCSEC=0.0054 \
/home/liyang/anaconda3/envs/py311/bin/python et_sim_transit_full_multi_3.py
```

## 5. 当前版本中“相关但不能靠环境变量覆盖”的硬编码项

这些项和运行结果强相关，但在当前这条脚本链中不是通过环境变量对外暴露的：

| 项目 | 当前值 | 说明 |
| --- | --- | --- |
| `GUIDE_GAIA_CATALOG_DIR` | `/home/liyang/ETguider/gaia_dr3_19mag` | 星表目录，当前版本写死在主仿真脚本里 |
| `ET_FOCALPLANE_ROOT` | `/home/liyang/ET7/et_focalplane` | 焦平面代码根目录，当前版本写死 |
| `ET_FOCALPLANE_DATA_DIR` | `/home/liyang/ET7/et_focalplane/data` | 焦平面数据目录，随上项推导 |
| `ET_FOCALPLANE_SRC_DIR` | `/home/liyang/ET7/et_focalplane/src` | 焦平面源码目录，随上项推导 |
| `GUIDE_OBSERVING_DURATION_S` | `15` | 代码常量；若设置了 `ET_PROFILE_TARGET_FRAMES`，最终有效输出帧数仍可被缩短 |
| `PSF_BUNDLE_NAME_DEFAULT` | `241006/D280mm-focus-guide6p5um` | 未显式覆盖时的默认 PSF bundle |

## 6. 推荐的冒烟测试命令

为了尽量便宜地验证整条链路，建议把 batch 数、帧数和星数都压小：

```bash
cd /home/liyang/ET7/ET-fsgsim/scripts/yangli

ET_RUN_ALL_BATCHES=0 \
ET_PROFILE_TARGET_FRAMES=1 \
ET_MAX_SIM_STARS=5 \
ET_OUTPUT_ROOT_OVERRIDE=/tmp/et_transit_smoke \
/home/liyang/anaconda3/envs/py311/bin/python et_sim_transit_full_multi_3.py \
  --indices 1 \
  --parent-run-folder smoke_parent \
  --seed-base 12345
```

这条命令会让当前包装脚本只生成一个 run，下游只跑单个 batch，且只生成 1 帧，适合做链路连通性检查。

## 7. 2026-06-08 冒烟测试结果

已在当前机器上成功执行以下命令：

```bash
ET_RUN_ALL_BATCHES=0 \
ET_PROFILE_TARGET_FRAMES=1 \
ET_MAX_SIM_STARS=5 \
ET_OUTPUT_ROOT_OVERRIDE=/tmp/et_transit_smoke \
ET_OUTPUT_RUN_NAME_OVERRIDE=smoke_transit_full_multi3 \
/home/liyang/anaconda3/envs/py311/bin/python et_sim_transit_full_multi_3.py \
  --indices 1 \
  --parent-run-folder smoke_parent \
  --seed-base 12345
```

实际结果：

- 成功进入 `et_sim_transit_full_3.py` 和 `et_sim_transit_guide_final_3.py`
- 成功生成 1 个 batch、1 帧 coadd 输出
- 产物位于 `/tmp/et_transit_smoke/smoke_parent/transit_full_01/`

确认存在的关键产物包括：

- `batch0_ra278.1844_dec37.5704/frames/scope0_coadd_000000_000000.npz`
- `batch0_ra278.1844_dec37.5704/run_meta.json`
- `batch0_ra278.1844_dec37.5704/stars.ecsv`
- `batch0_ra278.1844_dec37.5704/guide_candidates.ecsv`
- `cache/photsim7_config/et_inputs_transit_guider_photsim7_units.xlsx`

当前机器上应优先使用 `/home/liyang/anaconda3/envs/py311/bin/python`。