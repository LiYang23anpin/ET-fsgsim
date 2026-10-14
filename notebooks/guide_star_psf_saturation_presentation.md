# ET 导星 PSF 与单星饱和分析汇报稿

## 1. 研究目标

本报告用于汇报 ET 导星焦平面在 14 度视场角处的 PSF 形态，以及基于该 PSF 的单星导星 stamp 满阱饱和边界。

本次整理覆盖两部分内容：

- PSF 形态对比：聚焦、离焦 -0.03 mm、离焦 +0.03 mm，均取 field ID = 7，对应 14 度视场角。
- 单星饱和评估：使用 31x31 导星 stamp，在加入背景、杂散光、暗电流、光子噪声、读出噪声、满阱裁剪、偏置和 12 bit ADC 后，评估不同星等是否出现饱和像素。

## 2. 数据来源与参数设置

### 2.1 PSF 数据

- 聚焦 PSF：241006/D280mm-focus-guide6p5um
- 离焦 PSF：241006/D280mm-defocus(-0.03mm)-guide6p5um、241006/D280mm-defocus(+0.03mm)-guide6p5um
- 使用视场：field ID = 7，对应 14.0 deg FoV
- 像元物理尺寸：6.5 um
- 本次汇报图使用 1x1 子像元网格，而不是最高过采样网格

### 2.2 单星饱和仿真参数

| 参数 | 数值 |
| --- | --- |
| Stamp size | 31 x 31 pix |
| Subpixels per pixel dim | 1 |
| PSF bundle | 241006/D280mm-focus-guide6p5um |
| PSF field ID | 7 |
| Exposure | 0.25 s |
| Pixel scale | 3.146 arcsec/pix |
| Sky background brightness | 22.0 mag/arcsec^2 |
| Background count per frame | 4.352917 e/pix |
| Scattered light | 2.110 e/s/pix |
| Dark current | 0.700 e/s/pix |
| Readout noise | 5.000 e/pix |
| Full well | 1800 e |
| ADC | 12 bit |
| Bias | 135 ADU |
| Gain | 0.45 e/ADU |
| Realizations per mag | 500 |

## 3. 分析过程

### 3.1 PSF 形态分析流程

1. 从三组 sim_psf_images.pkl 中读取 field ID = 7 的 PSF 数据。
2. 显式选用 1x1 子像元网格，即 key = 1 的 PSF，而不是默认取最大采样率。
3. 对归一化 PSF 计算包围能量曲线 Encircled Energy。
4. 通过累计光度达到 90% 时对应的半径，定义 EE90 半径。
5. 在图像上叠加 EE90 圆，并在累计光度曲线上标出 EE90 位置。

### 3.2 单星饱和评估流程

1. 使用聚焦导星 PSF bundle 的 field ID = 7，在 31x31 局部 stamp 上渲染单星图像。
2. 叠加天空背景、杂散光、暗电流，并对各项电子计数进行泊松采样。
3. 以满阱前电荷 $Q \ge 1800\,e$ 作为像素饱和判据。
4. 之后再执行满阱裁剪、读出噪声、gain 转 ADU、加 bias 和 12 bit ADC 截断。
5. 对每个星等运行 500 次噪声 realization，统计至少出现 1 个饱和像素的帧比例。
6. 在饱和边界附近进一步做 0.1 mag 细步长扫描，精细定位阈值。

## 4. 结果展示

### 4.1 14 度视场角 PSF 对比

下图给出 14 度视场角、field ID = 7、1x1 子像元网格下的 PSF 图像与累计光度曲线。左列为 PSF 图像，右列为 Encircled Energy 曲线，均标出了 EE90 半径。

![](../../../Photosim6/data/psf/et/241006/field7_fov14deg_psf_ee90.png)

对应 EE90 结果如下：

| 工况 | EE90 [pix] | EE90 [um] |
| --- | ---: | ---: |
| 聚焦 | 5.831 | 37.90 |
| 离焦 -0.03 mm | 7.000 | 45.50 |
| 离焦 +0.03 mm | 5.657 | 36.77 |

从汇报角度可以直接指出：

- 14 度视场角下，离焦会明显改变能量集中程度。
- -0.03 mm 离焦的 EE90 最大，说明能量分布更扩散。
- 在本组模型下，+0.03 mm 离焦的 EE90 略小于聚焦值，表明离焦响应具有方向性不对称。

### 4.2 星等与峰值电荷关系

下图展示无噪声 PSF 最亮像素与加噪后最亮像素的变化。红色虚线为满阱 1800 e。

![](guide_star_saturation_report_files/peak_charge_vs_mag.png)

图中可见：

- 11.0 等时，最亮像素约为 1046 e，明显低于满阱。
- 10.5 等时，期望最亮像素约为 1658 e，仍低于 1800 e。
- 10.0 等时，期望最亮像素约为 2628 e，已明显超过满阱。

### 4.3 粗步长饱和概率

下图给出 0.5 mag 星等网格上，每个星等 500 次噪声 realization 的饱和比例。

![](guide_star_saturation_report_files/saturation_fraction_vs_mag.png)

粗步长结论很清楚：

- 11.0 等及更暗：500 次 realization 中均未出现饱和。
- 10.5 等：500 次 realization 中未出现饱和。
- 10.0 等及更亮：500 次 realization 中全部出现饱和。

因此仅从 0.5 mag 扫描看，饱和边界位于 10.5 等与 10.0 等之间。

### 4.4 细步长边界定位

为了精细定位边界，在 10.4 至 11.2 等之间进行了 0.1 mag 细扫。下图同时给出期望最亮像素电子数与饱和比例。

![](guide_star_saturation_report_files/fine_magnitude_peak_e_and_saturation.png)

细扫得到的关键数字如下：

| 星等 | 期望最亮像素 [e] | 饱和比例 | 饱和帧数 / 500 |
| --- | ---: | ---: | ---: |
| 10.5 | 1658.22 | 0.000 | 0 |
| 10.4 | 1818.20 | 0.726 | 363 |
| 10.3 | 1993.62 | 1.000 | 500 |

从图和表可以得出：

- 连续阈值条件 expected peak = 1800 e 对应的星等约为 10.411 mag。
- 10.5 等仍处于安全侧。
- 10.4 等已经进入明显风险区，约 72.6% 的 realization 出现饱和。
- 10.3 等及更亮已经进入稳定饱和区。

### 4.5 代表性图像示意

下图展示 11.0、10.5、10.0、9.5 等下的代表性 31x31 stamp，包括：

- 无噪声 PSF 电子图
- 加噪后满阱裁剪前电荷图
- ADC 输出图像

![](guide_star_saturation_report_files/representative_stamps.png)

图中可以直观看到：

- 11.0 等时星像峰值仍明显低于满阱。
- 10.5 等时中心像素已接近边界，但尚未稳定饱和。
- 10.0 等开始出现稳定饱和像素。
- 9.5 等时饱和区域进一步扩展。

## 5. 最终结论

### 5.1 PSF 结论

- 在 14 度视场角、field ID = 7、6.5 um 像元条件下，聚焦 PSF 的 EE90 为 5.831 pix。
- 离焦 -0.03 mm 时，EE90 增大到 7.000 pix，PSF 扩展最明显。
- 离焦 +0.03 mm 时，EE90 为 5.657 pix，本组模型中略小于聚焦值。

### 5.2 饱和边界结论

- 粗步长扫描显示，导星单星 stamp 的满阱饱和边界位于 10.5 等与 10.0 等之间。
- 细步长扫描进一步将边界收敛到 10.5 等与 10.4 等之间。
- 10.4 等时已出现显著饱和风险，500 次 realization 中有 363 次出现饱和。
- 10.3 等及更亮可视为稳定饱和区。
- 如果报告需要给出一个便于传达的保守阈值，可以表述为：14 度视场角、field ID = 7、聚焦导星 PSF 条件下，10.4 等附近已进入饱和风险区，10.3 等及更亮稳定饱和。

## 6. 汇报时可直接使用的表述

可在汇报中直接使用如下结论句：

“基于 14 度视场角、field ID = 7 的 1x1 子像元导星 PSF，我们对 31x31 单星导星 stamp 做了 500 次噪声 realization 饱和评估。结果表明，满阱饱和边界位于 10.5 等和 10.4 等之间，其中 10.4 等已经出现约 72.6% 的饱和概率，10.3 等及更亮稳定饱和。同时，离焦会显著改变 PSF 能量集中程度，-0.03 mm 离焦下 EE90 从聚焦时的 5.831 pix 增大到 7.000 pix。”

## 7. 复现文件

- PSF 出图脚本：../../cluster_mount/cluster_ET7/ET-fsglib/examples2/plot_field7_psf_ee90.py
- 饱和分析 notebook：guide_star_saturation_report.ipynb
- 饱和扫描脚本：../scripts/guide_stamp_saturation_sweep.py
