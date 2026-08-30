# PAL-NURBS 优化链路改造规格

## CTX 背景与目标

当前链路以方法 A（19×19 B-spline，FPS 采样）为改造基线；旧的 80-case 及 7×7/243-case 固定网格仅作历史对照，不是本分支训练合约。当前问题是：

1. **M2 与 MTF 失配。**优化器通过压缩光斑长尾降低 M2，远区 M2 无梯度（已近衍射极限），近区 M2 改善 37% 但佩戴态 MTF 反而 −0.092。
2. **走廊物距单一。**走廊横跨 ADD 0.2–2.0D，设计物距从 5026mm 连续变化到 539mm，当前统一用 D1000 造成最高 5× 物距偏差。
3. **下缘无监管。**Y≈−21mm 处优化器倾倒像差（ΔPower 峰值 0.544D，ΔAstig 峰值 1.332D），该区在 monitored mask 内但既不在任何训练分区、也不被处方门禁约束。
4. **分区白区。**zones.json v2 有 8.49%（412/4853 px）未归属，导致 `classify_partition_point()` 对这些点返回 `None`，候选点被整体丢弃。
5. **FOV 覆盖不足。**`generate_dense_candidate_fields` 生成方形网格，±40° 在 Y 向只覆盖镜片 −20.5mm，近用参考平台（Y=[−17,−37]mm）仅有 23% 落在内。

**改造目标：**保留方法 A（FPS + 联合训练 + 19×19 精化），修复以上五项缺陷，同时将周边 case 的 GPU 追迹节省约 40%，预期训练时间从 6.6h 降至约 4.2h。

> **架构决策**
>
> 新分支基于方法 A 进行改造，不采用方法 B。方法 B 的 MTF 表现来自自由度限制（7×7 天然带限），不是可控的优化机制；方法 A 的问题均为可修的工程缺陷。

## S1 分区更新（zones.json → v3）

### v3 算法摘要

v3 分区算法已在 `_scratch_wf/complete_partition_v3.py` 实现，产出 `_scratch_wf/partition_v3.npz`，验证达到 0/4853 未分配。核心方法：对每个 monitored 像素，按归一化加光度 *t = (P − P_far)/ADD* 和面形像散 A_D 做分层分类，得到 9 个语义标签：

| 标签                            | 物理含义                 | t 范围    | A_D 范围          |
| ------------------------------- | ------------------------ | --------- | ----------------- |
| `far_reference`               | 高纯度远用处方平台       | t < 0.025 | < 0.05D           |
| `far`                         | 远用功能区（宽）         | t < 0.10  | < 0.5D            |
| `corridor`                    | 走廊中心线               | 0.1–0.9  | < 1.0D,\|x\|<8mm  |
| `corridor_flank`              | 走廊侧翼                 | 0.1–0.9  | < 1.0D,\|x\|≥8mm |
| `near`                        | 近用功能区（宽）         | t > 0.90  | < 0.5D            |
| `near_reference`              | 高纯度近用处方平台       | t > 0.975 | < 0.05D           |
| `peripheral_astig_left/right` | 周边高像散翼（左/右）    | 任意      | ≥ 0.65D          |
| `transition`                  | 功能区与周边区之间的过渡 | 任意      | 0.5–0.65D        |
| `monitored`                   | 完整监管孔径             | —        | —                |

### zones.json 接口与写入

需将 v3 结果写入正式的 `inputs/pal/zones.json`，格式与现有 schema_version=2 保持兼容，但增加新标签 mask。**以下字段必须保留**，因为多处代码直接读取：

- `masks.far`、`masks.far_reference`、`masks.near`、`masks.near_reference`
- `masks.corridor`、`masks.peripheral_astig_left`、`masks.peripheral_astig_right`、`masks.monitored`
- `statistics.corridor.physical_y_range_mm`——由 `biot/e2e/pal_nurbs.py:1223` 直接读取用于确定走廊层边界

新增 mask：`corridor_flank`、`transition`。

在 `biot/e2e/pal_case_layout.py` 的 `PARTITION_ORDER`（当前第 29 行）中，`corridor_flank` 和 `transition` 的归属规则：

```
# biot/e2e/pal_case_layout.py — PARTITION_ORDER 扩展
PARTITION_ORDER = (
    "far", "corridor", "corridor_flank", "near",
    "peripheral_astig_left", "peripheral_astig_right",
)
```

在 `classify_partition_point()`（第 83–93 行）的返回映射中新增：

```
return {
    "peripheral_astig_left":  "astig_left",
    "peripheral_astig_right": "astig_right",
    "corridor_flank": "corridor",   # 侧翼归入 corridor 分组
    "transition":     "far",         # 过渡区按 t<0.5 归 far，t≥0.5 归 near（见下）
}.get(active[0], active[0])
```

> **注意**
>
> `transition` 像素需在 classify 前先判断 t 值：t < 0.5 → `far`，t ≥ 0.5 → `near`。`classify_partition_point()` 已有 averfang maps 可用，在 active 判断后追加这条分支即可。

> **下缘监管带（独立任务）**
>
> 在 v3 zones.json 中新增一个 `lower_edge_guard` mask，覆盖 Y∈[−18,−23]mm 的环形带。此 mask **不进入** PARTITION_ORDER（不影响 case 分组），仅用于 `prescription_metrics()` 中追加审计：
>
> ```
> guard_mask = zones["lower_edge_guard"] & valid
> if guard_mask.any():
>     dp = maps["power_D"][guard_mask] - baseline_maps["power_D"][guard_mask]
>     da = maps["astigmatism_D"][guard_mask] - baseline_maps["astigmatism_D"][guard_mask]
>     if dp.abs().max() > 0.5 or da.max() > 0.8:
>         raise ValueError("lower_edge_guard prescription violated")
> ```
>
> 这里的 Δ 是候选面形相对同一 Original PAL 基线的逐像素变化，不是该区域相对
> `P_far` 的绝对处方值；否则 Original PAL 本身会在近用/周边带立即失败。容差：
> max|ΔPower| < 0.5D，max|ΔAstig| < 0.8D（当前 run_004 峰值
> 0.544D/1.332D，均已超标）。

## S2 候选场角生成

### 非对称 X/Y 网格

当前 `generate_dense_candidate_fields`（`biot/e2e/pal_case_layout.py:117`）仅接受 `field_min_deg`/`field_max_deg`（方形对称网格）。PAL 镜片 Y 向不对称，Y 向下需要 −55° 以覆盖近区参考平台，Y 向上无需超过 +55°；X 向可保持对称。需扩展函数签名：

```
def generate_dense_candidate_fields(
    *,
    field_x_min_deg: float,
    field_x_max_deg: float,
    field_y_min_deg: float,
    field_y_max_deg: float,
    field_step_deg: float,
    # 向后兼容别名（可选）
    field_min_deg: float | None = None,
    field_max_deg: float | None = None,
) -> list[dict[str, Any]]:
    # 向后兼容：若旧参数传入则覆盖 XY
    if field_min_deg is not None:
        field_x_min_deg = field_y_min_deg = field_min_deg
    if field_max_deg is not None:
        field_x_max_deg = field_y_max_deg = field_max_deg
    x_values = _linspace_exact(field_x_min_deg, field_x_max_deg, field_step_deg)
    y_values = _linspace_exact(field_y_min_deg, field_y_max_deg, field_step_deg)
    return [
        {"candidate_id": f"cand_{idx+1:05d}",
         "field_x_deg": float(fx),
         "field_y_deg": float(fy)}
        for idx, (fy, fx) in enumerate(
            (fy, fx) for fy in y_values for fx in x_values
        )
    ]
```

在 `MinimalConfig`（`biot/e2e/pal_nurbs.py:63`）新增字段，并更新 `_prepare_case_layout` 的调用：

```
# MinimalConfig 新增字段
candidate_field_x_min_deg: float = -45.0
candidate_field_x_max_deg: float =  45.0
candidate_field_y_min_deg: float = -60.0
candidate_field_y_max_deg: float =  55.0
```

旧字段 `candidate_field_min_deg` / `candidate_field_max_deg` 保留但废弃（加注释），`_prepare_case_layout` 改用新四字段调用。

| 阶段                 | X 范围          | Y 范围          | 步长 | 候选数（约） |
| -------------------- | --------------- | --------------- | ---- | ------------ |
| 候选生成             | [−45°, +45°] | [−60°, +55°] | 1°  | ~10 500      |
| 训练分组（FPS 源池） | —              | —              | —   | 同上全量     |
| 评价网格             | [−40°, +40°] | [−40°, +40°] | 10° | 81（9×9 ）  |

> **FOV 非对称的物理依据**
>
> 用真实追迹落点拟合：*h_y = 24.4·tan(θ_y)*，±40° 的 Y 向下仅覆盖镜片 −20.5mm，近用参考平台 Y=[−17,−37]mm 仅有 23% 覆盖率。Y 向上无需超过 +40°（far_reference 上界在 +21mm，上方 0.12D 功率漂移超容差）。

## S3 训练 case 合约

### 新分组定义与数量

历史合约（`biot/e2e/pal_case_layout.py:17`）为 5 组共 80 个 case：far 18 + intermediate 12 + near 18 + peripheral_{L/R} 各 16；本分支使用下方的 109-case 新合约。

新合约：**109 个 case**，其中 79 个功能 case 需要追迹、30 个周边 case 免追迹（仅面形计算）。

| 分组键名             | 对应 zone         | case 数       | 物距（mm）     | 权重           | 追迹 |
| -------------------- | ----------------- | ------------- | -------------- | -------------- | ---- |
| `far`              | far               | 20            | Inf            | 0.22           | 是   |
| `far_robustness`   | far               | 8             | 1000           | 0.02           | 是   |
| `corridor_upper`   | corridor          | 5             | 插值（见 S3b） | 0.07           | 是   |
| `corridor_middle`  | corridor          | 5             | 插值           | 0.10           | 是   |
| `corridor_lower`   | corridor          | 5             | 插值           | 0.11           | 是   |
| `near`             | near              | 20            | 500            | 0.18           | 是   |
| `near_robustness`  | near              | 8             | 1000           | 0.02           | 是   |
| `near_edge_astig`  | near (\|x\|>10mm) | 8             | 500            | 0.04           | 是   |
| `peripheral_left`  | astig_left        | 15            | Inf            | 0.12           | 否   |
| `peripheral_right` | astig_right       | 15            | Inf            | 0.12           | 否   |
| **合计**       |                   | **109** |                | **1.00** |      |

> **合约扩展注意**
>
> 现有代码在 `biot/e2e/pal_case_layout.py:487` 有硬编码校验：`"intermediate selection remains fixed at 4 layers x 3 points"`。新合约废弃 `intermediate` 分组，改为 `corridor_upper/middle/lower` 三组，需删除该校验并更新 `TRAINING_GROUP_COUNTS`、`FUNCTIONAL_GROUPS` 和 `GROUP_TO_ZONE` 映射。同时需将 `CASE_LAYOUT_STATE_SCHEMA_VERSION`（当前为 6，`biot/e2e/pal_nurbs.py:142`）递增至 7，并更新 `METHOD_NAME`。

### 走廊 case 插值物距

走廊每个 case 的物距从原始 PAL 的 averfang maps 中计算：

```
def corridor_object_distance_mm(
    power_map: np.ndarray,
    pfar: float,
    case_lens_y_mm: float,
    zones_payload: dict,
) -> float:
    """已知候选点的镜片后表面 Y 坐标，计算走廊设计物距。"""
    y_mm = np.asarray(zones_payload["physical_y_mm"])
    x_mm = np.asarray(zones_payload["x_mm"])
    # 在 x=0 中心线（±2mm）内取功率均值
    cx = np.abs(x_mm) <= 2.0
    iy = int(np.argmin(np.abs(y_mm - case_lens_y_mm)))
    local_power = np.nanmean(power_map[iy, cx])
    local_add = local_power - pfar
    local_add = max(local_add, 0.05)   # 避免除零
    return 1000.0 / local_add             # D → mm
```

走廊三层按 add_target 分带：

- `corridor_upper`：ADD ∈ [0.2, 0.5D] → d ≈ 2000–5000mm
- `corridor_middle`：ADD ∈ [0.5, 1.3D] → d ≈ 770–2000mm
- `corridor_lower`：ADD ∈ [1.3, 2.0D] → d ≈ 500–770mm

分带边界从 `corridor.statistics.power_min/max_D`（zones.json）读取后计算 ADD，无需硬编码。物距在 `select_training_cases` 写入每个 case 的 `distance_mm` 字段。

### select_training_cases 修改要点

文件：`biot/e2e/pal_case_layout.py:472`。主要改动：

1. 新增 `power_map` 和 `pfar` 参数，供走廊物距插值使用。
2. 将原 `"intermediate": _select_corridor(...)` 替换为三个分组的 FPS 选取：
   各组从 corridor zone 候选中按 ADD 分带过滤，再在镜片坐标 (x_mm, y_mm) 上做 FPS。
3. 新增 `near_edge_astig`：从 near zone 候选中过滤 |x_mm| > 10mm 的，再 FPS 取 8 个。
4. `far_robustness`：从 far zone 候选中用 seed_target=[0, 15] 做 FPS 取 8 个。
5. `near_robustness`：从 near zone 候选中 FPS 取 8 个（与 near 分开 FPS，允许重叠）。
6. 周边减少到 15 对（各 15），带数可改为 `{"upper": 4, "middle": 5, "lower": 6}` 共 15。

> **FPS 种子点**
>
> `_fps_rows` 已支持 `seed_target`（`biot/e2e/pal_case_layout.py:314`）。corridor_upper FPS 种子取走廊上层中心；far_robustness 种子取 [0, +15mm]（仪表板视距）；near_edge_astig 种子取 [±12mm, −28mm]（斜入射区中心）。

## S4 损失函数改造

### 指标路由

修改 `_loss_metrics_for_batch`（`biot/e2e/pal_nurbs.py:1746`）。当前逻辑：PERIPHERAL_GROUPS → A_D，其余 → M2。新逻辑按训练分组路由：

| 分组                                          | 指标                             | 物理依据                             |
| --------------------------------------------- | -------------------------------- | ------------------------------------ |
| `far`, `far_robustness`                   | CSF 加权 MTF 积分（0–30 lp/mm） | 衍射主导，M2 无梯度                  |
| `corridor_*`, `near`, `near_robustness` | Zernike Z₄²（离焦波前均方）    | 离焦主导，Z₄ 与视觉单调             |
| `near_edge_astig`                           | Zernike Z₄² + 小权重 A_D       | 离焦 + 斜入射像散                    |
| `peripheral_left/right`                     | 面形 A_D（`astig_A_by_zone`）  | 几何像散，物距无关，**免追迹** |

### CSF 加权 MTF 计算

目前 evaluate_pal_nurbs.py 已有 CSF 实现（`COMMON_FREQ = np.linspace(0, 100, 1000)`，`CSF_MM_PER_DEG=0.291`）。在训练 loss 中集成：

```
def csf_weighted_mtf_loss(psf_kernel: torch.Tensor, pixel_pitch_mm: float) -> torch.Tensor:
    """
    CSF 加权 MTF 积分损失（越小越好）。
    采样：0–30 lp/mm（= 0–8.7 cyc/deg，位于人眼 CSF 峰附近）。
    """
    # 二维 FFT → MTF
    h, w = psf_kernel.shape[-2:]
    otf = torch.fft.fftshift(torch.fft.fft2(psf_kernel))
    mtf = otf.abs() / otf.abs()[..., h//2, w//2].unsqueeze(-1).unsqueeze(-1)
    # 径向平均到 0–30 lp/mm（离散 60 点）
    freq_lpmm = torch.linspace(0, 30, 60, device=psf_kernel.device, dtype=psf_kernel.dtype)
    mtf_radial = _radial_average_mtf(mtf, freq_lpmm, pixel_pitch_mm)
    # CSF 权重（Mannos-Sakrison 近似，已换算到 lp/mm）
    freq_cpd = freq_lpmm * 0.291
    csf = (2.6 * (0.0192 + 0.114 * freq_cpd)
           * torch.exp(-(0.114 * freq_cpd) ** 1.1))
    csf = csf / csf.sum()
    return 1.0 - (mtf_radial * csf).sum()   # 损失 = 1 - 加权 MTF
```

> **径向平均辅助函数**
>
> `_radial_average_mtf(mtf, freq_lpmm, pixel_pitch_mm)`：以中心为原点计算每个像素的空间频率（单位 lp/mm），对目标频率点做最近邻或线性插值取均值。频率上限 30 lp/mm 对应 8.7 cyc/deg，在人眼 CSF 峰以内，梯度可靠。

### Z4 离焦 Zernike 计算

复用稳定 BIOT 的数学合同，但不直接调用 `optics.py::fit_wavefront_zernike()`：

- 输入必须是追迹器在 `phase_reference="biot_reference_sphere"` 下输出的连续
  `reference_opl_mm`，以及同一批光线的 `valid` mask；禁止从强度 PSF 反演光瞳相位。
- 使用与 BIOT 一致的 OSA/ANSI 实 Zernike RMS 归一化、正 `m` 为 cosine、负
  `m` 为 sine，并联合拟合 `n<=2` 的六个低阶项，避免 piston、tilt、astigmatism
  在离散或非完整有效光瞳上泄漏到 defocus。
- BIOT 现有实现是 `detach -> CPU -> NumPy` 的诊断链，不能进入训练。需在
  `biot/e2e` 中实现 Torch QR/三角求解版本，保持 dtype/device 和 autograd；样本不足、
  非 finite 或设计矩阵秩不足时显式失败，不加正则化或回退指标。
- Z4 以 `(n,m)=(2,0)` 取值；函数返回 OPD 系数的平方，单位固定为 `mm²`。
  若另行输出 waves，仅作为诊断值 `coefficient_mm / wavelength_mm`，不得混入该损失。
- `pixel_pitch_mm` 是像面采样参数，不属于 Zernike OPD 拟合；`pupil_radius_mm`
  已在光线生成时用于物理瞳孔，拟合使用与 BIOT 相同的单位圆归一化坐标。

```python
def fit_low_order_opd_zernike_torch(
    reference_opl_mm: torch.Tensor,
    valid: torch.Tensor,
    *,
    sample_count: int,
) -> torch.Tensor:
    """返回 [..., 6] 的 n<=2 OSA/ANSI RMS 归一化 OPD 系数 [mm]。"""
    # 单位圆采样顺序必须与 pupil_disk_grid()/FFT 光瞳完全一致。
    # 对每个 case 取 valid 行构造 A，并以 reduced QR 解 A @ c ~= OPD。
    # A 不依赖待优化参数；c 对 reference_opl_mm 保持可微。
    ...


def z4_defocus_loss(
    reference_opl_mm: torch.Tensor,
    valid: torch.Tensor,
    *,
    sample_count: int,
) -> torch.Tensor:
    coefficients_mm = fit_low_order_opd_zernike_torch(
        reference_opl_mm, valid, sample_count=sample_count
    )
    z4_mm = coefficients_mm[..., 4]  # OSA/ANSI j=4 == (n,m)=(2,0)
    return z4_mm.square()
```

> **直接验证策略（替代 M2 过渡期）**
>
> 不保留 M2 过渡模式，也不通过 10 步训练曲线间接判断 Z4 是否可用。先用合成连续
> OPD 精确恢复六个已知低阶系数，并对 Z4² 做 autograd/有限差分一致性检查；再取一个
> corridor 与一个 near 的真实 PAL 小批量，将 Torch 系数与同一份 detached OPD 送入
> BIOT NumPy 拟合所得 `(2,0)` 系数逐 case 对比。恢复误差和实现间误差均通过后，训练
> 直接启用 Z4；任何一项失败都停止，不回退到 M2。

### 目标函数权重合成

修改 `_evaluate`（`biot/e2e/pal_nurbs.py:1876`）的 coefficients 计算。新合约有 10 个分组，权重显式配置，不再依赖 `functional_weight / (3 * group_count)` 的均分公式：

```
# MinimalConfig 中的权重配置（替换 functional_objective_weight / peripheral_objective_weight）
group_weights: dict[str, float] = {
    "far":              0.22,
    "far_robustness":   0.02,
    "corridor_upper":   0.07,
    "corridor_middle":  0.10,
    "corridor_lower":   0.11,
    "near":             0.18,
    "near_robustness":  0.02,
    "near_edge_astig":  0.04,
    "peripheral_left":  0.12,
    "peripheral_right": 0.12,
}
# near_edge_astig 组内按归一化分数合成：90% Z4² + 10% near 区 A_D。
near_edge_astig_A_weight: float = 0.10
# coefficients 计算替换为：
coefficients = torch.as_tensor(
    [group_weights[case["training_group"]] / group_counts[case["training_group"]]
     for case in batch],
    device=scores.device, dtype=scores.dtype,
)
```

## S5 周边 case 免追迹

周边像散 A_D 由 `astig_A_by_zone()`（`biot/e2e/pal_nurbs.py:763`）计算，直接从 sag + perturbation delta 求面形曲率差，**不需要追迹任何光线**。此函数已存在，当前 `_loss_metrics_for_batch` 已在 `needs_astig=True` 时调用它（`biot/e2e/pal_nurbs.py:1751`）。

问题在于：`_field_batch` 对所有 batch 中的 case 都做追迹，包括周边 case。需要将周边 case **移出** `_field_batch` 的追迹范围：

```
def _evaluate(model, cases, baseline, ...):
    PERIPHERAL = {"peripheral_left", "peripheral_right"}
    traced_cases = [c for c in cases if c["training_group"] not in PERIPHERAL]
    periph_cases = [c for c in cases if c["training_group"] in PERIPHERAL]
    # 追迹非周边 case
    for batch in _case_batches(traced_cases, batch_size):
        result = _field_batch(model, batch)
        ... 原有逻辑 ...
    # 周边 case：直接从 astig_A_by_zone 取值，构造 dummy scores
    astig_by_zone = model.astig_A_by_zone()
    for case in periph_cases:
        zone = GROUP_TO_ZONE[case["training_group"]]
        raw_a = astig_by_zone[zone]
        baseline_a = baseline[case["case_id"]]["loss_metric"]
        scores_periph.append(raw_a / baseline_a)
        ... 构造 row 并追加到 rows ...
```

> **实现说明**
>
> `astig_A_by_zone` 当前对 `astig_left` 和 `astig_right` 各返回一个标量（zone 均值），而新合约每组有 15 个 case。这 15 个 case 共享同一个区均值分数，梯度仍正确（区均值对 sag 参数的梯度是 case 级梯度的均值）。如需 case 级 A_D，则需在 `astig_A_by_zone` 中按候选点坐标取局部均值——先按组均值实现，后续有需要再精化。

## S6 平滑正则化

仅在 19×19 阶段启用，防止高频振铃累积（run_004 在 80 步后 sag 仍以 0.11%/step 下降、面形振幅持续增加）。

```
def laplacian_regularizer(module: FixedWeightNURBSPerturbation) -> torch.Tensor:
    """控制点拉普拉斯正则项，约束面形高频。"""
    q = module.inner_q                      # shape: (n, n) 控制点
    lap_y = q[2:] - 2 * q[1:-1] + q[:-2]  # 二阶差分 Y 方向
    lap_x = q[:, 2:] - 2 * q[:, 1:-1] + q[:, :-2]  # X 方向
    return (lap_y ** 2).mean() + (lap_x ** 2).mean()
```

在 19×19 阶段的 loss 中加入：

```
if control_count == 19:
    smooth_loss = laplacian_regularizer(module)
    total_loss = candidate + config.smooth_lambda * smooth_loss
else:
    total_loss = candidate
```

在 `MinimalConfig` 增加字段：

```
smooth_lambda: float = 0.05   # 初始值，可在 run_pal_nurbs.py CLI 中覆盖
```

> **调参指南**
>
> `smooth_lambda` 的合理范围为 0.01–0.2。若训练 J 改善过慢（<0.5%/10步），先将 lambda 减半；若面形二阶差分仍在增大，则加倍。run_004 的振铃 max|Δsag|=0.022mm 集中在 19×19 阶段第 40 步后，lambda=0.05 目标是将其控制在 0.015mm 以内。

## S7 处方审计扩展

`prescription_metrics`（`biot/e2e/pal_nurbs.py:1148`）现有逻辑：计算 P_far、ADD、astig_mean，若 |ΔP_far| > 0.2D 或 |ΔADD| > 0.3D 则拒绝 step（在 `biot/e2e/pal_nurbs.py:2752` 检查）。

需新增 **下缘监管带检查**（详见 S1 callout）。此外，建议将 `far_tolerance_D` 和 `add_tolerance_D` 的默认值从 0.2/0.3D 收紧到 0.15/0.25D，因为新合约有更多 far 和 near case，优化器不再需要如此宽的处方自由度。

```
# MinimalConfig
far_tolerance_D: float = 0.15   # 原 0.2
add_tolerance_D: float = 0.25   # 原 0.3
lower_edge_power_tolerance_D: float = 0.50
lower_edge_astig_tolerance_D: float = 0.80
```

## S8 常量与方法名更新汇总

| 位置                                                      | 当前值                             | 新值                                                                                                                        |
| --------------------------------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `pal_nurbs.py:59` `METHOD_NAME`                       | `...bspline7_ladder_earlystop19` | `...stratified_corridor_csf_z4_smooth_v3zones`                                                                            |
| `pal_nurbs.py:142` `CASE_LAYOUT_STATE_SCHEMA_VERSION` | `6`                              | `7`                                                                                                                       |
| `pal_case_layout.py:17` `TRAINING_GROUP_COUNTS`       | 5 组 80 case                       | 10 组 109 case（见 S3 表格）                                                                                                |
| `pal_case_layout.py:27` `FUNCTIONAL_GROUPS`           | `("far","intermediate","near")`  | `("far","far_robustness","corridor_upper","corridor_middle","corridor_lower","near","near_robustness","near_edge_astig")` |
| `pal_case_layout.py:29` `PARTITION_ORDER`             | 5 项                               | 6 项（新增`corridor_flank`）                                                                                              |
| `pal_case_layout.py:22` `CORRIDOR_LAYER_COUNT`        | `4`                              | 废弃（由三组分别 FPS 替代）                                                                                                 |
| `pal_nurbs.py:104` `functional_objective_weight`      | `0.85`                           | 废弃（改为显式`group_weights` dict）                                                                                      |
| `pal_nurbs.py:98` `candidate_field_min/max_deg`       | ±55°                             | 废弃（改为四字段 XY 分离）                                                                                                  |
| `pal_nurbs.py:86` `early_stopping_patience`           | `7`                              | 保持`7`，阈值从 0.0001 改为 0.001（宽松）                                                                                 |
| `pal_nurbs.py:85` `max_steps_19` 默认                 | `10`                             | 实现默认`30`，`max_extra_19_steps` 为 `30`                                                                      |

## EX 改动顺序

| 步骤 | 优先级  | 操作                                                                                                                                                                                                                                       | 风险                                                                                              |
| ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| 1    | P1 安全 | 将 v3 结果写入`inputs/pal/zones.json`；更新 `PARTITION_ORDER`、`classify_partition_point` 映射；`prescription_metrics` 加下缘审计                                                                                                  | zones.json 格式变化可能触发`identity_sha256` 不匹配，需同步清空已有 run 缓存或新建 run 目录     |
| 2    | P1 安全 | 将周边 case 移出`_field_batch` 追迹路径（S5）                                                                                                                                                                                            | 最低风险，周边`astig_A_by_zone` 已有测试                                                        |
| 3    | P2 核心 | 扩展`generate_dense_candidate_fields` 支持非对称 XY（S2）；更新 `MinimalConfig` 四字段；更新 `_prepare_case_layout` 调用                                                                                                             | 候选 ID 格式变化（`cand_04d` → `cand_05d`），注意不影响 sha256 计算                          |
| 4    | P2 核心 | 重写`select_training_cases`：新增三层走廊分组、near_robustness、near_edge_astig、far_robustness；走廊插值物距（S3）；更新 `TRAINING_GROUP_COUNTS`、`FUNCTIONAL_GROUPS`、`GROUP_TO_ZONE`；递增 `CASE_LAYOUT_STATE_SCHEMA_VERSION` | 高：合约改变后所有已有`case_layout_state.json` 因 schema_version 不匹配会自动重算，耗时约 2–3h |
| 5    | P2 核心 | 在`biot/e2e` 实现连续 OPD 的可微 Torch 低阶 Zernike 拟合；完成合成 OPD 精确恢复、autograd/有限差分和真实 PAL 小批量对照（V2）                                                                                                                | 中：必须与 BIOT 的基、坐标、mask、单位和系数顺序逐项一致                                           |
| 6    | P3 增强 | 加入平滑正则化 lambda=0.05（S6）；更新`METHOD_NAME`                                                                                                                                                                                      | 低                                                                                                |
| 7    | P3 增强 | 在`_loss_metrics_for_batch` 一次性启用完整指标路由：far → CSF-MTF，corridor/near → Z4，near_edge_astig → Z4 + A_D；不保留 M2 回退                                                                                                        | 中：需逐组检查 finite、单位、baseline 分母和 NURBS 梯度                                           |

## V 验证节点

#### V1 — 分区完整性（步骤 1 完成后）

```
python -c "
import json, numpy as np
z = json.load(open('inputs/pal/zones.json'))
from biot.e2e.pal_case_layout import PARTITION_ORDER
mon = np.array(z['masks']['monitored'], bool)
covered = np.zeros(mon.shape, bool)
for k in PARTITION_ORDER:
    covered |= np.array(z['masks'].get(k, np.zeros(mon.shape, bool)), bool)
uncovered = mon & ~covered
print(f'uncovered: {uncovered.sum()}/{mon.sum()} ({100*uncovered.sum()/mon.sum():.2f}%)')
assert uncovered.sum() == 0, 'v3 分区不完整！'
"
```

期望：`uncovered: 0/4853 (0.00%)`

#### V2 — Z4 直接数值与梯度验证（步骤 5 完成后）

```
python -m pytest tests/test_e2e_opd_zernike.py -q --basetemp .tmp_v3_zernike
```

必须同时满足：

- 合成 `n<=2` 连续 OPD 的六项系数恢复绝对误差 `<=1e-10 mm`；
- Z4² 对输入 OPD 的 autograd 与中心有限差分相对误差 `<=1e-5`；
- 一个 corridor 与一个 near 真实 PAL case 的 Torch `(2,0)` 系数，与同一份 detached
  OPD 经 `optics.py::fit_wavefront_zernike(..., n_max=2)` 得到的系数绝对误差
  `<=1e-10 mm`；
- 两个真实 case 的 Z4² 均 finite、非负，并能反传到 PAL NURBS 参数，梯度 finite 且
  至少一个可训练控制点梯度非零。

上述门槛验证实现本身，不声称优化收益。完整指标路由启用后只再运行一个 attempt 的
CLI smoke，核对 10 组 metric 名称、baseline 分母、权重和日志；V3 完整训练仍是收益验收入口。

#### V3 — 完整训练（所有步骤完成后）

```
python run_pal_nurbs.py \
  --output results/optimization/v3_branch/run_001 \
  --steps 0 15 30 \
  --max-extra-19-steps 30 \
  --smooth-lambda 0.05 \
  --early-stopping-patience 7 \
  --relative-improvement-threshold 0.001 \
  --case-batch-size 8 \
  --device cuda
```

成功标准：

- `improvement_percent` ≥ 12%
- `final_metrics.J_far` ≤ 0.98（远区 MTF 不退化）
- `max_abs_sag_delta_mm` ≤ 0.015（正则化有效）
- `runtime_seconds` ≤ 16000（约 4.4h，对应 79 traced + 30 untraced × 45 steps）
- evaluate_pal_nurbs.py 的 weighted_mtf 图：Dinf 远区、D1000 走廊、D500 近区均正向改善

---

本规格基于 main/run_004 与 codex/run_001 实验结果分析生成；实施版已升级为 `CASE_LAYOUT_STATE_SCHEMA_VERSION=7`，并废弃 `CORRIDOR_LAYER_COUNT` 固定层方案。
