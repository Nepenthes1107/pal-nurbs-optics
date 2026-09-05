# BIOT 与 PAL-NURBS 光学优化

本目录是唯一开发与运行项目。后续上传到云服务器时只上传源码、测试、Excel、`requirements.txt` 和必要历史证据；`.venv` 是 Windows 本地环境，不能上传到 Linux。

## 目录结构

```text
biot/                  BIOT 与 PAL-NURBS 实现
inputs/pal/            PAL 固定输入（分区和 PSF support）
tests/                 自动化测试
results/optimization/  新优化运行；由程序按需创建
results/archive/       本机保留的旧实验，不进入 Git
run_pal_nurbs.py       PAL-NURBS 命令行入口
```

当前主线不使用 Phase 编号。每次正式运行只使用递增且易读的目录名，例如 `run_001`、`run_002`；运行身份仍由配置、输入和实现闭包的 SHA-256 决定，而不是由目录名决定。

## 环境

固定环境：Python 3.8.20、PyTorch 2.0.1、CUDA runtime 11.8、NumPy 1.24.4。依赖见 `requirements.txt`。

Windows 创建本地虚拟环境：

```powershell
D:\Anaconda\envs\myenv\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

环境检查：

```powershell
python -c "import sys,torch,numpy; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

云服务器上传源码后重新创建 Linux `.venv`，不要上传 Windows `.venv`。

## 稳定 BIOT

```powershell
python multi_rays.py eye_image_glass.xlsx inf 0 0 --cutoff 100 --device auto --output results/smoke_inf_0_0
python multi_rays.py eye_image_glass.xlsx inf 0 0 --cutoff 100 --device auto --with-mtf --output results/smoke_psf_mtf
python averfang.py eye_image_glass.xlsx --output results/averfang_map
python lens_metrics.py eye_image_glass.xlsx all --device auto --output results/lens_metrics
python -m biot.gui
```

长度用 mm、角度用 degree、波长用 nm。物理 PSF 必须 finite、非负并按总能量归一化；MTF 从物理 PSF 计算并做 DC 归一化。

## PAL-NURBS 合同

- 仅优化 PAL 后表面 cubic NURBS/B-spline 的 `zp`；`xp/yp/weight` 固定，边界控制环固定为零。
- 控制网格为 `7x7 -> 11x11 -> 19x19`，通过精确 knot refinement 晋级。
- `--steps S7 S11 S19` 定义三个阶段的最低完整 attempt 数，拒绝的候选也消耗
  attempt；最后一个非零阶段是 terminal stage。terminal 之前的阶段严格执行各自
  固定配额，terminal 达到最低配额后才允许 early stop，并可由
  `--max-extra-terminal-stage-steps` 追加最多 30 个 attempt。默认 `10/10/30`
  的 terminal 为 19×19；例如 `50/10/0` 的 terminal 为 11×11，19×19 只做
  不改变物理面形的精确 knot refinement。patience 从 terminal 第一步累计，最低
  配额前不允许 early stop；连续 7 个 attempt 未使 best 相对改善严格超过
  `1e-3` 时停止。
- 121 个 case 分为 9 组：Far 28、Corridor upper/middle/lower 各 5、
  Near 28、Near-robustness 12、Near-edge-astig 8、Peripheral-left/right 各 15。
  Near 与 Near-robustness 按 `field_y >= -40°` 的 core 和 `< -40°` 的 deep
  分层；Near forward/final 为 core `24/8`、deep `56/20`，Near-robustness
  forward/final 为 core `8/4`、deep `24/8`。Near-edge 保持
  `|lens_x| > 10 mm` 的 8 个独立边缘 case。
- 420-case WFNO 合格池的最终固定数量选择保持各自 `training_group`，并在不改变
  覆盖门槛和各组数量的前提下做 coverage-constrained 选择；确定性选择无法通过时失败关闭。
  phase 进度使用 `training_group/candidate_id/distance/field` 稳定源键，不依赖每轮
  重新编号的 `case_id`。
- Far 的 28 个 case 全部使用真实 `Dinf`，Near-robustness 使用 `D1000`，Near 使用 `D500`；
  corridor 从 Original PAL 中心带局部 ADD 逐行计算 `distance_mm=1000/ADD_D`。
- 使用真实可微追迹、GRIN3 固定步长 RK4、连续参考球 OPL 和 FFT PSF；
  离线 PSF 不参与反传。七个真实追迹功能组都直接使用归一化物理 FFT PSF 的
  Ahumada weighted-MTF loss；Zernike 仅保留为诊断量，不参与该分支目标。
- 两个分支统一使用 `legacy_pupil_phase=False`、`phase_reference="biot_reference_sphere"` 和 `remove_tilt=False`；训练 loss、健康检查和梯度均直接基于原始 `512×512` 物理 FFT PSF 及其物理像素间距。`130×130` crop/render 仅用于评价数据库与拼接显示。
- 默认 pupil 采样为 `np=256`、FFT 为 `512`；91 个功能 case 按
  `case_batch_size=8` 做 GPU tensor 追迹和 FFT，30 个周边 case 保持原来的
  `surface_only` A_D 目标且不做光线追迹；不因 OOM 自动缩小 batch。
- 训练输出统一包含 `stage`、`step`、`batch`、`loss`、`update` 和 `lr`；每个 run 的 `training.log` 持久化同样的进度摘要，中断时追加异常信息，各阶段仍保留结构化 `history.csv`。
- 追迹失败保持失败关闭，由当前 run 的资格筛选进度记录错误；底层追迹不在项目根目录自动导出 `wrong_result` Excel。
- `J=sum(group_weight*spatial_weight*case_metric)`，9 组权重显式且和为 1。
  功能组在合格候选凸包内按固定物理网格的 Voronoi/最近邻覆盖面积加权；
  peripheral 保持组内等权。七个真实追迹功能组
  使用与正式评价一致的 Ahumada weighted-MTF loss，并统一除以固定的 `0.10`
  无量纲容差；左右 peripheral 继续使用对应区平均 A_D 与固定 `0.80 D` 容差，
  保持 `surface_only` 且不做光线追迹。Original PAL baseline 不参与 loss 分母，
  只用于健康比例、改善率和审计。
- 每个新 run 在 Original PAL 7×7 和 `stage_7x7/final.pt` 保存九组梯度范数、
  两两余弦及与总梯度余弦；产物位于 `gradient_diagnostics/` 并绑定 run/checkpoint 哈希。
- 功能质量同时计算 `0°/45°/90°/135°` 四方向 Ahumada weighted-MTF，并用
  归一化 soft-min 聚合；legacy mean 仅保留用于兼容和报告。
- 面形平滑使用 monitored mask 内 81×81 物理网格的 Hessian bending energy。
  7×7 阶段只记录且权重为零；11×11/19×19 阶段按配置的 `smooth_lambda`
  生效。该项直接参与可微目标，不是控制点后处理或显示滤波。
- 约束包括 trace/PSF health、`P_far`、`ADD`、下缘监管带相对 Original PAL 的
  最大光焦度/像散变化和单步 sag trust region。
- 不自动运行历史 192-case posthoc 或 SSIM 链；PSF 数据库、weighted MTF、PSF stitch 和 chart stitch 仅在正式评价入口中运行。

## 完成优化后的评价

评价不会改写训练 run，并在 run 内创建独立的 `evaluation/` 身份。它先生成并
完整核验 D500/D1000/Dinf × baseline/optimized 六个条件级 HDF5 PSF 数据库，
数据库状态为 `complete` 后才依次生成 Ahumada weighted-MTF Mean、PSF stitch
和视标 stitch；同时输出 Sag 和 AverFang 光焦度/像散：

```powershell
python evaluate_pal_nurbs.py --run results/optimization/run_001 --device cuda --psf-batch-size 8 --blur-scale 4
```

`evaluation/averfang/` 的六张分布 PNG 使用与 `averfang.py` 一致的物理
`X/Y (mm)` 坐标、Y 方向、等比例轴、中心标记、边缘 3 像素显示裁剪和色标设计；
baseline/optimized 保留 14 级等高线，delta 使用以零为中心的对称冷暖色标且不画
等高线。对应 NPZ 数值不裁剪、不改写。

若完整 run 中需要单独评价某个已完成阶段，可直接选择该阶段保存的最优
`final.pt`，无需重跑训练。例如复用 `50/25/0` run 的 7×7 阶段：

```powershell
python evaluate_pal_nurbs.py --run results/optimization/v3_branch/run_005 --checkpoint-stage 7 --device cuda --psf-batch-size 8 --blur-scale 4
```

阶段评价要求源 run 已完整结束，且 `summary.json`、源 identity 与 checkpoint
元数据一致；结果独立写入 `<run>/evaluation_stage_7x7/`。默认不指定该参数时仍
评价最终 checkpoint 并写入 `<run>/evaluation/`。不得通过复制 checkpoint 或
修改 `summary.json` 冒充另一训练身份。

每个条件文件保存 81 个场点的原始 512×512 FFT PSF 和统一物理裁剪后的
130×130 渲染 PSF；weighted MTF 只读取原始 PSF，拼接只读取渲染 PSF。
每个 D500/D1000/Dinf 的 baseline、optimized 和 delta 除原有 9×9
`*_mean.png`/`*_mean_map.npz` 外，还在 `weighted_mtf/` 中生成
`*_mean_interpolated.png`，共 9 张。该显示产物复用 BIOT_vis 的规则网格
cubic 方法，将 9×9 原始场点上采样到 200×200；不外推、不填补
NaN/Inf，原始节点必须在 `1e-12` 容差内保持不变。插值仅用于平滑显示，
不修改 NPZ、weighted-MTF 数值、评价门禁或科学结论。
已完成的旧评价可用原命令加 `--resume` 补生这些图：程序保留原
`evaluation_identity.json` 及其 SHA-256，只允许源 run 因后续训练 schema
升级而从“当前”被重分类为“legacy”的非物理标签漂移。源 identity、
checkpoint SHA-256、PSF batch、场网格、运行库和所有其他评价字段
仍必须严格相同，已完成 HDF5 只校验和跳过，不重新追迹 PSF。
评价网格为 `[-40,40]` degree、步长 10 degree。`--resume` 精确核验并跳过
HDF5 中已完成节点；损坏或身份不符的节点失败关闭。`--blur-scale` 默认 4，
只控制视标拼接的显示模糊，不改变 PSF 数据库、MTF 或 PSF stitch。
PSF 追迹通过必选的 `raw_psf_batch()` 接口做 CUDA case 小批量并行，
`--psf-batch-size` 默认 8；最后一批按实际剩余场点数运行，OOM 或批量追迹
失败直接终止，不自动缩批也不回退到串行。恢复时仅对未完成场点重新分批，
已完成节点仍逐个核验并跳过。
运行期间控制台以 `[pal-eval]` 前缀即时显示阶段、六个条件、PSF batch、当前条件
场点数和全局场点数；中间 batch 使用 `status=RUNNING`，每个条件只在 HDF5 完成
核验后报告一次满计数 `status=DONE`，不重复打印末批终点。`--resume` 对已完成条件
或后处理阶段显示 `status=SKIP`。评价图统一使用 Matplotlib 自带的 DejaVu Serif，
不依赖系统安装 Times New Roman。
批量生成已提升评价 identity schema；改造前未完成的串行评价不能用
新 `--resume` 接续，应保留为历史证据并在新的空 `evaluation/` 目录从头评价。

旧 run 的训练物距会在评价身份中如实记录；评价物距固定为
D500/D1000/Dinf，不能据此改写旧训练结论。`inputs/evaluation/E1.xlsx` 是受控
视标输入；`.venv` 和 `results/` 不上传到 Git。

## 从零开始运行

```powershell
python run_pal_nurbs.py `
  --output results/optimization/run_001 `
  --excel eye_image_glass_grad3.xlsx --device cuda `
  --requested-np 256 --fft-size-px 512 --case-batch-size 8 `
  --steps 10 10 30 `
  --early-stopping-patience 7 `
  --relative-improvement-threshold 1e-3 `
  --max-extra-terminal-stage-steps 30 `
  --weighted-mtf-loss-tolerance 0.10 `
  --astigmatism-tolerance-D 0.80 `
  --smooth-lambda 0.05 `
  --smooth-curvature-scale-per-mm 1e-4 `
  --directional-softmin-temperature 0.02
```

该示例最低训练 50 个 attempt，最多训练 80 个 attempt。terminal stage 每个 attempt
完成后先原子保存 resume/history，再按 early stopping、学习率下限和额外预算
顺序判断；最低预算前若学习率跌破下限，则 run 失败且不写成功 `summary.json`。
后续零预算阶段只做精确 refinement；最终 19×19 表示完成 121-case 无梯度复核后
才标记 complete。`summary.json` 显式记录 terminal control count、terminal extra
attempt 数及其停止原因。

普通 `--resume` 只允许恢复同一运行目录中 identity、配置、输入哈希、实现闭包、
最低/最大预算、patience 计数和终止规则完全一致的 checkpoint。本次 early
stopping 改造提升了 run identity 与 stage-resume schema；旧 run/checkpoint
保持只读，不可按新方法恢复，必须使用新输出目录。当前项目不再支持跨平台
training-state/parity 导入。

## 从已完成父 run 分叉续训

`--parent-run` 与 `--start-stage {7,11,19}` 可从已完成父 run 的阶段 best 创建
全新 child run。父目录始终只读，child 使用新的 `--output`、run identity、fresh
Adam、初始学习率和 patience 计数；父 Adam 动量不会导入。child 的 `--steps`
只表示新增 attempt，因此起点之前必须为 0，起点本身必须大于 0。

合法起点不得早于父 run 最后一个非零训练阶段。例如父 run 为 `50/25/0` 时只能
从 11×11 或 19×19 开始；父 run 为 `50/0/0` 时可从 7×7、11×11 或 19×19
开始。若起点晚于父 terminal，程序逐级核验父 run 保存的零预算 `final.pt`/
`resume.pt`，并以 `1e-10` 容差确认 knot refinement 未改变面形或一、二阶导数。

下面从一个当前合同的已完成父 run 的 11×11 best 新增 11×11 的 10 个 attempt，再新增 19×19
的 10 个最低 attempt；terminal 19×19 最多再执行 30 个 attempt：

```powershell
python run_pal_nurbs.py `
  --output results/optimization/v3_branch/run_006 `
  --parent-run results/optimization/run_010 `
  --start-stage 11 `
  --steps 0 10 10 `
  --max-extra-terminal-stage-steps 30 `
  --smooth-lambda 0.05 `
  --early-stopping-patience 7 `
  --relative-improvement-threshold 0.001 `
  --case-batch-size 8 `
  --device cuda
```

从 19×19 开始使用 `--start-stage 19 --steps 0 0 10`。父子除输出、分叉参数、
新增预算、terminal 额外预算和导入路径外，其余物理、目标、约束、采样、batch、
seed、优化器超参数和 `smooth_lambda` 必须完全一致。程序还会验证并复用父 run 的
candidate trace、forward qualification、final-phase qualification 与
`baseline_state.pt`；任何缺失、损坏或身份不符都会直接失败，不重新计算兜底。

child 的 `summary.json` 分开保存父阶段历史、child 阶段历史、逐阶段新增/累计
steps、父 checkpoint/evidence 哈希和
`optimizer_policy=parent_best_fresh_adam`；`actual_training_steps` 仅统计 child
实际执行量。child 中断后只能在 child 自身目录使用 `--resume`。run identity
schema 11 的 run 只能使用同为 schema 11 且方法身份一致的已完成父源；旧
schema run、旧 baseline、旧资格池和旧 checkpoint 均保持只读，不能接入新目标。

已完成 child 也可继续作为下一代 `--parent-run`。累计训练步数读取并核验
`lineage_actual_training_steps_by_stage`，而不是只统计直接父 run 的本地新增步数。
新 child 会在自身 `gradient_diagnostics/` 保存绑定当前 child identity、同时记录
祖先来源哈希的诊断文件。早期只保存继承 manifest、未保存两份诊断 JSON 的已完成
child 仍保持只读；下一代会从其封印的 run identity 输入解析祖先诊断，并同时核验
identity、manifest 和文件 SHA-256，无法解析时明确失败。

完整 `candidate-trace` 进度可显式导入新 run：保存的原身份必须先通过自身哈希
校验，镜片内容 SHA-256 与物理追迹参数也必须匹配；Linux/Windows checkout 的
绝对路径不是物理身份字段。forward/final-phase 进度仍须严格匹配 pool identity。

预优化产物还包括 `dense_candidate_fields.json` 与
`dense_candidate_grid_on_lens.png`：前者保存全部物方密集场点经真实 PAL 追迹后
映射到镜片后表面的物理坐标，后者用于与资格池候选点图比较。
`candidate_reachability_on_lens.png` 使用叉号表示 true-traceable、空心圆表示
通过后续 WFNO/phase qualification 的 eligible（不再应用 zone/aperture clearance
安全门槛）；初始分区资格另记录为 `trace_eligible`。
`corridor_flank` 是 `corridor` 内部的诊断子 mask，
不是独立训练分区或独立 loss；分区分类和训练 case 均纳入 `corridor`。
候选进入资格池后仍必须通过 forward WFNO、final-phase qualification 和覆盖门禁；
这些后续失败不会被 clearance 规则掩盖。

## 验证

```powershell
python -m pytest tests -q --basetemp .tmp_pytest
```

最小 prepare：

```powershell
python run_pal_nurbs.py --output .tmp_prepare --excel eye_image_glass_grad3.xlsx --device cpu --prepare-only
```

正式实验只有在 `summary.json` 存在且 `run_state.status=complete` 时才算完成。r12 只作为历史未完成证据，不是当前运行的恢复入口或收益结论。
