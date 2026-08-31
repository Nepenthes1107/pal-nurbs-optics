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
- `--steps S7 S11 S19` 中，7×7 和 11×11 恰好各运行 `S7`、`S11`
  个完整 attempt，19×19 至少运行 `S19` 个；拒绝的候选也消耗 attempt。
  达到最低总预算 `S7+S11+S19` 后只在 19×19 继续训练。默认 `S19=30`，
  连续 7 个 attempt 未使 best 相对改善严格超过 `1e-3` 时 early stop，
  最多额外运行 30 个 attempt。最低 19×19 配额前只累计 patience，不允许 early stop。
- 109 个 case 分为 10 组：Far 20、Far-robustness 8、Corridor upper/middle/lower 各 5、
  Near 20、Near-robustness 8、Near-edge-astig 8、Peripheral-left/right 各 15。
- 420-case WFNO 合格池的最终固定数量选择保持各自 `training_group`，并在不改变
  覆盖门槛和各组数量的前提下做 coverage-constrained 选择；确定性选择无法通过时失败关闭。
  phase 进度使用 `training_group/candidate_id/distance/field` 稳定源键，不依赖每轮
  重新编号的 `case_id`。
- Far 使用真实 `Dinf`，robustness 使用 `D1000`，Near 使用 `D500`；
  corridor 从 Original PAL 中心带局部 ADD 逐行计算 `distance_mm=1000/ADD_D`。
- 使用真实可微追迹、GRIN3 固定步长 RK4、连续参考球 OPL 和 FFT PSF；
  离线 PSF 不参与反传。Z4 复用 BIOT 的 OSA/ANSI RMS 基、全低阶最小二乘与
  `(n,m)=(2,0)` 映射，执行层为 Torch QR，不从 PSF 逆变换相位。
- 两个分支统一使用 `legacy_pupil_phase=False`、`phase_reference="biot_reference_sphere"` 和 `remove_tilt=False`；训练 loss、健康检查和梯度均直接基于原始 `512×512` 物理 FFT PSF 及其物理像素间距。`130×130` crop/render 仅用于评价数据库与拼接显示。
- 默认 pupil 采样为 `np=256`、FFT 为 `512`；79 个功能 case 按
  `case_batch_size=8` 做 GPU tensor 追迹和 FFT，30 个周边 case 直接使用面形 A_D，
  不做光线追迹；不因 OOM 自动缩小 batch。
- 训练输出统一包含 `stage`、`step`、`batch`、`loss`、`update` 和 `lr`；每个 run 的 `training.log` 持久化同样的进度摘要，中断时追加异常信息，各阶段仍保留结构化 `history.csv`。
- 追迹失败保持失败关闭，由当前 run 的资格筛选进度记录错误；底层追迹不在项目根目录自动导出 `wrong_result` Excel。
- `J=sum(group_weight*group_mean)`，10 组权重显式且和为 1。Far 路由到
  Mannos-Sakrison CSF 加权 MTF loss，corridor/near 路由到 Z4²，near-edge 为
  90% Z4² + 10% near 区 A_D，周边区为对应区平均 A_D；均用 Original PAL 归一化。
- 19×19 阶段额外使用 `smooth_lambda=0.05` 的控制点归一化二阶差分正则。
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

每个条件文件保存 81 个场点的原始 512×512 FFT PSF 和统一物理裁剪后的
130×130 渲染 PSF；weighted MTF 只读取原始 PSF，拼接只读取渲染 PSF。
评价网格为 `[-40,40]` degree、步长 10 degree。`--resume` 精确核验并跳过
HDF5 中已完成节点；损坏或身份不符的节点失败关闭。`--blur-scale` 默认 4，
只控制视标拼接的显示模糊，不改变 PSF 数据库、MTF 或 PSF stitch。
PSF 追迹通过必选的 `raw_psf_batch()` 接口做 CUDA case 小批量并行，
`--psf-batch-size` 默认 8；最后一批按实际剩余场点数运行，OOM 或批量追迹
失败直接终止，不自动缩批也不回退到串行。恢复时仅对未完成场点重新分批，
已完成节点仍逐个核验并跳过。
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
  --max-extra-19-steps 30 `
  --smooth-lambda 0.05
```

该示例最低训练 50 个 attempt，最多训练 80 个 attempt。19×19 每个 attempt
完成后先原子保存 resume/history，再按 early stopping、学习率下限和额外预算
顺序判断；最低预算前若学习率跌破下限，则 run 失败且不写成功 `summary.json`。
最终结果始终加载 19×19 best state 并完成 109-case 无梯度复核后才标记 complete。

普通 `--resume` 只允许恢复同一运行目录中 identity、配置、输入哈希、实现闭包、
最低/最大预算、patience 计数和终止规则完全一致的 checkpoint。本次 early
stopping 改造提升了 run identity 与 stage-resume schema；旧 run/checkpoint
保持只读，不可按新方法恢复，必须使用新输出目录。当前项目不再支持跨平台
training-state/parity 导入。

完整 `candidate-trace` 进度可显式导入新 run：保存的原身份必须先通过自身哈希
校验，镜片内容 SHA-256 与物理追迹参数也必须匹配；Linux/Windows checkout 的
绝对路径不是物理身份字段。forward/final-phase 进度仍须严格匹配 pool identity。

预优化产物还包括 `dense_candidate_fields.json` 与
`dense_candidate_grid_on_lens.png`：前者保存全部物方密集场点经真实 PAL 追迹后
映射到镜片后表面的物理坐标，后者用于与资格池候选点图比较。
`candidate_reachability_on_lens.png` 使用叉号表示 true-traceable、空心圆表示
safety-margin 后 eligible。`corridor_flank` 是 `corridor` 内部的诊断子 mask，
不是独立训练分区或独立 loss；分区分类和训练 case 均纳入 `corridor`。

## 验证

```powershell
python -m pytest tests -q --basetemp .tmp_pytest
```

最小 prepare：

```powershell
python run_pal_nurbs.py --output .tmp_prepare --excel eye_image_glass_grad3.xlsx --device cpu --prepare-only
```

正式实验只有在 `summary.json` 存在且 `run_state.status=complete` 时才算完成。r12 只作为历史未完成证据，不是当前运行的恢复入口或收益结论。
