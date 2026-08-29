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
- 80 个真实 case：Far 18、Intermediate 12、Near 18、Peripheral-left/right 各 16。
- 三个物距统一为 `D500=500 mm`、`D1000=1000 mm`、`Dinf=Infinity`；Infinity 使用真实无穷远条件。
- 使用真实可微追迹、GRIN3 固定步长 RK4、连续 OPL、去 pupil tilt FFT PSF；离线 PSF 不参与反传。
- 默认 pupil 采样为 `np=256`、FFT 为 `512`；80 个 case 按 `case_batch_size=8` 做 GPU tensor 追迹和 FFT，每批聚合一次 loss 并 backward，不因 OOM 自动缩小 batch。
- 训练输出统一包含 `stage`、`step`、`batch`、`loss`、`update` 和 `lr`；每个 run 的 `training.log` 持久化同样的进度摘要，中断时追加异常信息，各阶段仍保留结构化 `history.csv`。
- 追迹失败保持失败关闭，由当前 run 的资格筛选进度记录错误；底层追迹不在项目根目录自动导出 `wrong_result` Excel。
- `J=(0.85*J_functional+0.15*J_peripheral)`：功能区使用逐 case 的
  `M2_mm²/Original PAL M2_mm²`，左右周边区使用各自区域平均
  `A_D/Original PAL A_D`；Original PAL 分母固定。
- 仅使用 trace/PSF health、`P_far`、`ADD` 和单步 sag trust region 约束。
- 不自动运行历史 192-case posthoc、PSF 数据库、渲染或 SSIM 链。

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
  --requested-np 256 --fft-size-px 512 --case-batch-size 8 --steps 10 10 10
```

普通 `--resume` 只允许恢复同一运行目录中 identity、配置、输入哈希和实现闭包完全一致的 checkpoint；本次 batch 接口变化后的旧串行 run 不可恢复，必须使用新输出目录。当前项目不再支持跨平台 training-state/parity 导入。

## 验证

```powershell
python -m pytest tests -q --basetemp .tmp_pytest
```

最小 prepare：

```powershell
python run_pal_nurbs.py --output .tmp_prepare --excel eye_image_glass_grad3.xlsx --device cpu --prepare-only
```

正式实验只有在 `summary.json` 存在且 `run_state.status=complete` 时才算完成。r12 只作为历史未完成证据，不是当前运行的恢复入口或收益结论。
