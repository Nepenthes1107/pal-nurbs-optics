# BIOT 与 PAL-NURBS 光学优化

本目录是唯一开发与运行项目。后续上传到云服务器时只上传源码、测试、Excel、`requirements.txt` 和必要历史证据；`.venv` 是 Windows 本地环境，不能上传到 Linux。

## 目录结构

```text
biot/                  BIOT 与 PAL-NURBS 实现
inputs/pal/            PAL 固定输入（分区和多物距权重）
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

- 仅优化 PAL 后表面的固定 cubic NURBS `zp` 控制量；`xp/yp/weight` 固定，外层控制环固定为零。参数网格为单一 `11×11`，实际可训练的内层为 `9×9`。
- 三个物距分别为 `D500=500 mm`、`D2000=2000 mm` 和 `Dinf=Infinity`。每个物距使用完全相同的 `11×11` FOV 角度网格，总计 `3×121=363` 个真实 case；无有限距离近似 Infinity。
- case 布局使用基线 PAL 后表面的 chief/reference ray 落点映射到 `inputs/pal/zones.json`。固定网格点不因稀疏、FPS、WFNO 资格或覆盖率筛选而删除；落在 monitored 安全带但未标注分区的点按显式最近分区规则分类，并记录分类模式和距离。
- 每个 case 依次执行真实可微光线追迹、连续 OPL、去 pupil tilt 的 raw 物理 FFT PSF，再计算能量归一化 PSF 的二阶矩（`mm²`）。PSF 不做 crop、resize、插值、滤波或显示增强，离线 PSF 也不参与反传。
- loss 为所有 case 的 `objective_weight × PSF_second_moment_mm2` 之和。分区总质量和分区内三物距分配由 `inputs/pal/multidistance_weights.json` 指定，15 个“分区×物距”组合必须都有 case，展开后权重总和严格为 1，不做缺项后的重新归一化。
- `best_feasible` 只在完整 363-case sweep 且健康、`P_far`、`ADD`、控制量和单步 sag 约束均通过时更新；不使用局部 case、`best_image` 或 `latest` 兜底。
- 旧的候选点选择、80-case 目标、PSF support 数据库、7×7→11×11→19×19 晋级和精确 refinement audit 不属于当前方法。

## 从零开始运行

```powershell
python run_pal_nurbs.py `
  --output results/optimization/run_001 `
  --excel eye_image_glass_grad3.xlsx --device cuda `
  --weights-json inputs/pal/multidistance_weights.json `
  --requested-np 1024 --fft-size-px 512 --steps 10
```

普通 `--resume` 只允许恢复同一运行目录中 identity、配置、输入哈希和实现闭包完全一致的 checkpoint。当前项目不再支持跨平台 training-state/parity 导入。

## 验证

```powershell
python -m pytest tests -q --basetemp .tmp_pytest
```

最小 prepare：

```powershell
python run_pal_nurbs.py --output .tmp_prepare_multidistance --excel eye_image_glass_grad3.xlsx --device cpu --requested-np 32 --fft-size-px 64 --prepare-only
```

正式实验只有在 `summary.json` 存在且 `run_state.status=complete` 时才算完成。r12 只作为历史未完成证据，不是当前运行的恢复入口或收益结论。
