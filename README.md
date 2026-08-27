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
- 使用真实可微追迹、GRIN3 固定步长 RK4、连续 OPL、去 pupil tilt FFT PSF；离线 PSF 不参与反传。
- 追迹失败保持失败关闭，由当前 run 的资格筛选进度记录错误；底层追迹不在项目根目录自动导出 `wrong_result` Excel。
- `J=(0.85*J_functional+0.15*J_peripheral)`，Original PAL 分母固定。
- 仅使用 trace/PSF health、`P_far`、`ADD` 和单步 sag trust region 约束。
- 不自动运行历史 192-case posthoc、PSF 数据库、渲染或 SSIM 链。

## 从零开始运行

```powershell
python run_pal_nurbs.py `
  --output results/optimization/run_001 `
  --excel eye_image_glass_grad3.xlsx --device cuda `
  --requested-np 1024 --fft-size-px 512 --steps 10 10 10
```

普通 `--resume` 只允许恢复同一运行目录中 identity、配置、输入哈希和实现闭包完全一致的 checkpoint。当前项目不再支持跨平台 training-state/parity 导入。

## 验证

```powershell
python -m pytest tests -q --basetemp .tmp_pytest
```

最小 prepare：

```powershell
python run_pal_nurbs.py --output .tmp_prepare --excel eye_image_glass_grad3.xlsx --device cpu --prepare-only
```

正式实验只有在 `summary.json` 存在且 `run_state.status=complete` 时才算完成。r12 只作为历史未完成证据，不是当前运行的恢复入口或收益结论。
