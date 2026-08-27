# 当前工程状态

更新时间：2026-08-26。

## 当前主线

项目现在只有稳定 BIOT 和 PAL-NURBS 从零优化主线，不再使用 Phase 编号。统一运行环境为 Python 3.8.20、PyTorch 2.0.1、CUDA runtime 11.8。Windows `.venv` 仅用于本地验证；Linux 云端上传源码后重新创建自己的 `.venv`。

当前 PAL 新方法在开发分支 `codex/pal-multi-distance-second-moment` 上实现，基于干净基线提交 `146111e chore: establish clean PAL-NURBS baseline`。该分支的正式 run identity 与基线不同，不能复用历史 checkpoint。

`.venv` 已核验为 Python 3.8.20、PyTorch 2.0.1+cu118，CUDA runtime 11.8 可用，并能识别本机 NVIDIA GeForce RTX 4060。

## 当前目录合同

- 固定输入：`inputs/pal/zones.json`、`inputs/pal/multidistance_weights.json`。
- 新实验输出：`results/optimization/run_001`、`run_002`，依次递增。
- 历史证据：`results/archive/`，不进入 Git，也不参与新实验身份。
- 当前入口：`run_pal_nurbs.py`；PAL 实现位于 `biot/e2e/pal_nurbs.py` 和 `biot/e2e/pal_case_layout.py`。

## 当前 PAL 多物距方法合同

- 参数化后表面为固定 cubic NURBS `11×11` 控制网格，外层控制环为零；不再存在 7×7→11×11→19×19 阶梯或 refinement audit。
- 物距块严格为 `D500=500 mm`、`D2000=2000 mm`、`Dinf=Infinity`，每块复用同一个 `11×11` FOV 网格，总 case 数为 363。
- case 使用真实可微光线追迹和去 pupil tilt 的 raw 物理 FFT PSF；loss 是能量归一化 PSF 二阶矩 `mm²` 的分区/物距加权和。禁止 PSF crop、resize、插值、滤波和离线 PSF 反传。
- 分区分类以 `zones.json` 的存储 mask 为主；monitored 内未标注单元使用记录在 case metadata 中的最近分区规则，任何固定点都不能静默丢弃。
- 权重定义在 `inputs/pal/multidistance_weights.json`，分区总质量及每个分区的物距分配均机器可读；15 个“分区×物距”组合必须都有 case，展开后的 `objective_weight` 总和必须为 1，禁止缺项后重新归一化。
- `best_feasible` 只能来自完整 363-case sweep，并同时满足 PSF health、`P_far`、`ADD`、控制量和单步 sag 约束。

## 历史 r12

- 路径：`results/archive/r12_incomplete`
- identity：`9e30366cd353f39b63782aafceabacdb2b2498d8202123aff54a793b2d72c92d`
- 7x7 完成 `10/10`；11x11 完成 `4/10`。
- 该 run 未完成，无最终收益结论；仅保留为历史证据，不再支持跨平台导入或续跑。
- 外置 Git 恢复备份仍保留在 `D:\VSCODE\端到端光学设计_git_backup_20260826.tar.zst`。

## 有效合同与限制

- 当前训练 case 数量固定为 `3×121=363`，不是历史 80-case 合同。
- `best_feasible` 必须来自完整覆盖周期且所有工程与健康约束通过。
- 当前为去 tilt、单波长结果，不能外推为色差、棱镜、真实视物位置或几何畸变合格。
- 普通 `--resume` 仅在同一目录 identity、配置、输入哈希和实现闭包完全一致时有效。
- 跨平台 training-state、parity fixture、cloud_run 和 migration 导出链已废弃并删除。

## 资源与恢复

- Windows/WDDM 下 PyTorch CUDA allocator inactive blocks 曾造成 host commit 压力。
- 当前实现逐 case 释放图并调用 `torch.cuda.empty_cache()`；GRIN3 使用 activation checkpoint，物理方程和梯度路径不变。
- 长任务必须持续写 checkpoint、history、run state 和退出信息；异常后从最近可验证 checkpoint 恢复，不降低正式采样或门槛。

## 验证入口

```powershell
python -m pytest tests -q --basetemp .tmp_pytest
python run_pal_nurbs.py --output .tmp_prepare_multidistance --excel eye_image_glass_grad3.xlsx --device cpu --requested-np 32 --fft-size-px 64 --prepare-only
```

2026-08-26 验证记录：本分支 `.venv\Scripts\python.exe -m pytest tests -q --basetemp .tmp_pytest_final_identity` 为 `131 passed`，无失败；当前代码身份下的 prepare-only smoke 为 `363` cases、状态 `prepared`，并通过同目录 `--resume` 的布局哈希校验。旧基线的 `150 passed` 仍是基线参考，不与本分支测试数量混用。PyTorch 2.0.1 在 Windows 中文路径下的 checkpoint 原子写入改用 Python 二进制文件句柄，仍采用原 torch 序列化、`fsync` 和原子替换。

缩小 CPU 物理 sweep smoke（requested pupil 8、FFT 32、steps 0）进入 `baseline_psf_sweep` 后因单 case 成本过高主动中断；中断前已写入 `.tmp_run_multidistance_zero/baseline_progress.pt` 的 17/363 行。该临时 run 未作为结果或科学结论使用。
