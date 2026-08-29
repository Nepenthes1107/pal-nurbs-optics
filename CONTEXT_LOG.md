# 当前工程状态

更新时间：2026-08-28。

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

- 参数化后表面为固定权重 cubic B-spline `7×7` 控制网格，外圈控制环为零，仅内部 `5×5=25` 个控制量训练；不再存在 7×7→11×11→19×19 阶梯或 refinement audit。
- 物距块严格为 `D500=500 mm`、`D1000=1000 mm`、`Dinf=Infinity`，每块复用 `(-55:11:55)` 的 `11×11` FOV 网格，总 case 数为 363。
- case 使用真实可微光线追迹、`legacy_pupil_phase=False` 的 `biot_reference_sphere` 参考和 `remove_tilt=False` 的 raw 物理 FFT PSF；far/corridor/near 的 loss 为 baseline-normalized PSF 二阶矩，astig-left/right 的 loss 为 baseline-normalized M/A 像散量 A。禁止训练路径使用 PSF crop、resize、插值或滤波。
- 分区分类以 `zones.json` 的存储 mask 为主；monitored 内未标注单元使用记录在 case metadata 中的最近分区规则，任何固定点都不能静默丢弃。
- 权重定义在 `inputs/pal/multidistance_weights.json`，联合权重总和严格为 1；far-D500 和 near-Dinf 为严格零权重，case 保留用于 baseline/validation 但不反传。
- 默认 `case_batch_size=8`：每批执行 `[B,N,3]` 真实张量追迹、`[B,P,P]` FFT PSF、批 loss 聚合和一次 backward；下一批继续在 PAL 参数上累积梯度，完整 363-case sweep 后才执行一次 optimizer step。最后一批使用实际 case 数；OOM 直接失败，不自动缩批。
- 优化预算为最多 50 个 accepted steps，默认 patience=7、相对改善阈值=1e-4；拒绝步同时恢复 PAL 参数与 Adam optimizer state。
- `best_feasible` 只能来自完整 363-case sweep，并同时满足 PSF health、`P_far`、`ADD`、控制量和单步 sag 约束。
- `evaluate_pal_nurbs.py` 在 `<run>/evaluation` 生成独立评价身份；三物距×双状态
  分别保存为六个 HDF5，每个文件含 81 个场点的原始 FFT PSF、130×130 渲染
  PSF及最小恢复信息。只有数据库整体 `complete` 后才运行 weighted-MTF Mean、
  PSF stitch 和 chart stitch；chart 的 `blur-scale` 默认 4且仅影响显示。
- PSF 数据库默认通过 `raw_psf_batch()` 以 `psf_batch_size=8` 做原生 FFT PSF case 批量追迹；批大小纳入评价 identity。已完成 HDF5 节点仍逐个核验，未完成节点按小批量恢复；不自动缩批或串行回退。

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
- 普通 `--resume` 仅在同一目录 identity、配置、输入哈希和实现闭包完全一致时有效。当前批处理 method identity 及 run/evaluation/training schema 已更新，旧逐 case run 不兼容。
- 跨平台 training-state、parity fixture、cloud_run 和 migration 导出链已废弃并删除。

## 资源与恢复

- Windows/WDDM 下 PyTorch CUDA allocator inactive blocks 曾造成 host commit 压力。
- 当前实现按完整 case batch 释放图并清理 inactive CUDA cache。普通 B-spline 面使用收敛交点的隐函数梯度，GRIN3 使用 activation checkpoint；GRIN 反向按固定 pupil-ray chunk 重算，但每个 chunk 始终包含整个 case batch，且步长/最大步数由每个 case 的完整光线集合预先确定，不改变单 case 数值轨迹。
- 长任务必须持续写 checkpoint、history、run state 和退出信息；异常后从最近可验证 checkpoint 恢复，不降低正式采样或门槛。
- 训练进度同时写入 run 根目录 `training.log`：记录 baseline、resume、每次 attempt 的 accepted/max steps、loss、更新接受/拒绝原因、学习率、early-stopping patience 和中断原因；`history.csv` 仍保存结构化逐 attempt 记录。
- 孔径判定与 BIOT_vis 已验证合同一致：圆孔径使用 `2*r*300e-6+(300e-6)^2` 的 SDF 容差，方孔径使用 `300e-6 mm` 线性容差；评价入口为 `evaluate_pal_nurbs.py`，结果写入 run-owned `evaluation/`。

## 验证入口

```powershell
python -m pytest tests -q --basetemp .tmp_pytest
python run_pal_nurbs.py --output .tmp_prepare_multidistance --excel eye_image_glass_grad3.xlsx --device cpu --requested-np 32 --fft-size-px 64 --prepare-only
```

2026-08-26 验证记录：本分支 `.venv\Scripts\python.exe -m pytest tests -q --basetemp .tmp_pytest_final_identity` 为 `131 passed`，无失败；当前代码身份下的 prepare-only smoke 为 `363` cases、状态 `prepared`，并通过同目录 `--resume` 的布局哈希校验。旧基线的 `150 passed` 仍是基线参考，不与本分支测试数量混用。PyTorch 2.0.1 在 Windows 中文路径下的 checkpoint 原子写入改用 Python 二进制文件句柄，仍采用原 torch 序列化、`fsync` 和原子替换。

2026-08-29 验证记录：统一 HDF5 PSF 数据库与评价链的定向测试为 `7 passed`；`.venv\Scripts\python.exe -m pytest tests -q --basetemp .tmp_pytest_multi_full` 为 `151 passed`，无失败。本次未启动正式优化或完整 486-field PSF 数据库生成。

2026-08-29 CUDA 评价批量化验收：多物距分支定向测试与原生 `field_batch()` 适配为 `23 passed`；完整测试为 `154 passed`，无失败。评价批量仅复用该分支已验证的原生 PSF 批量追迹，不改变多物距的去 tilt 和隐式求交合同。本次未启动正式优化或完整 486-field 评价。

缩小 CPU 物理 sweep smoke（requested pupil 8、FFT 32、steps 0）进入 `baseline_psf_sweep` 后因单 case 成本过高主动中断；中断前已写入 `.tmp_run_multidistance_zero/baseline_progress.pt` 的 17/363 行。该临时 run 未作为结果或科学结论使用。
