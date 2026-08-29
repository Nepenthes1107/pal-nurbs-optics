# 当前工程状态

更新时间：2026-08-29。

## 当前主线

项目现在只有稳定 BIOT 和 PAL-NURBS 从零优化主线，不再使用 Phase 编号。统一运行环境为 Python 3.8.20、PyTorch 2.0.1、CUDA runtime 11.8。Windows `.venv` 仅用于本地验证；Linux 云端上传源码后重新创建自己的 `.venv`。

`.venv` 已核验为 Python 3.8.20、PyTorch 2.0.1+cu118，CUDA runtime 11.8 可用，并能识别本机 NVIDIA GeForce RTX 4060。

## 当前目录合同

- 固定输入：`inputs/pal/psf_supports.json`、`inputs/pal/zones.json`。
- 新实验输出：`results/optimization/run_001`、`run_002`，依次递增。
- 历史证据：`results/archive/`，不进入 Git，也不参与新实验身份。
- 当前入口：`run_pal_nurbs.py`；PAL 实现位于 `biot/e2e/pal_nurbs.py` 和 `biot/e2e/pal_case_layout.py`。
- `main` 的 80-case/分阶段架构已接入真实 GPU tensor case batch：默认 `case_batch_size=8`、`requested_np=256`、FFT `512`；每批完成追迹和 FFT 后聚合一次 loss/backward，最后一批按实际数量运行，不做 OOM 自动降级。训练前向直接使用 raw 物理 FFT PSF，不再把 130×130 crop kernel 送入 loss。
- `main` 的分阶段训练输出统一显示 `stage/step/batch/loss/update/lr`，并写入 run 根目录 `training.log`；各阶段 `history.csv` 和 resume checkpoint 继续作为结构化恢复依据。
- `--steps S7 S11 S19` 现定义最低训练预算：7×7/11×11 严格完成各自
  attempt 数，19×19 至少完成 `S19`，随后按默认 patience 7、严格相对改善
  阈值 `1e-4` 继续，最多额外 50 个 attempt。19×19 patience 从第一步累计，
  但最低配额前不允许 early stop；旧的 11×11 整阶段改善门槛已删除。
- 19×19 每次 attempt 后先原子保存 resume/history，再依次判断 early stopping、
  学习率下限和额外预算。最低预算前学习率跌破下限会将 run 标为 failed 且不写
  成功 summary；完成后始终加载 19×19 best state，再做完整 80-case 无梯度复核。
- 新训练合同统一使用 `D500/D1000/Dinf`；已完成的旧 `run_001` 保持历史训练身份，不原位改写。
- `main` 与多物距分支均显式固定 `legacy_pupil_phase=False`、`phase_reference="biot_reference_sphere"`、`remove_tilt=False`；相位/PSF 表示变化已提升 run、case-layout 和 baseline schema，旧 checkpoint 不可按新合同恢复。
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

- 训练 case 数量固定为 18/12/18/16/16，目标权重固定为 0.85/0.15；
  功能区 loss 为逐 case 的归一化 PSF M2，左右周边区 loss 为对应
  M/A 区域平均 A 的 Original PAL 归一化值。
- `main` 训练的 M2、valid-fraction ratio、edge health 和 NURBS gradient 均从 raw 物理 PSF 计算；评价阶段才生成 130×130 render PSF。
- `best_feasible` 必须来自完整覆盖周期且所有工程与健康约束通过。
- 当前为去 tilt、单波长结果，不能外推为色差、棱镜、真实视物位置或几何畸变合格。
- 普通 `--resume` 仅在同一目录 identity、配置、输入哈希、实现闭包及训练预算/
  patience/阈值/计数完全一致时有效。19×19 early-stopping 改造已提升 run
  identity 和 stage-resume schema，旧 run/checkpoint 不可续接。
- 跨平台 training-state、parity fixture、cloud_run 和 migration 导出链已废弃并删除。

## 资源与恢复

- Windows/WDDM 下 PyTorch CUDA allocator inactive blocks 曾造成 host commit 压力。
- 当前实现逐 case 释放图并调用 `torch.cuda.empty_cache()`；GRIN3 使用 activation checkpoint，物理方程和梯度路径不变。
- startup gradient 的中心/边缘 case 会作为 `_retain_training_cache()` 的显式 `extra_cases` 在缓存冻结前物化；不增加额外全局完整性扫描，baseline 后仍仅保留正式训练 case。
- 底层前向追迹失败只抛出异常，不再向项目根目录自动写入 `optimization/wrong_result`；PAL 候选失败由对应 run 的资格筛选进度文件记录。
- 长任务必须持续写 checkpoint、history、run state 和退出信息；异常后从最近可验证 checkpoint 恢复，不降低正式采样或门槛。
- 孔径判定与 BIOT_vis 对齐：圆孔径采用 `2*r*300e-6+(300e-6)^2` 的 SDF 容差，方孔径采用 `300e-6 mm` 线性容差；真超口径仍失败关闭。

## 验证入口

```powershell
python -m pytest tests -q --basetemp .tmp_pytest
python run_pal_nurbs.py --output .tmp_prepare --excel eye_image_glass_grad3.xlsx --device cpu --prepare-only
```

2026-08-28 验收：PAL-NURBS 定向测试 `21 passed`，完整测试 `154 passed`，无跳过；真实 2-case CPU batch smoke 的 kernel shape、能量归一化和 PAL 梯度检查通过。GUI smoke 使用 `pytest-qt==4.4.0`。PyTorch 2.0.1 在 Windows 中文路径下的 checkpoint 原子写入改用 Python 二进制文件句柄，仍采用原 torch 序列化、`fsync` 和原子替换。

2026-08-29 验收：周边 A 目标及统一 HDF5 PSF 数据库/评价链的 `.venv\Scripts\python.exe -m pytest tests -q --basetemp .tmp_pytest_main_full` 为 `165 passed`，无失败；其中评价器定向测试为 `7 passed`。本次未启动正式优化或完整 486-field PSF 数据库生成。

2026-08-29 CUDA 评价批量化验收：评价器与 PAL 定向测试 `32 passed`；此前批量接口基线为 `168 passed`。本次统一 raw PSF/非 legacy 相位合同后，`.venv\Scripts\python.exe -m pytest tests -q --basetemp .tmp_main_raw_final` 为 `169 passed`，无失败；新增测试覆盖固定相位配置、raw 512×512 训练 PSF 和 identity/schema 边界。本次未启动正式优化或完整 486-field 评价。

2026-08-29 main 云端启动修复：Infinity 物距的训练 case ID 不再将 `inf` 转换为整数，统一序列化为 `Dinf`；case-layout state schema 提升为 6，旧布局进度不复用。针对性布局测试通过；未执行完整物理计算。

2026-08-29 main 云端启动二次修复：case-layout 几何审计现在仅允许 far/upper 使用真实正 Infinity，intermediate/near 与 middle/lower 仍必须为有限正物距；不使用大有限距离近似。PAL 定向回归 `37 passed`，完整测试 `171 passed`，无失败；未启动正式训练。

2026-08-29 main 评价清理修复：`MinimalOpticalModel.close()` 现在显式释放缓存系统、模板持有的 BIOT lens 引用和处方上下文，修复六条件 PSF 数据库在单条件完成后的清理阶段报错；PAL 定向测试 `25 passed`、评价器测试 `9 passed`、完整测试 `172 passed`，无失败；未启动正式评价。

2026-08-29 main 训练预算/early-stopping 改造验收：`--steps` 改为 7×7/11×11
固定配额与 19×19 最低配额，新增 patience、严格相对改善阈值和 19×19 额外
attempt 上限；run identity schema 提升为 6、stage-resume schema 提升为 2。
PAL 定向测试 `40 passed`，完整测试 `187 passed`，无失败；`py_compile` 与
`git diff --check` 通过。本次未启动正式训练，不形成实验收益结论。
