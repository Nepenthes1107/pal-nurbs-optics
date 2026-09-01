# 当前工程状态

更新时间：2026-09-01。

## 当前主线

项目现在只有稳定 BIOT 和 PAL-NURBS 从零优化主线，不再使用 Phase 编号。统一运行环境为 Python 3.8.20、PyTorch 2.0.1、CUDA runtime 11.8。Windows `.venv` 仅用于本地验证；Linux 云端上传源码后重新创建自己的 `.venv`。

`.venv` 已核验为 Python 3.8.20、PyTorch 2.0.1+cu118，CUDA runtime 11.8 可用，并能识别本机 NVIDIA GeForce RTX 4060。

## 当前目录合同

- 固定输入：`inputs/pal/psf_supports.json`、`inputs/pal/zones.json`。
- 新实验输出：`results/optimization/run_001`、`run_002`，依次递增。
- 历史证据：`results/archive/`，不进入 Git，也不参与新实验身份。
- 当前入口：`run_pal_nurbs.py`；PAL 实现位于 `biot/e2e/pal_nurbs.py` 和 `biot/e2e/pal_case_layout.py`。
- 当前分支将训练布局提升为 10 组 109 case；其中 79 个功能 case 使用
  GPU tensor 真实追迹，30 个周边 case 仅用可微面形 A_D，不进入光线追迹。
  默认 `case_batch_size=8`、`requested_np=256`、FFT `512`，不做 OOM 自动降级。
- `main` 的分阶段训练输出统一显示 `stage/step/batch/loss/update/lr`，并写入 run 根目录 `training.log`；各阶段 `history.csv` 和 resume checkpoint 继续作为结构化恢复依据。
- `--steps S7 S11 S19` 定义三个阶段的最低训练预算，最后一个非零阶段为 terminal
  stage；此前阶段严格完成固定 attempt 数，terminal 才使用 patience、严格相对
  改善阈值和 `--max-extra-terminal-stage-steps`。例如 `50/10/0` 在 11×11 上
  追加 extra/early-stop，19×19 只做精确 refinement。terminal patience 从第一步
  累计，但最低配额前不允许 early stop。
- terminal 每次 attempt 后先原子保存 resume/history，再依次判断 early stopping、
  学习率下限和额外预算。最低预算前学习率跌破下限会将 run 标为 failed 且不写
  成功 summary；完成后后续零预算阶段只做精确 refinement，再以最终 19×19 表示
  做完整 109-case 无梯度复核。summary 显式记录 terminal control count、extra
  attempt 数和 terminal 停止原因。
- 已完成父 run 可通过 `--parent-run` 和 `--start-stage 7|11|19` 分叉为新
  child identity；合法起点必须不早于父 terminal。child `--steps` 只统计新增
  attempt，使用父阶段 best 但重置 Adam/学习率/patience。父 candidate trace、
  forward/final-phase qualification、baseline 及从 terminal 到起点的逐级
  zero-budget refinement 必须通过身份、checkpoint 和 `1e-10` 面形/导数审计；
  不存在重算或导入父 Adam 的兜底。父目录保持只读，child 只能在自身
  目录 `--resume`。新 child 使用 run identity schema 9；已完成 schema 8
  run 只获得父源导入兼容，不获得原目录新 schema resume 兼容。
- 新训练合同统一使用 `D500/D1000/Dinf`；已完成的旧 `run_001` 保持历史训练身份，不原位改写。
- `main` 与多物距分支均显式固定 `legacy_pupil_phase=False`、`phase_reference="biot_reference_sphere"`、`remove_tilt=False`；相位/PSF 表示变化已提升 run、case-layout 和 baseline schema，旧 checkpoint 不可按新合同恢复。
- `evaluate_pal_nurbs.py` 在 `<run>/evaluation` 生成独立评价身份；三物距×双状态
  分别保存为六个 HDF5，每个文件含 81 个场点的原始 FFT PSF、130×130 渲染
  PSF及最小恢复信息。只有数据库整体 `complete` 后才运行 weighted-MTF Mean、
  PSF stitch 和 chart stitch；chart 的 `blur-scale` 默认 4且仅影响显示。
- PSF 数据库默认通过 `raw_psf_batch()` 以 `psf_batch_size=8` 做原生 FFT PSF case 批量追迹；批大小纳入评价 identity。已完成 HDF5 节点仍逐个核验，未完成节点按小批量恢复；不自动缩批或串行回退。
- `evaluate_pal_nurbs.py` 的控制台进度统一使用 `[pal-eval]`，显示当前阶段、条件、
  PSF batch、条件内及全局场点完成数；恢复时显式报告已跳过的完整条件/阶段。
- `weighted_mtf/` 保留 9×9 原始 mean map 和 NPZ，并为三物距×
  baseline/optimized/delta 额外生成 9 张 `*_mean_interpolated.png`。
  插值是仅显示的规则网格 cubic 200×200 上采样，严格限定在原生
  `[-40,40]°` 域内、原节点误差不超过 `1e-12`；非有限或超出
  MTF 物理范围的原生数据直接失败，不填洞或外推。
- 完整 run 可用 `evaluate_pal_nurbs.py --checkpoint-stage 7|11|19` 直接评价对应
  已完成阶段的 `final.pt`；评价器严格核对 summary 阶段记录、checkpoint
  `control_count` 与源 identity，并写入独立的 `evaluation_stage_NxN/`。

## 历史 r12

- 路径：`results/archive/r12_incomplete`
- identity：`9e30366cd353f39b63782aafceabacdb2b2498d8202123aff54a793b2d72c92d`
- 7x7 完成 `10/10`；11x11 完成 `4/10`。
- 该 run 未完成，无最终收益结论；仅保留为历史证据，不再支持跨平台导入或续跑。
- 外置 Git 恢复备份仍保留在 `D:\VSCODE\端到端光学设计_git_backup_20260826.tar.zst`。

## 有效合同与限制

- 训练 case 数量固定为 20/8/5/5/5/20/8/8/15/15，显式分组权重和为 1。
  Far 使用 CSF-MTF，corridor/near 使用连续 OPD 拟合的 OSA/ANSI Z4²，
  near-edge 小权重合入 near A_D，周边使用面形 A_D。
- Z4 执行层使用 Torch reduced-QR/三角求解保持 autograd；BIOT NumPy 拟合仅用于
  detached 对比验证，不从 PSF 逆变换恢复相位，也不保留 M2 训练回退。
- 处方门禁除 `P_far`/`ADD` 外，对 lower-edge guard 检查 candidate 相对
  Original PAL 的最大光焦度/像散变化；19×19 阶段使用归一化二阶差分正则。
- `best_feasible` 必须来自完整覆盖周期且所有工程与健康约束通过。
- 当前为去 tilt、单波长结果，不能外推为色差、棱镜、真实视物位置或几何畸变合格。
- 普通 `--resume` 仅在同一目录 identity、配置、输入哈希、实现闭包及训练预算/
  patience/阈值/计数完全一致时有效。terminal-stage 额外预算合同已将 run
  identity schema 提升为 8、stage-resume schema 提升为 3，旧 run/checkpoint
  不可续接。
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

2026-08-30 V3 链路改造验收：109-case 分层布局、CSF-MTF/Z4/A_D 指标路由、
连续 OPD Torch Zernike 拟合、周边免追迹、19×19 二阶差分正则和 lower-edge
baseline-relative 处方门禁已接入。合成六模 Zernike、Z4² AD/FD 与真实
corridor/near PAL 连续 OPD 对 BIOT NumPy 拟合的直接验证通过；完整测试
`.venv\Scripts\python.exe -m pytest -q tests --basetemp=.pytest_tmp_v3_full` 为
`189 passed`，py_compile 与 `git diff --check` 通过。本次未启动 V3 完整训练，
不形成优化收益或运行时长结论。

2026-08-31 V3 资格池最终 FPS 串组修复：420-case WFNO 池排除失败项后，
最终 FPS 现在仅能在各自 `training_group` 的已合格源记录中选择，不再将
共享物理分区的 `far/far_robustness` 或 `near/near_robustness/near_edge_astig`
混合采样。该错误曾使 `far` 最终 case 继承 robustness 的 `D1000`，并被稳定键
成员身份门禁正确拒绝。case-layout state schema 提升为 8；旧失败 run 不得直接
`--resume`，但可在新 run identity 中显式导入身份匹配的完整 candidate-trace 和
forward-qualification 进度。完整测试
`.venv\Scripts\python.exe -m pytest tests -q --basetemp .tmp_pytest_group_pool_all` 为
`190 passed`，无失败；本次未启动正式训练，不形成科学收益结论。

2026-08-31 V3 合格池成员审计修复：最终相位门禁经多轮 FPS 排除后，
preoptimization 审计不再错误要求组别合格池的 `candidate_id` 全局唯一。物理源候选
仍以 candidate ID、视场和后表面坐标严格唯一；组别资格记录以
`training_group/candidate_id/distance/field` 稳定键唯一。允许同一物理候选被
多个目标组合法复用，但同 ID 的物理坐标冲突仍失败关闭。覆盖率审计先按物理源
去重，防止跨组复用人为改变候选间距统计；组别资格不同时按至少一个组仍合格
保留该物理覆盖点。case-layout state schema 提升为 9，
case manifest/candidate-fields/coverage schema 分别提升为 7/2/4。PAL 定向测试
`57 passed`；完整测试
`.venv\Scripts\python.exe -m pytest tests -q --basetemp .tmp_pytest_membership_pool_all` 为
`193 passed`，无失败。本次未启动正式训练，不形成科学收益结论。

2026-08-31 V3 最终覆盖选择与跨平台 candidate-trace 导入修复：最终 qualified
pool 不再用普通 FPS 后等待 coverage audit 才报错；在各组数量、已有覆盖公式和
门槛完全不变的条件下，far/far-robustness 使用组内确定性交换，周边 upper/middle/
lower 使用固定镜像对数量的确定性精确组合搜索，找不到合格子集时仍失败关闭。
final-phase 历史状态优先按 `training_group/candidate_id/distance/field` 稳定键匹配，
避免多轮选择重新编号的 `case_id` 串接错误。candidate-trace 导入先验证原 identity
自哈希，再按镜片 SHA-256 和物理参数匹配；仅忽略 Linux/Windows checkout 的绝对
Excel 路径。case-layout state schema 提升为 10。

本地证据 `results/optimization/v3_branch/local_validation_run_002` 使用 run_003 的
三份完整进度、真实 Infinity、requested_np 256、FFT 512、batch 8 和验证预算
0/1/0；prepare 在 2626.36 s 后完成。manifest 为 109 cases/10 组，qualified-pool
membership 通过，coverage `overall_passed=true` 且失败门禁为空，phase 进度累计
235 次尝试。随后 startup autograd gradient check 通过，Original PAL baseline
完成 2 个 GPU batch、保存 16/109 rows；按用户要求在继续训练前停止，目标进程已
确认不存在。`run_state.json` 因 Ctrl-C 仍为陈旧 `running/baseline_training_cases`，
不能作为活进程或完成训练的证据；本次不形成优化收益结论。

2026-08-31 V3 分区图例颜色修复：`partition_map.png` 的分区编号包含 6 个 zone，
原绘图调色板只提供 5 个分区颜色，导致最后一个 `corridor_flank` 编号被 matplotlib
截为 peripheral-right 橙色。绘图器现按 `PARTITION_ORDER` 从 `ZONE_COLORS` 生成
“背景 + 6 分区”完整调色板；已用 `run_004/preoptimization` 重生成分区图及相关图，
并新增颜色表长度/顺序回归测试。该修复只影响可视化，不改变 zones mask、候选点或训练
case 数值数据。

2026-08-31 V3 预优化密集物理网格图：新增 `dense_candidate_fields.json` 与
`dense_candidate_grid_on_lens.png`，显示全部物方 1° 密集场点经 Original PAL
中心瞳孔光线追迹后的后表面物理坐标；与 qualified candidate 图分开，避免把 364
个资格池点误认为 10,556 个密集场点。`candidate_reachability_on_lens.png` 中
true-traceable 改为叉号、post-qualification eligible 改为空心圆。`corridor_flank` 明确标为 corridor
的诊断子 mask，不再作为独立训练分区。

2026-08-31 V3 候选资格契约更新：删除 zone-boundary 和 aperture-edge clearance
安全筛选。候选只要 Original PAL 真实追迹成功且能归入声明分区即为 eligible；后续
forward WFNO、phase qualification、覆盖门禁和最终 case 几何/距离校验仍保持硬失败。
历史 clearance 参数仅为旧配置和调用兼容保留，不再进入 candidate eligibility 或 coverage
定义；方法名和 run identity schema 已更新，旧 candidate-trace 进度不能直接恢复。

2026-08-31 PAL 评价进度输出验收：`evaluate_pal_nurbs.py` 现在按阶段、六个条件和
GPU PSF batch 即时刷新 `[pal-eval]` 控制台进度，显示条件内与全局场点完成数；
`--resume` 会显式报告完整条件/阶段的 `status=SKIP`。评价器定向测试 `9 passed`，
完整测试 `.venv\Scripts\python.exe -m pytest tests -q --basetemp
.tmp_pytest_eval_progress_full` 为 `198 passed`；`py_compile` 与 `git diff --check`
通过。本次未运行正式评价，不形成新的光学或性能结论。

2026-08-31 PAL terminal-stage 预算合同验收：额外预算不再固定属于 19×19，而是
绑定到 `--steps` 最后一个非零训练阶段；CLI 更名为
`--max-extra-terminal-stage-steps`，不保留旧参数别名。`50/10/0` 因此在 11×11
执行最低 10 个及最多额外指定数量的 attempt，19×19 只做精确 refinement。
run identity schema 提升为 8、stage-resume schema 提升为 3，旧 checkpoint 不可
续接。PAL 定向测试 `43 passed`，完整测试
`.venv\Scripts\python.exe -m pytest tests -q --basetemp .tmp_pytest_terminal_stage_full`
为 `199 passed`；`py_compile` 与 `git diff --check` 通过。本次未启动正式训练，
不形成新的优化收益结论。

2026-09-01 PAL 阶段 checkpoint 独立评价验收：完整 run 可通过
`evaluate_pal_nurbs.py --checkpoint-stage 7|11|19` 直接读取对应阶段的
`final.pt`，无需重跑该阶段训练；阶段评价使用独立 `evaluation_stage_NxN/`
身份，并校验 summary 阶段记录、checkpoint `control_count`、源 run identity
和 checkpoint SHA-256。评价器定向测试 `10 passed`，完整测试
`.venv\Scripts\python.exe -m pytest tests -q --basetemp .tmp_pytest_eval_stage_full`
为 `200 passed`；`py_compile` 与 `git diff --check` 通过。本次未运行正式评价，
不形成新的光学或性能结论。

2026-09-01 PAL 父阶段分叉续训验收：`run_pal_nurbs.py` 新增
`--parent-run`/`--start-stage`，实现完成父 run 只读验证、合法起点矩阵、
父 best + fresh Adam 的 child 初始化、逐级零预算精确 refinement 审计、四类
预处理/baseline 证据复用、child-only 步数与累计 lineage 记录。定向
PAL 测试 `58 passed`；完整测试
`.venv\Scripts\python.exe -m pytest tests -q --basetemp .tmp_pytest_parent_fork_full`
为 `215 passed`，无失败；`py_compile`、CLI `--help` 与 `git diff --check` 通过。
本次只做代码/合成测试验收，未启动正式续训，不形成优化收益或
运行时长结论。

2026-09-01 PAL weighted-MTF 插值显示图验收：评价链在原有三物距×
baseline/optimized/delta 的 9×9 PNG/NPZ 之外，新增 9 张 cubic
200×200 `*_mean_interpolated.png`。显示插值严核原节点、边界、finite 与
MTF `[0,1]` 物理范围，delta 由分别插值后的 optimized-baseline 得到；
原生 NPZ 不改写。使用已有 `run_004` D500 optimized NPZ 完成一张真实
渲染方向/样式检查。评价器定向测试 `13 passed`；完整测试
`.venv\Scripts\python.exe -m pytest tests -q --basetemp
.tmp_pytest_weighted_mtf_interpolated_full` 为 `218 passed`，无失败；
`py_compile` 与 `git diff --check` 通过。本次未重跑正式评价，不新增
光学或性能结论。
