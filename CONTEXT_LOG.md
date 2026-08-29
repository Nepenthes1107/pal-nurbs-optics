# 当前工程状态

更新时间：2026-08-27。

## 当前主线

项目现在只有稳定 BIOT 和 PAL-NURBS 从零优化主线，不再使用 Phase 编号。统一运行环境为 Python 3.8.20、PyTorch 2.0.1、CUDA runtime 11.8。Windows `.venv` 仅用于本地验证；Linux 云端上传源码后重新创建自己的 `.venv`。

`.venv` 已核验为 Python 3.8.20、PyTorch 2.0.1+cu118，CUDA runtime 11.8 可用，并能识别本机 NVIDIA GeForce RTX 4060。

## 当前目录合同

- 固定输入：`inputs/pal/psf_supports.json`、`inputs/pal/zones.json`。
- 新实验输出：`results/optimization/run_001`、`run_002`，依次递增。
- 历史证据：`results/archive/`，不进入 Git，也不参与新实验身份。
- 当前入口：`run_pal_nurbs.py`；PAL 实现位于 `biot/e2e/pal_nurbs.py` 和 `biot/e2e/pal_case_layout.py`。
- `main` 的 80-case/分阶段架构已接入真实 GPU tensor case batch：默认 `case_batch_size=8`、`requested_np=256`、FFT `512`；每批完成追迹和 FFT 后聚合一次 loss/backward，最后一批按实际数量运行，不做 OOM 自动降级。
- `main` 的分阶段训练输出统一显示 `stage/step/batch/loss/update/lr`，并写入 run 根目录 `training.log`；各阶段 `history.csv` 和 resume checkpoint 继续作为结构化恢复依据。
- 新训练合同统一使用 `D500/D1000/Dinf`；已完成的旧 `run_001` 保持历史训练身份，不原位改写。
- `evaluate_pal_nurbs.py` 在 `<run>/evaluation` 生成独立评价身份；三物距×双状态
  分别保存为六个 HDF5，每个文件含 81 个场点的原始 FFT PSF、130×130 渲染
  PSF及最小恢复信息。只有数据库整体 `complete` 后才运行 weighted-MTF Mean、
  PSF stitch 和 chart stitch；chart 的 `blur-scale` 默认 4且仅影响显示。

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
- `best_feasible` 必须来自完整覆盖周期且所有工程与健康约束通过。
- 当前为去 tilt、单波长结果，不能外推为色差、棱镜、真实视物位置或几何畸变合格。
- 普通 `--resume` 仅在同一目录 identity、配置、输入哈希和实现闭包完全一致时有效。
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
