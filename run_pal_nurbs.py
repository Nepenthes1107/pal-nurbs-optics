from __future__ import annotations

import argparse

from biot.e2e.pal_nurbs import (
    MinimalConfig,
    prepare_only,
    run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the independent PAL-NURBS minimal optimizer")
    parser.add_argument("--output", default=MinimalConfig.output)
    parser.add_argument("--excel", default=MinimalConfig.excel, help="PAL-NURBS 使用的镜片系统 Excel")
    parser.add_argument("--support-json", default=MinimalConfig.support_json, help="Original PAL 支持参数 JSON")
    parser.add_argument("--zones-json", default=MinimalConfig.zones_json, help="V3 PAL 功能分区 JSON")
    parser.add_argument("--device", default=MinimalConfig.device)
    parser.add_argument("--requested-np", type=int, default=MinimalConfig.requested_np)
    parser.add_argument("--fft-size-px", type=int, default=MinimalConfig.fft_size_px)
    parser.add_argument(
        "--case-batch-size", type=int, default=MinimalConfig.case_batch_size,
        help="GPU tensor case batch size；必须为正整数，不自动因 OOM 降级",
    )
    parser.add_argument(
        "--candidate-trace-import",
        default=None,
        help=(
            "导入完整且身份匹配的候选追迹进度；文件哈希进入新 run identity，"
            "导入后仍由采样器严格校验 trace identity"
        ),
    )
    parser.add_argument(
        "--forward-qualification-import",
        default=None,
        help="导入已完成且 pool identity 匹配的 WFNO 资格进度；源文件哈希进入新 run identity",
    )
    parser.add_argument(
        "--final-phase-qualification-import",
        default=None,
        help="导入已完成且 pool identity 匹配的完整相位门禁进度；源文件哈希进入新 run identity",
    )
    parser.add_argument(
        "--baseline-state-import",
        default=None,
        help=(
            "导入已完成的 Original PAL 109-case baseline 状态；源文件哈希进入新 run identity，"
            "导入后仍严格校验 case IDs、7x7 零 residual 与 baseline schema"
        ),
    )
    parser.add_argument(
        "--parent-run",
        default=None,
        help="已完成且保持只读的父 run；其阶段 best 和预处理证据用于创建新身份",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=(7, 11, 19),
        default=None,
        help="child 首个新增训练阶段；必须不早于父 run 的 terminal stage",
    )
    parser.add_argument(
        "--steps", type=int, nargs=3, metavar=("S7", "S11", "S19"),
        default=(MinimalConfig.max_steps_7, MinimalConfig.max_steps_11, MinimalConfig.max_steps_19),
        help=(
            "三个参数化阶段的最低 attempt 数；最后一个非零阶段为 terminal stage，"
            "此前阶段固定执行，拒绝的 attempt 同样计数"
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=MinimalConfig.early_stopping_patience,
        help="terminal stage 达到最低预算后允许停止的连续无显著改善 attempt 数",
    )
    parser.add_argument(
        "--relative-improvement-threshold",
        type=float,
        default=MinimalConfig.relative_improvement_threshold,
        help="terminal stage 刷新 best 时用于重置 patience 的严格相对改善阈值",
    )
    parser.add_argument(
        "--max-extra-terminal-stage-steps",
        type=int,
        default=MinimalConfig.max_extra_terminal_stage_steps,
        help="terminal stage 最低预算后允许的最大额外 attempt 数",
    )
    parser.add_argument(
        "--smooth-lambda",
        type=float,
        default=MinimalConfig.smooth_lambda,
        help="仅在 19x19 阶段启用的 NURBS 归一化二阶差分权重",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只输出优化前分区图、case 清单和镜片/物方 case 分布图，不启动 PSF 优化",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="仅在 config、输入哈希和实现闭包完全一致时从原运行目录精确恢复",
    )
    args = parser.parse_args()
    config = MinimalConfig(
        output=args.output, excel=args.excel, support_json=args.support_json, zones_json=args.zones_json,
        device=args.device, requested_np=args.requested_np,
        fft_size_px=args.fft_size_px, case_batch_size=args.case_batch_size,
        max_steps_7=args.steps[0],
        max_steps_11=args.steps[1], max_steps_19=args.steps[2],
        early_stopping_patience=args.early_stopping_patience,
        relative_improvement_threshold=args.relative_improvement_threshold,
        max_extra_terminal_stage_steps=args.max_extra_terminal_stage_steps,
        smooth_lambda=args.smooth_lambda,
        candidate_trace_import=args.candidate_trace_import,
        forward_qualification_import=args.forward_qualification_import,
        final_phase_qualification_import=args.final_phase_qualification_import,
        baseline_state_import=args.baseline_state_import,
        parent_run=args.parent_run,
        start_stage=args.start_stage,
    )
    if args.prepare_only:
        output = prepare_only(config, resume=args.resume)
        print(output)
        return 0
    output = run(config, resume=args.resume)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
