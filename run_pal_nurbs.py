from __future__ import annotations

import argparse

from biot.e2e.pal_nurbs import (
    MinimalConfig,
    prepare_only,
    run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed three-distance 7x7 PAL B-spline optimizer"
    )
    parser.add_argument("--output", default=MinimalConfig.output)
    parser.add_argument("--excel", default=MinimalConfig.excel, help="PAL-NURBS 使用的镜片系统 Excel")
    parser.add_argument("--zones-json", default=MinimalConfig.zones_json, help="Far/Intermediate/Near 分区 JSON")
    parser.add_argument(
        "--weights-json",
        default=MinimalConfig.weights_json,
        help="三物距×PAL分区 loss 权重 JSON",
    )
    parser.add_argument("--device", default=MinimalConfig.device)
    parser.add_argument("--requested-np", type=int, default=MinimalConfig.requested_np)
    parser.add_argument("--fft-size-px", type=int, default=MinimalConfig.fft_size_px)
    parser.add_argument("--fov-min-deg", type=float, default=MinimalConfig.fov_min_deg)
    parser.add_argument("--fov-max-deg", type=float, default=MinimalConfig.fov_max_deg)
    parser.add_argument(
        "--fov-step-deg",
        type=float,
        default=MinimalConfig.fov_step_deg,
        help="FOV 采样间隔（degree）；范围必须能被该间隔整除",
    )
    parser.add_argument(
        "--case-batch-size",
        type=int,
        default=MinimalConfig.case_batch_size,
        help="并行追迹、FFT 并执行一次 backward 的 case 数，默认 8",
    )
    parser.add_argument(
        "--accepted-steps",
        type=int,
        default=MinimalConfig.max_accepted_steps,
        help="最大 accepted optimizer steps（拒绝步不消耗该预算）",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=MinimalConfig.early_stopping_patience,
    )
    parser.add_argument(
        "--relative-improvement-threshold",
        type=float,
        default=MinimalConfig.relative_improvement_threshold,
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只生成固定 3×11×11 case/weight 布局，不启动 PSF 优化",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="仅在 config、输入哈希和实现闭包完全一致时从原运行目录精确恢复",
    )
    args = parser.parse_args()
    config = MinimalConfig(
        output=args.output,
        excel=args.excel,
        zones_json=args.zones_json,
        weights_json=args.weights_json,
        device=args.device,
        requested_np=args.requested_np,
        fft_size_px=args.fft_size_px,
        fov_min_deg=args.fov_min_deg,
        fov_max_deg=args.fov_max_deg,
        fov_step_deg=args.fov_step_deg,
        case_batch_size=args.case_batch_size,
        max_accepted_steps=args.accepted_steps,
        early_stopping_patience=args.early_stopping_patience,
        relative_improvement_threshold=args.relative_improvement_threshold,
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
