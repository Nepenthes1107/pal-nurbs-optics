# -*- coding: utf-8 -*-
"""
单次 PSF 计算脚本。

保持现有位置参数接口:
python multi_rays.py <excel_path> <obj_dist> <field_x> <field_y> [--cutoff ...] [--output ...]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from biot.services.single_field_service import (  # noqa: E402
    _compute_psf_once,
    calculate_psf,
    get_psf_health_metrics,
    modify_excel_config,
    resolve_device,
    save_psf_outputs,
)

__all__ = [
    "_compute_psf_once",
    "calculate_psf",
    "get_psf_health_metrics",
    "modify_excel_config",
    "resolve_device",
    "save_psf_outputs",
]


def _parse_object_distance(text: str):
    obj_dist_text = text.strip().lower()
    if obj_dist_text in {"inf", "infinity"}:
        return "Infinity", "inf"
    obj_distance = float(text)
    return obj_distance, str(int(obj_distance))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计算单个视场的 PSF（可选 MTF 导出）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "# 先激活环境: conda activate myenv\n"
            "  python multi_rays.py eye_image_glass.xlsx inf 0 0\n"
            "  python multi_rays.py eye_image_glass.xlsx 1000 10 10 --device auto --cutoff 100 --output results/test\n"
            "  python multi_rays.py eye_image_glass.xlsx inf 0 0 --with-mtf --cutoff 100 --output results/test_mtf"
        ),
    )

    parser.add_argument("excel_file", type=str, help="Excel 配置文件路径")
    parser.add_argument("object_distance", type=str, help='物距（mm），使用 "inf" 表示无穷远')
    parser.add_argument("field_x", type=float, help="视场角 X（degree）")
    parser.add_argument("field_y", type=float, help="视场角 Y（degree）")
    parser.add_argument("--cutoff", type=float, default=100.0, help="截止频率（cycles/mm）[默认: 100]")
    parser.add_argument("--np", type=int, default=256, help="光瞳采样数 [默认: 256]")
    parser.add_argument("--ni", type=int, default=512, help="像面采样数 [默认: 512]")
    parser.add_argument(
        "--zernike-n-max",
        type=int,
        default=5,
        help="标准 FFT 波前 Zernike 拟合最高阶 [默认: 5]",
    )
    parser.add_argument("--output", type=str, default=None, help="输出目录 [默认: 不保存文件]")
    parser.add_argument("--no-modify", action="store_true", help="不改写 Excel，直接使用原始配置")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="运行设备 [默认: auto]",
    )
    parser.add_argument(
        "--with-mtf",
        action="store_true",
        help="启用 MTF 曲线与指标输出（默认关闭）",
    )
    parser.add_argument(
        "--legacy-pupil-phase",
        action="store_true",
        help="使用旧的像面中心 pupil 相位参考（默认使用主光线/参考像点相位）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    obj_distance, obj_dist_name = _parse_object_distance(args.object_distance)

    print("=" * 60)
    print("PSF 计算配置")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"Excel 文件: {args.excel_file}")
    print(f"物距: {obj_dist_name} mm")
    print(f"视场角: ({args.field_x}, {args.field_y}) deg")
    print(f"截止频率: {args.cutoff} cycles/mm")
    print(f"光瞳采样: {args.np}")
    print(f"像面采样: {args.ni}")
    print(f"波前 Zernike 最高阶: {args.zernike_n_max}")
    print(f"设备策略: {args.device}")
    print(f"MTF 输出: {'开启' if args.with_mtf else '关闭'}")
    print(f"Pupil 相位参考: {'旧像面中心' if args.legacy_pupil_phase else '主光线/参考像点'}")
    if args.output:
        print(f"输出目录: {args.output}")
    print("=" * 60)

    temp_excel = None
    if args.no_modify:
        excel_to_use = args.excel_file
        print("\n[跳过 Excel 修改] 使用原始配置文件")
    else:
        temp_excel = Path(f"temp_config_{obj_dist_name}_field{args.field_x}_{args.field_y}.xlsx")
        modify_excel_config(args.excel_file, temp_excel, obj_distance, args.field_x, args.field_y)
        excel_to_use = str(temp_excel)

    try:
        calculate_psf(
            excel_path=excel_to_use,
            field_x=args.field_x,
            field_y=args.field_y,
            cutoff_freq=args.cutoff,
            n_p=args.np,
            n_i=args.ni,
            zernike_n_max=args.zernike_n_max,
            output_dir=args.output,
            device_pref=args.device,
            with_mtf=args.with_mtf,
            legacy_pupil_phase=args.legacy_pupil_phase,
        )
        print("\n" + "=" * 60)
        print("计算完成")
        print("=" * 60)
    except Exception as exc:
        print(f"\n[错误] 计算失败: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if temp_excel is not None and temp_excel.exists():
            temp_excel.unlink()
            print("[已清理] 临时配置文件")

    return 0


if __name__ == "__main__":
    sys.exit(main())

