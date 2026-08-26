"""Command line entry for BIOT lens geometric metrics."""

from __future__ import annotations

import os

# Workaround for OpenMP DLL conflict in conda/Windows environments.
# Multiple OpenMP runtimes (libomp.dll / libiomp5md.dll) may be
# linked into the same process, causing OMP Error #15.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import sys
from pathlib import Path

from biot.domain import (
    Device,
    DistortionCurveRequest,
    DistortionGridRequest,
    PowerAstigmatismRequest,
    ResultStatus,
    SystemConfig,
)
from biot.services import compute_distortion_curve, compute_distortion_grid, compute_power_astigmatism
from lens_metrics_core import (
    compute_footprint_coverage as compute_footprint_coverage_core,
    load_lens as load_lens_metrics_lens,
    resolve_device as resolve_lens_metrics_device,
    save_footprint_coverage_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute lens power, astigmatism, distortion, and footprint metrics")
    parser.add_argument("excel_path", help="Lens Excel configuration path")
    parser.add_argument(
        "mode",
        choices=["power", "distortion-curve", "distortion-grid", "footprint", "all"],
        help="Metric mode to run",
    )
    parser.add_argument("--wavelength", type=float, default=555.0, help="Wavelength in nm [default: 555]")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device strategy")
    parser.add_argument("--output", default="results/lens_metrics", help="Output directory")
    parser.add_argument("--lens-fov", type=float, default=50.0, help="Lensdata maximum FOV in degree [default: 50]")
    parser.add_argument("--aperture", type=float, default=2.0, help="System aperture radius in mm [default: 2]")
    parser.add_argument(
        "--fov",
        type=float,
        default=50.0,
        help="One-dimensional positive FOV limit in degree; power/distortion curves report the first view as (0,0) [default: 50]",
    )
    parser.add_argument("--fov-x", type=float, default=None, help="Grid max absolute X FOV in degree [default: --fov]")
    parser.add_argument("--fov-y", type=float, default=None, help="Grid max absolute Y FOV in degree [default: --fov]")
    parser.add_argument("--field-num", type=int, default=None, help="Field sample count")
    parser.add_argument("--display-grid-num", type=int, default=None, help="Display grid sample count")
    parser.add_argument("--axis", choices=["x", "y"], default="y", help="One-dimensional field axis")
    parser.add_argument(
        "--distortion-type",
        choices=[
            "rotating_eye_far",
            "fixed_eye_far",
            "rotating_eye_near",
            "fixed_eye_near",
            "handheld_near",
        ],
        default="rotating_eye_far",
        help="Distortion definition",
    )
    parser.add_argument(
        "--near-object-distance",
        type=float,
        default=250.0,
        help="Legacy length1: near object plane distance in mm [default: 250]",
    )
    parser.add_argument(
        "--pupil-distance",
        type=float,
        default=250.0,
        help="Legacy length2: eyeglass-to-pupil distance for handheld mode in mm [default: 250]",
    )
    parser.add_argument(
        "--lens-front-index",
        type=int,
        default=None,
        help="Eyeglass front surface index [default: surfaces[1], Excel row 5]",
    )
    parser.add_argument(
        "--lens-back-index",
        type=int,
        default=None,
        help="Eyeglass back surface index [default: surfaces[2], Excel row 6]",
    )
    parser.add_argument(
        "--fix-original-grid-axis-bug",
        action="store_true",
        help="Use rx for the grid boundary X angle instead of replicating the original ry/ry behaviour",
    )
    parser.add_argument(
        "--differential-aperture",
        type=float,
        default=0.01,
        help="Differential ray offset in mm for power calculation",
    )
    parser.add_argument("--focal-power", type=float, default=0.0, help="Target correction power in D")
    parser.add_argument("--averfang-crib", type=float, default=80.0, help="AverFang footprint aperture diameter in mm")
    parser.add_argument(
        "--map-trim-pixels",
        type=int,
        default=3,
        help="Display crop pixels on each AverFang map edge [default: 3]",
    )
    parser.add_argument(
        "--field-radius",
        type=float,
        default=50.0,
        help="Circular field boundary radius in degree for footprint mode [default: 50]",
    )
    parser.add_argument(
        "--field-ring-num",
        type=int,
        default=72,
        help="Field azimuth sample count for footprint mode; endpoint is excluded [default: 72]",
    )
    parser.add_argument(
        "--pupil-ring-num",
        type=int,
        default=72,
        help="Pupil-boundary sample count for footprint mode; endpoint is excluded [default: 72]",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output)
    print("=" * 60)
    print("Lens metric evaluation")
    print("=" * 60)
    print(f"Excel: {args.excel_path}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Wavelength: {args.wavelength} nm")
    print(f"Output: {output_root}")
    print("=" * 60)

    try:
        system = SystemConfig(
            excel_path=Path(args.excel_path),
            object_distance_mm=float("inf"),
            wavelength_nm=args.wavelength,
            pupil_radius_mm=args.aperture,
            device=Device(args.device),
            lens_front_index=args.lens_front_index,
            lens_back_index=args.lens_back_index,
        )
        if args.mode in {"power", "all"}:
            field_num = args.field_num if args.field_num is not None else 51
            out_dir = output_root / "power" if args.mode == "all" else output_root
            request = PowerAstigmatismRequest(
                system=system,
                fov_deg=args.fov,
                field_num=field_num,
                axis=args.axis,
                lens_fov_deg=args.lens_fov,
                aperture_mm=args.aperture,
                wavelength_nm=args.wavelength,
                differential_aperture_mm=args.differential_aperture,
                target_focal_power_d=args.focal_power,
                averfang_crib_diameter_mm=args.averfang_crib,
                output_dir=out_dir,
            )
            result = compute_power_astigmatism(request)
            if result.status != ResultStatus.SUCCEEDED:
                raise RuntimeError(result.error or f"power failed with status {result.status.value}")
            print(f"[power] wrote {result.artifacts['power_csv']}")

        if args.mode in {"distortion-curve", "all"}:
            field_num = args.field_num if args.field_num is not None else 51
            out_dir = output_root / "distortion_curve" if args.mode == "all" else output_root
            request = DistortionCurveRequest(
                system=system,
                fov_deg=args.fov,
                field_num=field_num,
                axis=args.axis,
                distortion_type=args.distortion_type,
                lens_fov_deg=args.lens_fov,
                aperture_mm=args.aperture,
                wavelength_nm=args.wavelength,
                near_object_distance_mm=args.near_object_distance,
                pupil_distance_mm=args.pupil_distance,
                output_dir=out_dir,
            )
            result = compute_distortion_curve(request)
            if result.status != ResultStatus.SUCCEEDED:
                raise RuntimeError(result.error or f"distortion curve failed with status {result.status.value}")
            print(f"[distortion-curve] wrote {result.artifacts['distortion_curve_csv']}")

        if args.mode in {"distortion-grid", "all"}:
            display_grid_num = args.display_grid_num if args.display_grid_num is not None else 21
            field_num = args.field_num if args.field_num is not None else display_grid_num
            out_dir = output_root / "distortion_grid" if args.mode == "all" else output_root
            request = DistortionGridRequest(
                system=system,
                fov_x_deg=args.fov_x if args.fov_x is not None else args.fov,
                fov_y_deg=args.fov_y if args.fov_y is not None else args.fov,
                field_num=field_num,
                display_grid_num=display_grid_num,
                distortion_type=args.distortion_type,
                lens_fov_deg=args.lens_fov,
                aperture_mm=args.aperture,
                wavelength_nm=args.wavelength,
                near_object_distance_mm=args.near_object_distance,
                pupil_distance_mm=args.pupil_distance,
                fix_original_grid_axis_bug=args.fix_original_grid_axis_bug,
                output_dir=out_dir,
            )
            result = compute_distortion_grid(request)
            if result.status != ResultStatus.SUCCEEDED:
                raise RuntimeError(result.error or f"distortion grid failed with status {result.status.value}")
            print(f"[distortion-grid] wrote {result.artifacts['distortion_grid_csv']}")

        if args.mode in {"footprint", "all"}:
            out_dir = output_root / "footprint" if args.mode == "all" else output_root
            device = resolve_lens_metrics_device(args.device)
            lens = load_lens_metrics_lens(
                args.excel_path,
                device=device,
                fov_deg=args.lens_fov,
                aperture_mm=args.aperture,
                wavelength_nm=args.wavelength,
            )
            payload = compute_footprint_coverage_core(
                lens,
                field_radius_deg=args.field_radius,
                field_sample_count=args.field_ring_num,
                pupil_sample_count=args.pupil_ring_num,
                pupil_radius_mm=args.aperture,
                wavelength_nm=args.wavelength,
                crib_diameter_mm=args.averfang_crib,
            )
            paths = save_footprint_coverage_outputs(payload, out_dir, trim_pixels=args.map_trim_pixels)
            print(f"[footprint] wrote {paths['png']}")

        print("=" * 60)
        print("Lens metric evaluation complete")
        print("=" * 60)
        return 0
    except Exception as exc:
        print(f"[error] lens metric evaluation failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
