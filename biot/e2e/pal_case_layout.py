"""Phase 16 dense-field training-case design and pre-optimization audit plots."""
from __future__ import annotations

import hashlib
import contextlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


TRAINING_GROUP_COUNTS = {
    "far": 20,
    "far_robustness": 8,
    "corridor_upper": 5,
    "corridor_middle": 5,
    "corridor_lower": 5,
    "near": 20,
    "near_robustness": 8,
    "near_edge_astig": 8,
    "peripheral_left": 15,
    "peripheral_right": 15,
}
TOTAL_TRAINING_CASES = sum(TRAINING_GROUP_COUNTS.values())
PERIPHERAL_BAND_COUNTS = {"upper": 4, "middle": 5, "lower": 6}
PERIPHERAL_REAR_MIRROR_TOLERANCE_MM = 1.0e-4
REFERENCE_RETRACE_TOLERANCE_MM = 1.0e-8
FUNCTIONAL_GROUPS = (
    "far",
    "far_robustness",
    "corridor_upper",
    "corridor_middle",
    "corridor_lower",
    "near",
    "near_robustness",
    "near_edge_astig",
)
PERIPHERAL_GROUPS = ("peripheral_left", "peripheral_right")
PARTITION_ORDER = (
    "far", "corridor", "corridor_flank", "near",
    "peripheral_astig_left", "peripheral_astig_right",
)
GROUP_TO_ZONE = {
    "far": "far",
    "far_robustness": "far",
    "corridor_upper": "corridor",
    "corridor_middle": "corridor",
    "corridor_lower": "corridor",
    "near": "near",
    "near_robustness": "near",
    "near_edge_astig": "near",
    "peripheral_left": "astig_left",
    "peripheral_right": "astig_right",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _zone_arrays(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    x = np.asarray(payload["x_mm"], dtype=np.float64)
    y = np.asarray(payload["physical_y_mm"], dtype=np.float64)
    masks = {name: np.asarray(mask, dtype=bool) for name, mask in dict(payload["masks"]).items()}
    shape = (y.size, x.size)
    if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
        raise ValueError("zone coordinates must be non-trivial 1-D arrays")
    if any(mask.shape != shape for mask in masks.values()):
        raise ValueError("zone masks do not match x_mm/physical_y_mm")
    return x, y, masks


def classify_partition_point(
    zones_payload: Mapping[str, Any], *, x_mm: float, physical_y_mm: float
) -> str | None:
    """Classify a traced rear-surface point using the stored mask cells."""
    x, y, masks = _zone_arrays(zones_payload)
    ix, iy = int(np.argmin(abs(x - x_mm))), int(np.argmin(abs(y - physical_y_mm)))
    xp, yp = float(np.median(abs(np.diff(x)))), float(np.median(abs(np.diff(y))))
    if abs(float(x[ix]) - x_mm) > 0.5 * xp + 1e-9 or abs(float(y[iy]) - physical_y_mm) > 0.5 * yp + 1e-9:
        return None
    if "transition" in masks and bool(masks["transition"][iy, ix]):
        normalized_add = np.asarray(zones_payload.get("normalized_add_t"), dtype=np.float64)
        if normalized_add.shape != masks["transition"].shape:
            raise ValueError("zones transition mask requires same-shape normalized_add_t")
        value = float(normalized_add[iy, ix])
        if not math.isfinite(value):
            raise ValueError("zones normalized_add_t contains a non-finite transition value")
        return "far" if value < 0.5 else "near"
    active = [name for name in PARTITION_ORDER if name in masks and bool(masks[name][iy, ix])]
    if set(active) == {"corridor", "corridor_flank"}:
        return "corridor"
    if len(active) != 1:
        return None
    return {
        "peripheral_astig_left": "astig_left",
        "peripheral_astig_right": "astig_right",
        "corridor_flank": "corridor",
    }.get(active[0], active[0])


def _mask_name(zone: str) -> str:
    return {"astig_left": "peripheral_astig_left", "astig_right": "peripheral_astig_right"}.get(zone, zone)


def _inside_clearance_mm(payload: Mapping[str, Any], mask_name: str, x_mm: float, y_mm: float) -> float:
    """Conservative point-to-nearest-outside-cell clearance of a raster mask."""
    x, y, masks = _zone_arrays(payload)
    mask = masks[mask_name]
    ix, iy = int(np.argmin(abs(x - x_mm))), int(np.argmin(abs(y - y_mm)))
    if not bool(mask[iy, ix]):
        return -math.inf
    oy, ox = np.nonzero(~mask)
    if ox.size == 0:
        return math.inf
    distance = np.hypot(x[ox] - x_mm, y[oy] - y_mm).min()
    half_diagonal = 0.5 * math.hypot(float(np.median(abs(np.diff(x)))), float(np.median(abs(np.diff(y)))))
    return max(0.0, float(distance) - half_diagonal)


def _linspace_exact(minimum: float, maximum: float, step: float) -> np.ndarray:
    if step <= 0 or maximum <= minimum:
        raise ValueError("invalid dense candidate field grid")
    intervals = int(round((maximum - minimum) / step))
    values = minimum + step * np.arange(intervals + 1, dtype=np.float64)
    if abs(float(values[-1]) - maximum) > 1e-9:
        raise ValueError("candidate range must be exactly divisible by step")
    return values


def generate_dense_candidate_fields(
    *,
    field_x_min_deg: float | None = None,
    field_x_max_deg: float | None = None,
    field_y_min_deg: float | None = None,
    field_y_max_deg: float | None = None,
    field_step_deg: float,
    field_min_deg: float | None = None,
    field_max_deg: float | None = None,
) -> list[dict[str, Any]]:
    """Generate a deterministic row-major asymmetric field grid.

    ``field_min_deg``/``field_max_deg`` are compatibility aliases.  Supplying
    an alias together with the corresponding explicit XY bound is rejected so
    the run identity cannot depend on an ambiguous precedence rule.
    """
    if field_min_deg is not None:
        if field_x_min_deg is not None or field_y_min_deg is not None:
            raise ValueError("field_min_deg cannot be mixed with explicit XY minima")
        field_x_min_deg = field_y_min_deg = float(field_min_deg)
    if field_max_deg is not None:
        if field_x_max_deg is not None or field_y_max_deg is not None:
            raise ValueError("field_max_deg cannot be mixed with explicit XY maxima")
        field_x_max_deg = field_y_max_deg = float(field_max_deg)
    bounds = (field_x_min_deg, field_x_max_deg, field_y_min_deg, field_y_max_deg)
    if any(value is None for value in bounds):
        raise ValueError("candidate grid requires explicit X/Y bounds")
    x_values = _linspace_exact(float(field_x_min_deg), float(field_x_max_deg), float(field_step_deg))
    y_values = _linspace_exact(float(field_y_min_deg), float(field_y_max_deg), float(field_step_deg))
    return [
        {"candidate_id": f"cand_{index + 1:05d}", "field_x_deg": float(fx), "field_y_deg": float(fy)}
        for index, (fy, fx) in enumerate((fy, fx) for fy in y_values for fx in x_values)
    ]


def trace_candidate_fields(
    candidates: Sequence[Mapping[str, Any]], *,
    trace_reference: Callable[[float, float], tuple[float, float]],
    zones_payload: Mapping[str, Any],
    zone_boundary_safety_mm: float | Mapping[str, float],
    aperture_edge_safety_mm: float,
    progress_path: str | Path | None = None,
    trace_identity: Mapping[str, Any] | None = None,
    progress_interval: int = 100,
) -> list[dict[str, Any]]:
    """Trace every field through Original PAL with fail evidence and exact resume state."""
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    aperture_safety = float(aperture_edge_safety_mm)
    if not math.isfinite(aperture_safety) or aperture_safety < 0.0:
        raise ValueError("aperture-edge safety margin must be finite and non-negative")
    partition_zones = {"far", "corridor", "near", "astig_left", "astig_right"}
    if isinstance(zone_boundary_safety_mm, Mapping):
        supplied_zone_safety = {
            str(name): float(value) for name, value in zone_boundary_safety_mm.items()
        }
        missing = sorted(
            zone for zone in partition_zones
            if zone not in supplied_zone_safety and "default" not in supplied_zone_safety
        )
        if missing:
            raise ValueError(
                "missing zone-boundary safety margins for: " + ", ".join(missing)
            )
        resolved_zone_safety = {
            zone: (
                supplied_zone_safety[zone]
                if zone in supplied_zone_safety
                else supplied_zone_safety["default"]
            )
            for zone in partition_zones
        }
        identity_zone_safety: float | dict[str, float] = supplied_zone_safety
    else:
        shared_zone_safety = float(zone_boundary_safety_mm)
        resolved_zone_safety = {
            zone: shared_zone_safety for zone in partition_zones
        }
        identity_zone_safety = shared_zone_safety
    if any(
        not math.isfinite(value) or value < 0.0
        for value in resolved_zone_safety.values()
    ):
        raise ValueError("zone-boundary safety margins must be finite and non-negative")
    progress = None if progress_path is None else Path(progress_path)
    identity_payload = {
        "candidates": [dict(candidate) for candidate in candidates],
        "zones_sha256": _canonical_json_sha256(zones_payload),
        "zone_boundary_safety_mm": identity_zone_safety,
        "aperture_edge_safety_mm": aperture_safety,
        "trace_identity": None if trace_identity is None else dict(trace_identity),
    }
    identity_sha256 = _canonical_json_sha256(identity_payload)
    traced: list[dict[str, Any]] = []
    if progress is not None and progress.exists():
        saved = _read_json(progress)
        if saved.get("schema_version") != 1 or saved.get("identity_sha256") != identity_sha256:
            raise ValueError(f"candidate progress identity mismatch: {progress}")
        saved_rows = saved.get("rows")
        if not isinstance(saved_rows, list):
            raise ValueError(f"candidate progress rows are malformed: {progress}")
        traced = [dict(row) for row in saved_rows]
        if int(saved.get("next_candidate_index", -1)) != len(traced):
            raise ValueError(f"candidate progress next index is inconsistent: {progress}")
        expected_prefix = [str(row["candidate_id"]) for row in candidates[: len(traced)]]
        actual_prefix = [str(row.get("candidate_id")) for row in traced]
        if actual_prefix != expected_prefix:
            raise ValueError(f"candidate progress prefix does not match candidate grid: {progress}")
        if len(traced) > len(candidates):
            raise ValueError(f"candidate progress exceeds candidate grid: {progress}")
        print(
            f"candidate trace resume: {len(traced)}/{len(candidates)} from {progress}",
            flush=True,
        )

    def save_progress(status: str) -> None:
        if progress is None:
            return
        _write_json_atomic(
            progress,
            {
                "schema_version": 1,
                "status": status,
                "identity_sha256": identity_sha256,
                "identity": identity_payload,
                "candidate_count": len(candidates),
                "next_candidate_index": len(traced),
                "trace_success_count": sum(row.get("trace_status") == "ok" for row in traced),
                "trace_failure_count": sum(row.get("trace_status") != "ok" for row in traced),
                "rows": traced,
            },
        )

    for candidate_index, candidate in enumerate(
        candidates[len(traced) :], start=len(traced) + 1
    ):
        row = dict(candidate)
        diagnostic_stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(diagnostic_stream), contextlib.redirect_stderr(diagnostic_stream):
                x_mm, y_mm = trace_reference(
                    float(row["field_x_deg"]), float(row["field_y_deg"])
                )
            zone = classify_partition_point(zones_payload, x_mm=x_mm, physical_y_mm=y_mm)
            row.update({
                "trace_status": "ok", "reference_lens_x_mm": float(x_mm),
                "reference_lens_physical_y_mm": float(y_mm), "reference_partition_zone": zone,
            })
            if zone is None:
                row.update({"zone_boundary_clearance_mm": None, "aperture_edge_clearance_mm": None, "eligible": False})
            else:
                zone_clearance = _inside_clearance_mm(zones_payload, _mask_name(zone), x_mm, y_mm)
                aperture_clearance = _inside_clearance_mm(zones_payload, "monitored", x_mm, y_mm)
                zone_safety = resolved_zone_safety[zone]
                row.update({
                    "zone_boundary_clearance_mm": zone_clearance,
                    "aperture_edge_clearance_mm": aperture_clearance,
                    "zone_boundary_safety_mm": zone_safety,
                    "eligible": zone_clearance >= zone_safety and aperture_clearance >= aperture_safety,
                })
        except Exception as exc:
            diagnostic = diagnostic_stream.getvalue()
            row.update({
                "trace_status": "failed", "trace_error_type": type(exc).__name__,
                "trace_error": str(exc), "reference_partition_zone": None, "eligible": False,
                "trace_diagnostic_character_count": len(diagnostic),
                "trace_diagnostic_sha256": hashlib.sha256(diagnostic.encode("utf-8")).hexdigest(),
                "trace_diagnostic_tail": diagnostic[-4000:],
            })
        traced.append(row)
        if candidate_index % progress_interval == 0 or candidate_index == len(candidates):
            save_progress("complete" if candidate_index == len(candidates) else "running")
        if candidate_index % 500 == 0 or candidate_index == len(candidates):
            failure_count = sum(item.get("trace_status") != "ok" for item in traced)
            print(
                f"candidate trace progress: {candidate_index}/{len(candidates)}, "
                f"failures={failure_count}",
                flush=True,
            )
    save_progress("complete")
    return traced


def _fps_indices(
    points: np.ndarray,
    count: int,
    seed_target: Sequence[float] | None = None,
    initial_indices: Sequence[int] | None = None,
) -> list[int]:
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < count or count <= 0:
        raise ValueError(f"FPS needs at least {count} finite 2-D points, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("FPS points must be finite")
    if initial_indices is None:
        target = points.mean(0) if seed_target is None else np.asarray(seed_target, dtype=np.float64)
        selected = [int(np.argmin(np.square(points - target).sum(1)))]
    else:
        selected = [int(index) for index in initial_indices]
        if (
            not selected
            or len(selected) > count
            or len(set(selected)) != len(selected)
            or min(selected) < 0
            or max(selected) >= points.shape[0]
        ):
            raise ValueError("invalid initial FPS indices")
    minimum = np.full((points.shape[0],), np.inf, dtype=np.float64)
    for index in selected:
        minimum = np.minimum(minimum, np.square(points - points[index]).sum(1))
    while len(selected) < count:
        minimum[np.asarray(selected)] = -1.0
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(minimum, np.square(points - points[index]).sum(1))
    return selected


def _fps_rows(rows: Sequence[Mapping[str, Any]], count: int, seed_target: Sequence[float] | None = None) -> list[dict[str, Any]]:
    points = np.asarray([[row["reference_lens_x_mm"], row["reference_lens_physical_y_mm"]] for row in rows], dtype=np.float64)
    return [dict(rows[index]) for index in _fps_indices(points, count, seed_target)]


def _nearest_neighbour_p95_mm(rows: Sequence[Mapping[str, Any]]) -> float:
    if len(rows) < 2:
        raise ValueError("nearest-neighbour scale needs at least two points")
    points = np.asarray(
        [
            [row["reference_lens_x_mm"], row["reference_lens_physical_y_mm"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    distances = np.sqrt(
        np.square(points[:, None, :] - points[None, :, :]).sum(axis=2)
    )
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    if not np.isfinite(nearest).all():
        raise ValueError("eligible candidate coordinates contain duplicate-only geometry")
    return float(np.percentile(nearest, 95.0))


def corridor_object_distance_mm(
    power_map: np.ndarray,
    pfar: float,
    case_lens_y_mm: float,
    zones_payload: Mapping[str, Any],
) -> float:
    """Interpolate the corridor design distance from the Original PAL power map."""
    power = np.asarray(power_map, dtype=np.float64)
    x_mm = np.asarray(zones_payload["x_mm"], dtype=np.float64)
    y_mm = np.asarray(zones_payload["physical_y_mm"], dtype=np.float64)
    if power.shape != (y_mm.size, x_mm.size) or not np.isfinite(power).all():
        raise ValueError("corridor power map must be finite and match zones coordinates")
    centre = np.abs(x_mm) <= 2.0
    if not bool(centre.any()):
        raise ValueError("corridor power map has no x=0 +/-2 mm centre samples")
    row = int(np.argmin(np.abs(y_mm - float(case_lens_y_mm))))
    local_power = float(np.mean(power[row, centre]))
    local_add = max(local_power - float(pfar), 0.05)
    distance = 1000.0 / local_add
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("corridor object distance is not finite and positive")
    return distance


def _corridor_add_and_distance(
    row: Mapping[str, Any],
    *,
    power_map: np.ndarray,
    pfar: float,
    zones_payload: Mapping[str, Any],
) -> tuple[float, float]:
    distance = corridor_object_distance_mm(
        power_map,
        pfar,
        float(row["reference_lens_physical_y_mm"]),
        zones_payload,
    )
    return 1000.0 / distance, distance


def _select_corridor_stratum(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    add_min_D: float,
    add_max_D: float,
    include_upper: bool,
    power_map: np.ndarray,
    pfar: float,
    zones_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for source in rows:
        local_add, distance = _corridor_add_and_distance(
            source, power_map=power_map, pfar=pfar, zones_payload=zones_payload
        )
        in_band = add_min_D <= local_add <= add_max_D if include_upper else add_min_D <= local_add < add_max_D
        if in_band:
            members.append({
                **dict(source),
                "corridor_local_add_D": local_add,
                "distance_mm": distance,
            })
    if len(members) < int(count):
        raise ValueError(
            f"insufficient eligible corridor candidates in ADD [{add_min_D:g},{add_max_D:g}] D: "
            f"{len(members)} < {count}"
        )
    seed_y = float(np.median([
        float(row["reference_lens_physical_y_mm"]) for row in members
    ]))
    return _fps_rows(members, int(count), (0.0, seed_y))


def _peripheral_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(round(float(r["field_x_deg"]), 9), round(float(r["field_y_deg"]), 9)): dict(r) for r in rows}
    pairs: list[dict[str, Any]] = []
    for row in rows:
        if row["reference_partition_zone"] != "astig_left":
            continue
        partner = lookup.get((-round(float(row["field_x_deg"]), 9), round(float(row["field_y_deg"]), 9)))
        if partner is None or partner["reference_partition_zone"] != "astig_right":
            continue
        mirror_error_x = abs(
            float(row["reference_lens_x_mm"]) + float(partner["reference_lens_x_mm"])
        )
        mirror_error_y = abs(
            float(row["reference_lens_physical_y_mm"])
            - float(partner["reference_lens_physical_y_mm"])
        )
        if max(mirror_error_x, mirror_error_y) > PERIPHERAL_REAR_MIRROR_TOLERANCE_MM:
            raise ValueError(
                "field-mirrored peripheral candidates are not mirrored on the Original PAL rear "
                f"surface: {row['candidate_id']} / {partner['candidate_id']}, "
                f"dx={mirror_error_x:.6g} mm, dy={mirror_error_y:.6g} mm"
            )
        pairs.append({
            "left": dict(row), "right": partner,
            "pair_x_mm": 0.5 * (abs(float(row["reference_lens_x_mm"])) + abs(float(partner["reference_lens_x_mm"]))),
            "pair_y_mm": 0.5 * (float(row["reference_lens_physical_y_mm"]) + float(partner["reference_lens_physical_y_mm"])),
            "reference_rear_mirror_error_mm": {
                "x_antisymmetry": mirror_error_x, "y_symmetry": mirror_error_y
            },
        })
    return pairs


def _select_peripheral(
    rows: Sequence[Mapping[str, Any]], corridor_y_min_mm: float,
    corridor_y_max_mm: float, *,
    band_counts: Mapping[str, int] = PERIPHERAL_BAND_COUNTS,
) -> list[dict[str, Any]]:
    pairs = _peripheral_pairs(rows)
    specs = (
        ("upper", int(band_counts["upper"]), lambda y: y > corridor_y_max_mm),
        ("middle", int(band_counts["middle"]), lambda y: corridor_y_min_mm <= y <= corridor_y_max_mm),
        ("lower", int(band_counts["lower"]), lambda y: y < corridor_y_min_mm),
    )
    selected: list[dict[str, Any]] = []
    for band, count, predicate in specs:
        members = [pair for pair in pairs if predicate(float(pair["pair_y_mm"]))]
        points = np.asarray([[pair["pair_x_mm"], pair["pair_y_mm"]] for pair in members], dtype=np.float64)
        if band == "upper":
            # Three-point maximin coverage already spans inner-x and both y
            # limits on the audited domain; add the outer-x physical anchor as
            # the smallest evidence-driven correction to its sole failed bound.
            initial = _fps_indices(points, count - 1)
            outer_x = int(np.argmax(points[:, 0]))
            initial = list(dict.fromkeys((*initial, outer_x)))
            indices = _fps_indices(points, count, initial_indices=initial)
        else:
            indices = _fps_indices(points, count)
        for index in indices:
            selected.append({**members[index], "peripheral_band": band})
    return selected


def select_training_cases(
    traced_candidates: Sequence[Mapping[str, Any]], *,
    far_object_distance_mm: float, intermediate_object_distance_mm: float,
    near_object_distance_mm: float, corridor_y_min_mm: float, corridor_y_max_mm: float,
    power_map: np.ndarray,
    pfar: float,
    zones_payload: Mapping[str, Any],
    group_counts: Mapping[str, int] = TRAINING_GROUP_COUNTS,
    peripheral_band_counts: Mapping[str, int] = PERIPHERAL_BAND_COUNTS,
) -> list[dict[str, Any]]:
    """Select physical cases by lens-plane FPS after real tracing."""
    resolved_counts = {name: int(group_counts[name]) for name in TRAINING_GROUP_COUNTS}
    resolved_band_counts = {
        name: int(peripheral_band_counts[name]) for name in PERIPHERAL_BAND_COUNTS
    }
    if any(value <= 0 for value in resolved_counts.values()):
        raise ValueError("all training group counts must be positive")
    if resolved_counts["peripheral_left"] != resolved_counts["peripheral_right"]:
        raise ValueError("peripheral groups must contain equal mirror-pair counts")
    if sum(resolved_band_counts.values()) != resolved_counts["peripheral_left"]:
        raise ValueError("peripheral band counts do not match the group count")
    eligible = [dict(row) for row in traced_candidates if bool(row.get("eligible"))]
    tagged = [row.get("training_group") is not None for row in eligible]
    if any(tagged) and not all(tagged):
        raise ValueError(
            "eligible candidates must either all omit training_group or all retain it"
        )
    retain_source_groups = bool(tagged) and all(tagged)

    def source_rows(group: str, zone: str) -> list[dict[str, Any]]:
        """Return one group's own qualified source domain for final FPS.

        Dense traced candidates are initially untagged and may feed every
        objective group associated with their physical partition.  The
        forward-qualified oversampling pool is already tagged, however, and
        each record carries group-specific distance/qualification evidence.
        Re-selection from that pool must not mix sibling groups that happen
        to share a partition (for example far and far_robustness).
        """
        return [
            row for row in eligible
            if row.get("reference_partition_zone") == zone
            and (
                not retain_source_groups
                or str(row.get("training_group")) == group
            )
        ]

    far_rows = source_rows("far", "far")
    far_robustness_rows = source_rows("far_robustness", "far")
    corridor_upper_rows = source_rows("corridor_upper", "corridor")
    corridor_middle_rows = source_rows("corridor_middle", "corridor")
    corridor_lower_rows = source_rows("corridor_lower", "corridor")
    near_rows = source_rows("near", "near")
    near_robustness_rows = source_rows("near_robustness", "near")
    near_edge_astig_rows = source_rows("near_edge_astig", "near")
    groups: dict[str, list[dict[str, Any]]] = {
        "far": _fps_rows(far_rows, resolved_counts["far"]),
        "far_robustness": _fps_rows(
            far_robustness_rows,
            resolved_counts["far_robustness"],
            (0.0, 15.0),
        ),
        "corridor_upper": _select_corridor_stratum(
            corridor_upper_rows, resolved_counts["corridor_upper"],
            add_min_D=0.2, add_max_D=0.5, include_upper=False,
            power_map=power_map, pfar=pfar, zones_payload=zones_payload,
        ),
        "corridor_middle": _select_corridor_stratum(
            corridor_middle_rows, resolved_counts["corridor_middle"],
            add_min_D=0.5, add_max_D=1.3, include_upper=False,
            power_map=power_map, pfar=pfar, zones_payload=zones_payload,
        ),
        "corridor_lower": _select_corridor_stratum(
            corridor_lower_rows, resolved_counts["corridor_lower"],
            add_min_D=1.3, add_max_D=2.0, include_upper=True,
            power_map=power_map, pfar=pfar, zones_payload=zones_payload,
        ),
        "near": _fps_rows(near_rows, resolved_counts["near"]),
        "near_robustness": _fps_rows(
            near_robustness_rows, resolved_counts["near_robustness"]
        ),
        "near_edge_astig": _fps_rows(
            [
                row for row in near_edge_astig_rows
                if abs(float(row["reference_lens_x_mm"])) > 10.0
            ],
            resolved_counts["near_edge_astig"],
            (12.0, -28.0),
        ),
        "peripheral_left": [], "peripheral_right": [],
    }
    peripheral = (
        source_rows("peripheral_left", "astig_left")
        + source_rows("peripheral_right", "astig_right")
    )
    pairs = _select_peripheral(
        peripheral, corridor_y_min_mm, corridor_y_max_mm,
        band_counts=resolved_band_counts,
    )
    band_distance = {name: far_object_distance_mm for name in PERIPHERAL_BAND_COUNTS}
    for pair_index, pair in enumerate(pairs, 1):
        for side, group in (("left", "peripheral_left"), ("right", "peripheral_right")):
            groups[group].append({
                **pair[side], "peripheral_pair_id": f"peripheral_pair_{pair_index:02d}",
                "peripheral_band": pair["peripheral_band"], "distance_mm": float(band_distance[pair["peripheral_band"]]),
            })
    group_distance = {
        "far": far_object_distance_mm,
        "far_robustness": intermediate_object_distance_mm,
        "near": near_object_distance_mm,
        "near_robustness": intermediate_object_distance_mm,
        "near_edge_astig": near_object_distance_mm,
        "peripheral_left": far_object_distance_mm,
        "peripheral_right": far_object_distance_mm,
    }

    def distance_id(distance: float) -> str:
        """Serialize the physical object distance for a stable case ID."""
        if math.isinf(distance):
            if distance > 0.0:
                return "Dinf"
            raise ValueError("object distance must be positive or Infinity")
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError(f"object distance must be positive and finite, got {distance!r}")
        return f"D{int(round(distance))}"

    cases: list[dict[str, Any]] = []
    for group, expected in resolved_counts.items():
        if len(groups[group]) != expected:
            raise ValueError(f"wrong selected count for {group}: {len(groups[group])}, expected {expected}")
        for index, row in enumerate(groups[group], 1):
            distance = float(row.get("distance_mm", group_distance.get(group, math.nan)))
            cases.append({
                **row, "sample_id": f"{group}_{index:02d}",
                "case_id": f"{group}_{index:02d}_{distance_id(distance)}",
                "training_group": group, "zone": GROUP_TO_ZONE[group], "distance_mm": distance,
            })
    expected_total = sum(resolved_counts.values())
    if len(cases) != expected_total or len({case["case_id"] for case in cases}) != expected_total:
        raise AssertionError(
            f"training case contract requires exactly {expected_total} unique cases"
        )
    return cases


def partition_audit(payload: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _, _, masks = _zone_arrays(payload)
    topology = dict(dict(payload.get("rules", {})).get("topology", {}))
    stats = dict(payload.get("statistics", {}))
    zone_groups = {
        "far": ("far", "far_robustness"),
        "corridor": ("corridor_upper", "corridor_middle", "corridor_lower"),
        "corridor_flank": ("corridor_upper", "corridor_middle", "corridor_lower"),
        "near": ("near", "near_robustness", "near_edge_astig"),
        "peripheral_astig_left": ("peripheral_left",),
        "peripheral_astig_right": ("peripheral_right",),
    }
    zones: dict[str, Any] = {}
    for zone in PARTITION_ORDER:
        final = int(np.count_nonzero(masks[zone]))
        source = int(dict(topology.get(zone, {})).get("source_pixel_count", final))
        zones[zone] = {
            "source_pixel_count": source, "final_pixel_count": final,
            "topology_removed_pixel_count": source - final,
            "topology_retained_fraction": final / source if source else None,
            "statistics": stats.get(zone),
            "training_case_count": sum(case.get("training_group") in zone_groups[zone] for case in cases),
        }
    return {"interpretation": "topology retention and physical zone extent are reported separately", "zones": zones}


def _xy_bounds(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]] | None:
    if not rows:
        return None
    x = np.asarray([row["reference_lens_x_mm"] for row in rows], dtype=np.float64)
    y = np.asarray([row["reference_lens_physical_y_mm"] for row in rows], dtype=np.float64)
    return {
        "x_mm": [float(x.min()), float(x.max())],
        "physical_y_mm": [float(y.min()), float(y.max())],
    }


def _occupied_mask_cells(
    payload: Mapping[str, Any], mask_name: str, rows: Sequence[Mapping[str, Any]]
) -> int:
    x, y, masks = _zone_arrays(payload)
    occupied: set[tuple[int, int]] = set()
    for row in rows:
        ix = int(np.argmin(abs(x - float(row["reference_lens_x_mm"]))))
        iy = int(np.argmin(abs(y - float(row["reference_lens_physical_y_mm"]))))
        if bool(masks[mask_name][iy, ix]):
            occupied.add((iy, ix))
    return len(occupied)


def _convex_hull_area_mm2(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Return the selected-point envelope area; this is not a zone-area estimate."""
    if len(rows) < 3:
        return None
    points = sorted({
        (
            float(row["reference_lens_x_mm"]),
            float(row["reference_lens_physical_y_mm"]),
        )
        for row in rows
    })
    if len(points) < 3:
        return 0.0

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    twice_area = sum(
        hull[index][0] * hull[(index + 1) % len(hull)][1]
        - hull[(index + 1) % len(hull)][0] * hull[index][1]
        for index in range(len(hull))
    )
    return 0.5 * abs(float(twice_area))


def _selected_span_fraction(
    eligible: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]
) -> dict[str, float | None]:
    eligible_bounds = _xy_bounds(eligible)
    selected_bounds = _xy_bounds(selected)
    if eligible_bounds is None or selected_bounds is None:
        return {"x": None, "physical_y": None}
    result: dict[str, float | None] = {}
    for output_name, source_name in (("x", "x_mm"), ("physical_y", "physical_y_mm")):
        eligible_span = eligible_bounds[source_name][1] - eligible_bounds[source_name][0]
        selected_span = selected_bounds[source_name][1] - selected_bounds[source_name][0]
        result[output_name] = (
            1.0 if eligible_span <= 1.0e-12 else float(selected_span / eligible_span)
        )
    return result


def _coverage_gate(
    eligible: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    *,
    occupied_eligible_cells: int,
    cell_area_mm2: float,
) -> dict[str, Any]:
    if len(eligible) < 2 or not selected:
        return {
            "passed": False,
            "reason": "coverage gate requires at least two eligible candidates and one selected case",
        }
    ep = np.asarray(
        [
            [row["reference_lens_x_mm"], row["reference_lens_physical_y_mm"]]
            for row in eligible
        ],
        dtype=np.float64,
    )
    sp = np.asarray(
        [
            [row["reference_lens_x_mm"], row["reference_lens_physical_y_mm"]]
            for row in selected
        ],
        dtype=np.float64,
    )
    nearest = np.sqrt(
        np.square(ep[:, None, :] - sp[None, :, :]).sum(axis=2)
    ).min(axis=1)
    candidate_scale = _nearest_neighbour_p95_mm(eligible)
    characteristic_spacing = math.sqrt(
        occupied_eligible_cells * cell_area_mm2 / len(selected)
    )
    axis_audit: dict[str, Any] = {}
    bounds_passed = True
    for axis, name in ((0, "x"), (1, "physical_y")):
        eligible_min, eligible_max = float(ep[:, axis].min()), float(ep[:, axis].max())
        selected_min, selected_max = float(sp[:, axis].min()), float(sp[:, axis].max())
        span = eligible_max - eligible_min
        tolerance = max(2.0 * candidate_scale, 0.05 * span)
        minimum_gap = max(0.0, selected_min - eligible_min)
        maximum_gap = max(0.0, eligible_max - selected_max)
        passed = minimum_gap <= tolerance and maximum_gap <= tolerance
        bounds_passed = bounds_passed and passed
        axis_audit[name] = {
            "eligible_bounds_mm": [eligible_min, eligible_max],
            "selected_bounds_mm": [selected_min, selected_max],
            "inward_gap_mm": {"minimum_side": minimum_gap, "maximum_side": maximum_gap},
            "threshold_mm": tolerance,
            "passed": passed,
        }
    p95 = float(np.percentile(nearest, 95.0))
    maximum = float(nearest.max())
    p95_threshold = characteristic_spacing + candidate_scale
    maximum_threshold = 1.25 * characteristic_spacing + candidate_scale
    nearest_passed = p95 <= p95_threshold and maximum <= maximum_threshold
    return {
        "eligible_definition": "zone intersection true-traceable intersection safety margin",
        "candidate_nearest_neighbour_p95_mm": candidate_scale,
        "occupied_eligible_area_proxy_mm2": occupied_eligible_cells * cell_area_mm2,
        "selected_count": len(selected),
        "characteristic_spacing_mm": characteristic_spacing,
        "bounds": axis_audit,
        "nearest_distance_mm": {
            "p95": p95,
            "maximum": maximum,
            "p95_threshold": p95_threshold,
            "maximum_threshold": maximum_threshold,
            "passed": nearest_passed,
        },
        "passed": bounds_passed and nearest_passed,
    }


def coverage_audit(
    payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Quantify full masks, true-traceable candidates, safe candidates and FPS coverage."""
    x, y, masks = _zone_arrays(payload)
    pitch_x = float(np.median(abs(np.diff(x))))
    pitch_y = float(np.median(abs(np.diff(y))))
    cell_area = pitch_x * pitch_y
    zone_specs = {
        "far": ("far", ("far", "far_robustness")),
        "corridor": (
            "corridor", ("corridor_upper", "corridor_middle", "corridor_lower")
        ),
        "near": ("near", ("near", "near_robustness", "near_edge_astig")),
        "peripheral_astig_left": ("astig_left", ("peripheral_left",)),
        "peripheral_astig_right": ("astig_right", ("peripheral_right",)),
    }
    corridor_range = dict(dict(payload.get("statistics", {})).get("corridor", {})).get(
        "physical_y_range_mm"
    )
    if not isinstance(corridor_range, list) or len(corridor_range) != 2:
        raise ValueError("zones statistics must declare corridor physical_y_range_mm")
    corridor_min, corridor_max = sorted(float(value) for value in corridor_range)

    safe_peripheral = [
        row for row in candidates
        if row.get("trace_status") == "ok"
        and bool(row.get("eligible"))
        and row.get("reference_partition_zone") in {"astig_left", "astig_right"}
    ]
    peripheral_pairs = _peripheral_pairs(safe_peripheral)
    pairable_by_zone = {
        "astig_left": [pair["left"] for pair in peripheral_pairs],
        "astig_right": [pair["right"] for pair in peripheral_pairs],
    }

    zones: dict[str, Any] = {}
    for mask_name, (zone_name, group_names) in zone_specs.items():
        traceable = [
            row for row in candidates
            if row.get("trace_status") == "ok" and row.get("reference_partition_zone") == zone_name
        ]
        eligible = [row for row in traceable if bool(row.get("eligible"))]
        coverage_eligible = pairable_by_zone.get(zone_name, eligible)
        selected = [row for row in cases if row.get("training_group") in group_names]
        max_nearest = p95_nearest = median_nearest = None
        if coverage_eligible and selected:
            ep = np.asarray(
                [[row["reference_lens_x_mm"], row["reference_lens_physical_y_mm"]] for row in coverage_eligible],
                dtype=np.float64,
            )
            sp = np.asarray(
                [[row["reference_lens_x_mm"], row["reference_lens_physical_y_mm"]] for row in selected],
                dtype=np.float64,
            )
            nearest = np.sqrt(np.square(ep[:, None, :] - sp[None, :, :]).sum(axis=2)).min(axis=1)
            max_nearest = float(nearest.max())
            p95_nearest = float(np.percentile(nearest, 95.0))
            median_nearest = float(np.median(nearest))
        iy, ix = np.nonzero(masks[mask_name])
        full_bounds = None if ix.size == 0 else {
            "x_mm": [float(x[ix].min()), float(x[ix].max())],
            "physical_y_mm": [float(y[iy].min()), float(y[iy].max())],
        }
        full_cells = int(ix.size)
        occupied_traceable = _occupied_mask_cells(payload, mask_name, traceable)
        occupied_eligible = _occupied_mask_cells(payload, mask_name, eligible)
        occupied_coverage_eligible = _occupied_mask_cells(
            payload, mask_name, coverage_eligible
        )
        margins = sorted({
            float(row["zone_boundary_safety_mm"])
            for row in eligible if row.get("zone_boundary_safety_mm") is not None
        })
        zones[mask_name] = {
            "full_mask_cell_count": full_cells,
            "full_mask_area_mm2": full_cells * cell_area,
            "full_mask_bounds": full_bounds,
            "traceable_candidate_count": len(traceable),
            "traceable_occupied_mask_cell_count": occupied_traceable,
            "traceable_occupied_mask_cell_fraction": occupied_traceable / full_cells if full_cells else None,
            "traceable_bounds": _xy_bounds(traceable),
            "eligible_candidate_count": len(eligible),
            "eligible_occupied_mask_cell_count": occupied_eligible,
            "eligible_occupied_mask_cell_fraction": occupied_eligible / full_cells if full_cells else None,
            "eligible_bounds": _xy_bounds(eligible),
            "coverage_eligible_definition": (
                "safe and exact field/rear-mirror pairable candidates"
                if zone_name in pairable_by_zone
                else "safe candidates"
            ),
            "coverage_eligible_candidate_count": len(coverage_eligible),
            "coverage_eligible_occupied_mask_cell_count": occupied_coverage_eligible,
            "coverage_eligible_bounds": _xy_bounds(coverage_eligible),
            "selected_case_count": len(selected),
            "selected_bounds": _xy_bounds(selected),
            "selected_span_fraction_of_eligible": _selected_span_fraction(coverage_eligible, selected),
            "eligible_convex_hull_envelope_area_mm2": _convex_hull_area_mm2(coverage_eligible),
            "selected_convex_hull_envelope_area_mm2": _convex_hull_area_mm2(selected),
            "eligible_to_selected_nearest_distance_mm": {
                "median": median_nearest,
                "p95": p95_nearest,
                "maximum_coverage_radius": max_nearest,
            },
            "zone_boundary_safety_mm": margins,
            "coverage_gate": _coverage_gate(
                coverage_eligible,
                selected,
                occupied_eligible_cells=occupied_coverage_eligible,
                cell_area_mm2=cell_area,
            ),
        }
        if zone_name in pairable_by_zone:
            predicates = {
                "upper": lambda value: value > corridor_max,
                "middle": lambda value: corridor_min <= value <= corridor_max,
                "lower": lambda value: value < corridor_min,
            }
            band_gates: dict[str, Any] = {}
            for band, predicate in predicates.items():
                band_eligible = [
                    row for row in coverage_eligible
                    if predicate(float(row["reference_lens_physical_y_mm"]))
                ]
                band_selected = [
                    row for row in selected if row.get("peripheral_band") == band
                ]
                band_occupied = _occupied_mask_cells(
                    payload, mask_name, band_eligible
                )
                band_gates[band] = {
                    "eligible_candidate_count": len(band_eligible),
                    "selected_case_count": len(band_selected),
                    "eligible_bounds": _xy_bounds(band_eligible),
                    "selected_bounds": _xy_bounds(band_selected),
                    "coverage_gate": _coverage_gate(
                        band_eligible,
                        band_selected,
                        occupied_eligible_cells=band_occupied,
                        cell_area_mm2=cell_area,
                    ),
                }
            zones[mask_name]["peripheral_band_coverage"] = band_gates
    failed_coverage = [
        zone_name
        for zone_name, zone in zones.items()
        if not bool(zone["coverage_gate"]["passed"])
    ]
    failed_coverage.extend(
        f"{zone_name}:{band_name}"
        for zone_name, zone in zones.items()
        for band_name, band in zone.get("peripheral_band_coverage", {}).items()
        if not bool(band["coverage_gate"]["passed"])
    )
    return {
        "schema_version": 3,
        "interpretation": (
            "Coverage is contracted on zone intersection true-traceable candidates intersection safety margin; "
            "the complete raster mask can contain physically unreachable cells. Occupied-cell fractions count "
            "only raster cells hit by the finite candidate grid and are not continuous-area claims. Convex-hull "
            "values are selected/candidate envelopes only and are likewise not physical zone-area claims."
        ),
        "mask_grid_pitch_mm": {"x": pitch_x, "physical_y": pitch_y},
        "candidate_count": len(candidates),
        "trace_success_count": sum(row.get("trace_status") == "ok" for row in candidates),
        "trace_failure_count": sum(row.get("trace_status") != "ok" for row in candidates),
        "unclassified_trace_success_count": sum(
            row.get("trace_status") == "ok" and row.get("reference_partition_zone") is None
            for row in candidates
        ),
        "overall_passed": not failed_coverage,
        "failed_coverage_gates": failed_coverage,
        "zones": zones,
    }


def _validate_selected_case_geometry(
    payload: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    sampling_contract: Mapping[str, Any],
) -> dict[str, Any]:
    zone_margins = sampling_contract.get("zone_boundary_safety_mm")
    if not isinstance(zone_margins, Mapping) or not {"default", "corridor"}.issubset(zone_margins):
        raise ValueError("sampling contract must declare default and corridor zone margins")
    if "aperture_edge_safety_mm" not in sampling_contract:
        raise ValueError("sampling contract must declare aperture-edge safety margin")
    object_distances = sampling_contract.get("object_distance_mm")
    if not isinstance(object_distances, Mapping) or not {
        "far", "intermediate", "near"
    }.issubset(object_distances):
        raise ValueError("sampling contract must declare far/intermediate/near distances")
    band_distances = sampling_contract.get("peripheral_band_distance_mm")
    if not isinstance(band_distances, Mapping) or not {
        "upper", "middle", "lower"
    }.issubset(band_distances):
        raise ValueError("sampling contract must declare upper/middle/lower distances")
    resolved_distances = {
        name: float(object_distances[name])
        for name in ("far", "intermediate", "near")
    }
    resolved_band_distances = {
        name: float(band_distances[name])
        for name in ("upper", "middle", "lower")
    }

    def valid_object_distance(value: float, *, allow_infinity: bool) -> bool:
        return value > 0.0 and (
            math.isfinite(value) or (allow_infinity and math.isinf(value))
        )

    for name, value in resolved_distances.items():
        if not valid_object_distance(value, allow_infinity=name == "far"):
            raise ValueError(
                f"training {name} object distance must be positive"
                + (" or Infinity" if name == "far" else " and finite")
            )
    for name, value in resolved_band_distances.items():
        if not valid_object_distance(value, allow_infinity=True):
            raise ValueError(
                f"peripheral {name} object distance must be positive"
                + " or Infinity"
            )
    required_band_mapping = {
        "upper": resolved_distances["far"],
        "middle": resolved_distances["far"],
        "lower": resolved_distances["far"],
    }
    if resolved_band_distances != required_band_mapping:
        raise ValueError(
            "surface-only peripheral cases must all retain the far reference distance"
        )
    aperture_margin = float(sampling_contract["aperture_edge_safety_mm"])
    resolved_zone_margins = {
        "default": float(zone_margins["default"]),
        "corridor": float(zone_margins["corridor"]),
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (*resolved_zone_margins.values(), aperture_margin)
    ):
        raise ValueError("zone and aperture safety margins must be finite and non-negative")
    corridor_range = dict(dict(payload.get("statistics", {})).get("corridor", {})).get("physical_y_range_mm")
    if not isinstance(corridor_range, list) or len(corridor_range) != 2:
        raise ValueError("zones statistics must declare corridor physical_y_range_mm")
    corridor_min, corridor_max = sorted(float(value) for value in corridor_range)
    minimum_zone_clearance = math.inf
    minimum_aperture_clearance = math.inf
    maximum_assigned_distance_retrace_error = 0.0
    for case in cases:
        group = str(case["training_group"])
        expected_zone = GROUP_TO_ZONE[group]
        zone_margin = resolved_zone_margins[
            "corridor" if expected_zone == "corridor" else "default"
        ]
        for prefix, recorded_name, x_key, y_key in (
            ("reference", "reference_partition_zone", "reference_lens_x_mm", "reference_lens_physical_y_mm"),
            ("case", "case_position_partition_zone", "case_lens_x_mm", "case_lens_physical_y_mm"),
        ):
            x_mm, y_mm = float(case[x_key]), float(case[y_key])
            classified = classify_partition_point(payload, x_mm=x_mm, physical_y_mm=y_mm)
            if case.get(recorded_name) != classified or classified != expected_zone:
                raise ValueError(f"{case['case_id']} {prefix} rear point is {classified!r}, expected {expected_zone!r}")
            zone_clearance = _inside_clearance_mm(payload, _mask_name(expected_zone), x_mm, y_mm)
            aperture_clearance = _inside_clearance_mm(payload, "monitored", x_mm, y_mm)
            if zone_clearance < zone_margin or aperture_clearance < aperture_margin:
                raise ValueError(
                    f"{case['case_id']} {prefix} rear point violates safety margins: "
                    f"zone={zone_clearance:.6g}/{zone_margin:.6g} mm, "
                    f"aperture={aperture_clearance:.6g}/{aperture_margin:.6g} mm"
                )
            minimum_zone_clearance = min(minimum_zone_clearance, zone_clearance)
            minimum_aperture_clearance = min(minimum_aperture_clearance, aperture_clearance)
        assigned_distance_retrace_error = max(
            abs(float(case["reference_lens_x_mm"]) - float(case["case_lens_x_mm"])),
            abs(
                float(case["reference_lens_physical_y_mm"])
                - float(case["case_lens_physical_y_mm"])
            ),
        )
        if assigned_distance_retrace_error > REFERENCE_RETRACE_TOLERANCE_MM:
            raise ValueError(
                f"{case['case_id']} assigned-distance rear point differs from the "
                "common FPS reference point, so reference-domain coverage cannot be "
                f"transferred: error={assigned_distance_retrace_error:.6g} mm"
            )
        maximum_assigned_distance_retrace_error = max(
            maximum_assigned_distance_retrace_error,
            assigned_distance_retrace_error,
        )
        if group in PERIPHERAL_GROUPS:
            task_y = float(case["case_lens_physical_y_mm"])
            expected_band = "upper" if task_y > corridor_max else "lower" if task_y < corridor_min else "middle"
            if case.get("peripheral_band") != expected_band:
                raise ValueError(
                    f"{case['case_id']} task-distance rear point belongs to peripheral band "
                    f"{expected_band}, not {case.get('peripheral_band')}"
                )
            expected_distance = resolved_band_distances[expected_band]
        elif group == "far":
            expected_distance = resolved_distances["far"]
        elif group in {"far_robustness", "near_robustness"}:
            expected_distance = resolved_distances["intermediate"]
        elif group in {"near", "near_edge_astig"}:
            expected_distance = resolved_distances["near"]
        elif group.startswith("corridor_"):
            local_add = float(case.get("corridor_local_add_D", math.nan))
            if not math.isfinite(local_add) or local_add <= 0.0:
                raise ValueError(f"{case['case_id']} has invalid corridor_local_add_D")
            expected_distance = 1000.0 / local_add
        else:
            raise ValueError(f"unknown training group: {group}")
        if not math.isclose(float(case["distance_mm"]), expected_distance, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"{case['case_id']} distance is {case['distance_mm']}, "
                f"expected {expected_distance} mm"
            )

    corridor_bands = {
        "corridor_upper": (0.2, 0.5, False),
        "corridor_middle": (0.5, 1.3, False),
        "corridor_lower": (1.3, 2.0, True),
    }
    for group, (lower, upper, include_upper) in corridor_bands.items():
        members = [case for case in cases if case.get("training_group") == group]
        if len(members) != TRAINING_GROUP_COUNTS[group]:
            raise ValueError(f"{group} count does not match the training contract")
        for case in members:
            value = float(case["corridor_local_add_D"])
            in_band = lower <= value <= upper if include_upper else lower <= value < upper
            if not in_band:
                raise ValueError(
                    f"{case['case_id']} ADD {value:.6g} D is outside {group} band"
                )
    for case in cases:
        if case.get("training_group") == "near_edge_astig" and abs(
            float(case["reference_lens_x_mm"])
        ) <= 10.0:
            raise ValueError(f"{case['case_id']} violates near-edge |x| > 10 mm")
    for group in PERIPHERAL_GROUPS:
        for band, expected_count in PERIPHERAL_BAND_COUNTS.items():
            actual_count = sum(
                case.get("training_group") == group
                and case.get("peripheral_band") == band
                for case in cases
            )
            if actual_count != expected_count:
                raise ValueError(
                    f"{group} {band} count is {actual_count}, expected {expected_count}"
                )
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        if case.get("training_group") in PERIPHERAL_GROUPS:
            pair_id = str(case.get("peripheral_pair_id", ""))
            if not pair_id:
                raise ValueError(f"peripheral case {case['case_id']} has no pair id")
            pairs.setdefault(pair_id, []).append(case)
    maximum_reference_mirror_error = 0.0
    maximum_task_mirror_error = 0.0
    for pair_id, members in pairs.items():
        by_group = {str(case["training_group"]): case for case in members}
        if len(members) != 2 or set(by_group) != set(PERIPHERAL_GROUPS):
            raise ValueError(f"{pair_id} must contain exactly one left/right case")
        left, right = by_group["peripheral_left"], by_group["peripheral_right"]
        if (
            abs(float(left["field_x_deg"]) + float(right["field_x_deg"])) > 1.0e-9
            or abs(float(left["field_y_deg"]) - float(right["field_y_deg"])) > 1.0e-9
            or float(left["distance_mm"]) != float(right["distance_mm"])
            or left.get("peripheral_band") != right.get("peripheral_band")
        ):
            raise ValueError(f"{pair_id} is not an exact field/distance mirror pair")
        reference_error = max(
            abs(float(left["reference_lens_x_mm"]) + float(right["reference_lens_x_mm"])),
            abs(float(left["reference_lens_physical_y_mm"]) - float(right["reference_lens_physical_y_mm"])),
        )
        task_error = max(
            abs(float(left["case_lens_x_mm"]) + float(right["case_lens_x_mm"])),
            abs(float(left["case_lens_physical_y_mm"]) - float(right["case_lens_physical_y_mm"])),
        )
        if max(reference_error, task_error) > PERIPHERAL_REAR_MIRROR_TOLERANCE_MM:
            raise ValueError(
                f"{pair_id} is not mirrored on the PAL rear surface: "
                f"reference={reference_error:.6g} mm, task={task_error:.6g} mm"
            )
        maximum_reference_mirror_error = max(maximum_reference_mirror_error, reference_error)
        maximum_task_mirror_error = max(maximum_task_mirror_error, task_error)
    return {
        "passed": True,
        "minimum_zone_clearance_mm": minimum_zone_clearance,
        "minimum_aperture_clearance_mm": minimum_aperture_clearance,
        "maximum_assigned_distance_retrace_error_mm": (
            maximum_assigned_distance_retrace_error
        ),
        "assigned_distance_retrace_tolerance_mm": REFERENCE_RETRACE_TOLERANCE_MM,
        "peripheral_pair_count": len(pairs),
        "maximum_reference_rear_mirror_error_mm": maximum_reference_mirror_error,
        "maximum_task_rear_mirror_error_mm": maximum_task_mirror_error,
        "mirror_tolerance_mm": PERIPHERAL_REAR_MIRROR_TOLERANCE_MM,
    }


def _validate_selected_candidate_membership(
    candidates: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_ids = [str(row.get("candidate_id", "")) for row in candidates]
    if not all(candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate grid must contain unique non-empty candidate ids")
    case_ids = [str(case.get("case_id", "")) for case in cases]
    selected_candidate_ids = [str(case.get("candidate_id", "")) for case in cases]
    if not all(case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("training cases must contain unique non-empty case ids")
    if not all(selected_candidate_ids):
        raise ValueError("training cases must select non-empty candidate ids")
    lookup = {str(row["candidate_id"]): row for row in candidates}
    maximum_reference_retrace_error = 0.0
    for case in cases:
        candidate_id = str(case["candidate_id"])
        candidate = lookup.get(candidate_id)
        if candidate is None:
            raise ValueError(f"training case selects unknown candidate {candidate_id}")
        if candidate.get("trace_status") != "ok" or not bool(candidate.get("eligible")):
            raise ValueError(f"training case selects non-eligible candidate {candidate_id}")
        if (
            float(case["field_x_deg"]) != float(candidate["field_x_deg"])
            or float(case["field_y_deg"]) != float(candidate["field_y_deg"])
            or case.get("reference_partition_zone") != candidate.get("reference_partition_zone")
        ):
            raise ValueError(f"training case field/zone does not match candidate {candidate_id}")
        retrace_error = max(
            abs(float(case["reference_lens_x_mm"]) - float(candidate["reference_lens_x_mm"])),
            abs(
                float(case["reference_lens_physical_y_mm"])
                - float(candidate["reference_lens_physical_y_mm"])
            ),
        )
        if retrace_error > REFERENCE_RETRACE_TOLERANCE_MM:
            raise ValueError(
                f"training case reference retrace does not match candidate {candidate_id}: "
                f"error={retrace_error:.6g} mm"
            )
        maximum_reference_retrace_error = max(
            maximum_reference_retrace_error, retrace_error
        )
    return {
        "passed": True,
        "selected_candidate_count": len(selected_candidate_ids),
        "unique_case_id_count": len(set(case_ids)),
        "unique_candidate_id_count": len(set(selected_candidate_ids)),
        "candidate_reuse_count": len(selected_candidate_ids) - len(set(selected_candidate_ids)),
        "maximum_reference_retrace_error_mm": maximum_reference_retrace_error,
        "reference_retrace_tolerance_mm": REFERENCE_RETRACE_TOLERANCE_MM,
    }


def _run_case_layout_plotter(
    *, output: Path, zones_json: str | Path, candidate_json: Path, manifest_json: Path
) -> None:
    """Render plots in a clean no-Torch process to avoid duplicate Windows OpenMP runtimes."""
    script = Path(__file__).with_name("pal_case_layout_plotter.py")
    command = [
        sys.executable,
        str(script),
        "--zones", str(Path(zones_json).resolve()),
        "--candidates", str(candidate_json.resolve()),
        "--manifest", str(manifest_json.resolve()),
        "--output", str(output.resolve()),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "case-layout plotter failed with exit code "
            f"{completed.returncode}: stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )


def write_preoptimization_artifacts(
    *, output_dir: str | Path, excel_path: str | Path,
    zones_json: str | Path, candidates: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]], reference_distance_mm: float,
    sampling_contract: Mapping[str, Any],
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = _read_json(zones_json)
    counts = {group: sum(case.get("training_group") == group for case in cases) for group in TRAINING_GROUP_COUNTS}
    if counts != TRAINING_GROUP_COUNTS or len(cases) != TOTAL_TRAINING_CASES:
        raise ValueError(f"invalid {TOTAL_TRAINING_CASES}-case group contract: {counts}")
    mismatches = [case["case_id"] for case in cases if case.get("reference_partition_zone") != GROUP_TO_ZONE[case["training_group"]]]
    if mismatches:
        raise ValueError("selected reference rays do not belong to declared partitions: " + ", ".join(mismatches))
    membership_audit = _validate_selected_candidate_membership(candidates, cases)
    geometry_audit = _validate_selected_case_geometry(
        payload, cases, candidates, sampling_contract
    )
    candidate_payload = {
        "schema_version": 1, "reference_distance_mm": reference_distance_mm,
        "sampling_contract": dict(sampling_contract), "candidate_count": len(candidates),
        "trace_failure_count": sum(row.get("trace_status") != "ok" for row in candidates),
        "eligible_count": sum(bool(row.get("eligible")) for row in candidates),
        "candidates": [dict(row) for row in candidates],
    }
    candidate_json = output / "candidate_fields.json"
    candidate_json.write_text(json.dumps(candidate_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reachability = coverage_audit(payload, candidates, cases)
    coverage_json = output / "coverage_audit.json"
    coverage_json.write_text(
        json.dumps(reachability, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not bool(reachability["overall_passed"]):
        raise ValueError(
            "selected cases fail eligible-domain coverage gates: "
            + ", ".join(reachability["failed_coverage_gates"])
        )

    manifest = {
        "schema_version": 6,
        "purpose": f"pal_nurbs_dense_field_fps_{TOTAL_TRAINING_CASES}_case_contract",
        "source": {
            "excel": {"path": str(excel_path), "sha256": _sha256_file(excel_path)},
            "zones_json": {"path": str(zones_json), "sha256": _sha256_file(zones_json)},
        },
        "reference_geometry": {"object_distance_mm": reference_distance_mm, "ray": "aimed centre-pupil ray", "surface": "Original PAL rear surface", "coordinates": "physical local-surface x/y in mm"},
        "sampling_contract": dict(sampling_contract),
        "objective_contract": {
            "denominator": "per-case Original PAL baseline for the routed physical metric",
            "metrics": "far=CSF-MTF loss; corridor/near=Z4 OPD mm^2; peripheral=surface A_D",
            "J": "sum(group_weight * mean(per-case normalized score))",
            "aggregation_order": "mean within each of ten groups before explicit group weighting",
        },
        "coverage_audit": {
            "path": str(coverage_json.resolve()),
            "sha256": _sha256_file(coverage_json),
            "overall_passed": True,
        },
        "group_counts": counts,
        "candidate_membership_audit": membership_audit,
        "topology_audit": partition_audit(payload, cases),
        "case_geometry_audit": geometry_audit,
        "cases": [dict(case) for case in cases],
    }
    manifest_json = output / "case_manifest.json"
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _run_case_layout_plotter(
        output=output,
        zones_json=zones_json,
        candidate_json=candidate_json,
        manifest_json=manifest_json,
    )
    return output
