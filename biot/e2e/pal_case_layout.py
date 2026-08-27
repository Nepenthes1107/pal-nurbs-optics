"""固定多物距 11×11 FOV 网格及 PAL 分区权重。

本模块只负责非可微的实验布局：

* 三个固定物距 ``500 mm``、``1000 mm`` 和无穷远；
* 三个物距共用同一个 11×11 视场角网格；
* 用基线 PAL 后表面的 chief/reference ray 落点确定分区；
* 将显式权重矩阵展开为每个 case 的归一化 loss 权重。

这里不做候选池、FPS、资格筛选、覆盖率审计或 PSF 渲染。任何固定网格
case 的主光线追迹失败、落在 monitored aperture 外或权重配置不完整都会
直接失败，不能通过丢弃 case 继续运行。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PARTITION_ZONES = (
    "far",
    "corridor",
    "near",
    "astig_left",
    "astig_right",
)
MASK_NAME_BY_ZONE = {
    "far": "far",
    "corridor": "corridor",
    "near": "near",
    "astig_left": "peripheral_astig_left",
    "astig_right": "peripheral_astig_right",
}
PARTITION_MASK_ORDER = tuple(MASK_NAME_BY_ZONE[zone] for zone in PARTITION_ZONES)


@dataclass(frozen=True)
class DistanceSpec:
    """One fixed object-distance condition.

    ``object_distance_mm=math.inf`` is passed to the Excel/BIOT loader as the
    literal ``Infinity`` condition.  The label is used in case IDs and weight
    configuration, so no finite approximation is substituted for infinity.
    """

    label: str
    object_distance_mm: float
    focus_zone: str

    @property
    def serialized_distance(self) -> float | str:
        return "Infinity" if math.isinf(self.object_distance_mm) else float(self.object_distance_mm)


DISTANCE_SPECS = (
    DistanceSpec("D500", 500.0, "near"),
    DistanceSpec("D1000", 1000.0, "corridor"),
    DistanceSpec("Dinf", math.inf, "far"),
)
DISTANCE_LABELS = tuple(spec.label for spec in DISTANCE_SPECS)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _zone_arrays(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    try:
        x = np.asarray(payload["x_mm"], dtype=np.float64)
        physical_y = np.asarray(payload["physical_y_mm"], dtype=np.float64)
        raw_masks = payload["masks"]
    except KeyError as exc:
        raise ValueError(f"zones payload is missing {exc.args[0]!r}") from exc
    if not isinstance(raw_masks, Mapping):
        raise ValueError("zones masks must be an object")
    if x.ndim != 1 or physical_y.ndim != 1 or x.size < 2 or physical_y.size < 2:
        raise ValueError("zone coordinates must be non-trivial 1-D arrays")
    if not np.isfinite(x).all() or not np.isfinite(physical_y).all():
        raise ValueError("zone coordinates must be finite")
    masks = {
        str(name): np.asarray(mask, dtype=bool)
        for name, mask in raw_masks.items()
    }
    expected_shape = (physical_y.size, x.size)
    if any(mask.shape != expected_shape for mask in masks.values()):
        raise ValueError("zone masks do not match physical_y_mm/x_mm shape")
    required = set(PARTITION_MASK_ORDER) | {"monitored"}
    missing = sorted(required - set(masks))
    if missing:
        raise ValueError("zones payload is missing masks: " + ", ".join(missing))
    return x, physical_y, masks


@dataclass(frozen=True)
class PartitionClassification:
    zone: str
    mode: str
    nearest_partition_distance_mm: float
    grid_x_mm: float
    grid_y_mm: float


class PartitionMap:
    """Validated PAL partition masks with an explicit gap-extension rule.

    The stored masks are authoritative when a point falls in exactly one
    partition cell.  The current masks intentionally leave some monitored
    safety-band cells unlabelled.  For a fixed FOV grid those cells are mapped
    to the closest stored partition cell in physical local-surface ``(x,y)``;
    this rule is deterministic and recorded in each case's ``partition_mode``.
    A point outside ``monitored`` is an input/layout error and raises.
    """

    def __init__(
        self,
        x_mm: np.ndarray,
        physical_y_mm: np.ndarray,
        masks: Mapping[str, np.ndarray],
        *,
        nearest_tie_tolerance_mm: float = 1.0e-10,
    ) -> None:
        self.x_mm = np.asarray(x_mm, dtype=np.float64)
        self.physical_y_mm = np.asarray(physical_y_mm, dtype=np.float64)
        self.masks = {str(name): np.asarray(mask, dtype=bool) for name, mask in masks.items()}
        self.pitch_x_mm = float(np.median(np.abs(np.diff(self.x_mm))))
        self.pitch_y_mm = float(np.median(np.abs(np.diff(self.physical_y_mm))))
        if not math.isfinite(self.pitch_x_mm) or self.pitch_x_mm <= 0.0:
            raise ValueError("zone x coordinate pitch must be finite and positive")
        if not math.isfinite(self.pitch_y_mm) or self.pitch_y_mm <= 0.0:
            raise ValueError("zone y coordinate pitch must be finite and positive")
        self.nearest_tie_tolerance_mm = float(nearest_tie_tolerance_mm)
        if not math.isfinite(self.nearest_tie_tolerance_mm) or self.nearest_tie_tolerance_mm < 0.0:
            raise ValueError("nearest partition tie tolerance must be finite and non-negative")
        self._partition_points: dict[str, np.ndarray] = {}
        for zone in PARTITION_ZONES:
            mask_name = MASK_NAME_BY_ZONE[zone]
            rows, columns = np.nonzero(self.masks[mask_name])
            if rows.size == 0:
                raise ValueError(f"partition mask {mask_name!r} is empty")
            self._partition_points[zone] = np.column_stack(
                (self.x_mm[columns], self.physical_y_mm[rows])
            )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PartitionMap":
        x, physical_y, masks = _zone_arrays(payload)
        return cls(x, physical_y, masks)

    @classmethod
    def from_json(cls, path: str | Path) -> "PartitionMap":
        return cls.from_payload(_read_json(path))

    def _nearest_cell(self, x_mm: float, physical_y_mm: float) -> tuple[int, int, float, float]:
        if not math.isfinite(float(x_mm)) or not math.isfinite(float(physical_y_mm)):
            raise ValueError("partition point must be finite")
        ix = int(np.argmin(np.abs(self.x_mm - float(x_mm))))
        iy = int(np.argmin(np.abs(self.physical_y_mm - float(physical_y_mm))))
        grid_x = float(self.x_mm[ix])
        grid_y = float(self.physical_y_mm[iy])
        dx = abs(grid_x - float(x_mm))
        dy = abs(grid_y - float(physical_y_mm))
        if dx > 0.5 * self.pitch_x_mm + 1.0e-9 or dy > 0.5 * self.pitch_y_mm + 1.0e-9:
            raise ValueError(
                "partition point is outside the stored zone grid: "
                f"({float(x_mm):.9g}, {float(physical_y_mm):.9g}) mm"
            )
        return ix, iy, grid_x, grid_y

    def classify(self, *, x_mm: float, physical_y_mm: float) -> PartitionClassification:
        ix, iy, grid_x, grid_y = self._nearest_cell(x_mm, physical_y_mm)
        if not bool(self.masks["monitored"][iy, ix]):
            raise ValueError(
                "fixed FOV point falls outside monitored PAL aperture: "
                f"({float(x_mm):.9g}, {float(physical_y_mm):.9g}) mm"
            )
        active = [
            zone
            for zone in PARTITION_ZONES
            if bool(self.masks[MASK_NAME_BY_ZONE[zone]][iy, ix])
        ]
        if len(active) > 1:
            raise ValueError(
                "PAL partition masks overlap at grid cell "
                f"({grid_x:.9g}, {grid_y:.9g}) mm: {active}"
            )
        if len(active) == 1:
            return PartitionClassification(
                zone=active[0],
                mode="stored_mask",
                nearest_partition_distance_mm=0.0,
                grid_x_mm=grid_x,
                grid_y_mm=grid_y,
            )

        point = np.asarray((float(x_mm), float(physical_y_mm)), dtype=np.float64)
        distances = {
            zone: float(np.linalg.norm(points - point, axis=1).min())
            for zone, points in self._partition_points.items()
        }
        minimum = min(distances.values())
        tied = [
            zone
            for zone in PARTITION_ZONES
            if abs(distances[zone] - minimum) <= self.nearest_tie_tolerance_mm
        ]
        # The order is part of the fixed method identity.  Ties occur only on
        # a geometric boundary; choosing the sealed order is deterministic and
        # is recorded as a tie mode rather than hidden as a dropped sample.
        chosen = tied[0]
        mode = "nearest_partition_cell_tie" if len(tied) > 1 else "nearest_partition_cell"
        return PartitionClassification(
            zone=chosen,
            mode=mode,
            nearest_partition_distance_mm=minimum,
            grid_x_mm=grid_x,
            grid_y_mm=grid_y,
        )


def classify_partition_point(
    zones_payload: Mapping[str, Any], *, x_mm: float, physical_y_mm: float
) -> str:
    """Classify one point with the same explicit rule used by training layout."""
    return PartitionMap.from_payload(zones_payload).classify(
        x_mm=x_mm, physical_y_mm=physical_y_mm
    ).zone


def generate_fov_grid(
    *, field_min_deg: float, field_max_deg: float, count: int = 11
) -> list[dict[str, Any]]:
    """Return one deterministic square 11×11 field-angle grid in degree."""
    count = int(count)
    if count != 11:
        raise ValueError("the multidistance method requires an 11x11 FOV grid")
    field_min = float(field_min_deg)
    field_max = float(field_max_deg)
    if not math.isfinite(field_min) or not math.isfinite(field_max) or field_max <= field_min:
        raise ValueError("FOV bounds must be finite with max > min")
    values = np.linspace(field_min, field_max, count, dtype=np.float64)
    result: list[dict[str, Any]] = []
    for row, field_y in enumerate(values):
        for column, field_x in enumerate(values):
            fx = 0.0 if abs(float(field_x)) < 1.0e-14 else float(field_x)
            fy = 0.0 if abs(float(field_y)) < 1.0e-14 else float(field_y)
            result.append(
                {
                    "grid_row": row,
                    "grid_column": column,
                    "field_x_deg": fx,
                    "field_y_deg": fy,
                }
            )
    return result


def _validate_weight_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    required_top = {"schema_version", "zone_total_weight", "distance_fraction_by_zone"}
    missing = sorted(required_top - set(payload))
    if missing:
        raise ValueError("weight configuration is missing: " + ", ".join(missing))
    if int(payload["schema_version"]) != 1:
        raise ValueError("unsupported multidistance weight schema")
    raw_zone_total = payload["zone_total_weight"]
    raw_fraction = payload["distance_fraction_by_zone"]
    if not isinstance(raw_zone_total, Mapping) or not isinstance(raw_fraction, Mapping):
        raise ValueError("weight configuration fields must be objects")
    zone_total: dict[str, float] = {}
    for zone in PARTITION_ZONES:
        if zone not in raw_zone_total:
            raise ValueError(f"weight configuration lacks zone total for {zone}")
        value = float(raw_zone_total[zone])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"zone total weight for {zone} must be finite and positive")
        zone_total[zone] = value
    total_zone_weight = sum(zone_total.values())
    if abs(total_zone_weight - 1.0) > 1.0e-12:
        raise ValueError(f"zone total weights must sum to 1, got {total_zone_weight}")

    fraction: dict[str, dict[str, float]] = {}
    for zone in PARTITION_ZONES:
        row = raw_fraction.get(zone)
        if not isinstance(row, Mapping):
            raise ValueError(f"distance fractions for {zone} must be an object")
        fraction[zone] = {}
        for label in DISTANCE_LABELS:
            if label not in row:
                raise ValueError(f"distance fractions for {zone} lack {label}")
            value = float(row[label])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"distance fraction {zone}/{label} must be finite and non-negative")
            fraction[zone][label] = value
        row_total = sum(fraction[zone].values())
        if abs(row_total - 1.0) > 1.0e-12:
            raise ValueError(f"distance fractions for {zone} must sum to 1, got {row_total}")
    return {
        "schema_version": 1,
        "zone_total_weight": zone_total,
        "distance_fraction_by_zone": fraction,
        "description": payload.get("description"),
    }


def load_weight_spec(path: str | Path) -> dict[str, Any]:
    return _validate_weight_spec(_read_json(path))


def attach_objective_weights(
    cases: Sequence[Mapping[str, Any]], weight_spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Assign density-independent zone/distance masses to all fixed-grid cases."""
    validated = _validate_weight_spec(weight_spec)
    if not cases:
        raise ValueError("cannot assign weights to an empty case set")
    counts: dict[tuple[str, str], int] = {}
    for case in cases:
        key = (str(case["zone"]), str(case["distance_label"]))
        if key[0] not in PARTITION_ZONES or key[1] not in DISTANCE_LABELS:
            raise ValueError(f"unknown zone/distance in case {case.get('case_id')}: {key}")
        counts[key] = counts.get(key, 0) + 1
    expected_keys = {
        (zone, label)
        for zone in PARTITION_ZONES
        for label in DISTANCE_LABELS
    }
    missing = sorted(expected_keys - set(counts))
    if missing:
        raise ValueError(
            "fixed layout lacks a positive-count zone/distance combination: "
            + ", ".join(f"{zone}/{label}" for zone, label in missing)
        )
    weighted: list[dict[str, Any]] = []
    for case in cases:
        zone = str(case["zone"])
        label = str(case["distance_label"])
        count = counts[(zone, label)]
        mass = float(validated["zone_total_weight"][zone]) * float(
            validated["distance_fraction_by_zone"][zone][label]
        )
        raw_weight = mass / float(count)
        weighted.append(
            {
                **dict(case),
                "zone_distance_mass": mass,
                "objective_weight": raw_weight,
            }
        )
    result = weighted
    total = sum(float(case["objective_weight"]) for case in result)
    if abs(total - 1.0) > 1.0e-12:
        raise ValueError(
            "expanded objective weights are not normalized without renormalization: "
            f"{total}"
        )
    return result


def build_multidistance_layout(
    *,
    field_min_deg: float,
    field_max_deg: float,
    partition_map: PartitionMap,
    weight_spec: Mapping[str, Any],
    trace_reference: Callable[[float, float, float], tuple[float, float]],
    field_count: int = 11,
    distance_specs: Sequence[DistanceSpec] = DISTANCE_SPECS,
    prefix_cases: Sequence[Mapping[str, Any]] = (),
    progress_callback: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Build all ``3 × 11 × 11`` cases without selection or dropping."""
    specs = tuple(distance_specs)
    if tuple(spec.label for spec in specs) != DISTANCE_LABELS:
        raise ValueError("distance specs must be exactly D500, D1000, Dinf")
    field_grid = generate_fov_grid(
        field_min_deg=field_min_deg, field_max_deg=field_max_deg, count=field_count
    )
    expected_count = len(specs) * len(field_grid)
    prefix = [dict(row) for row in prefix_cases]
    if len(prefix) > expected_count:
        raise ValueError("layout progress exceeds the fixed multidistance case count")
    cases: list[dict[str, Any]] = []
    for distance_index, spec in enumerate(specs):
        for field_index, field in enumerate(field_grid):
            case_index = distance_index * len(field_grid) + field_index
            if case_index < len(prefix):
                cases.append(dict(prefix[case_index]))
                continue
            fx = float(field["field_x_deg"])
            fy = float(field["field_y_deg"])
            case_id = f"{spec.label}_r{int(field['grid_row']):02d}_c{int(field['grid_column']):02d}"
            x_mm, physical_y_mm = trace_reference(spec.object_distance_mm, fx, fy)
            classification = partition_map.classify(
                x_mm=float(x_mm), physical_y_mm=float(physical_y_mm)
            )
            cases.append(
                {
                    "case_index": case_index,
                    "case_id": case_id,
                    "distance_label": spec.label,
                    "object_distance_mm": spec.serialized_distance,
                    "focus_zone": spec.focus_zone,
                    "grid_row": int(field["grid_row"]),
                    "grid_column": int(field["grid_column"]),
                    "field_x_deg": fx,
                    "field_y_deg": fy,
                    "partition_x_mm": float(x_mm),
                    "partition_physical_y_mm": float(physical_y_mm),
                    "zone": classification.zone,
                    "partition_mode": classification.mode,
                    "nearest_partition_distance_mm": classification.nearest_partition_distance_mm,
                    "partition_grid_x_mm": classification.grid_x_mm,
                    "partition_grid_physical_y_mm": classification.grid_y_mm,
                }
            )
            if progress_callback is not None:
                progress_callback(tuple(cases))
    if len(cases) != expected_count:
        raise RuntimeError(
            f"fixed multidistance layout has {len(cases)} cases, expected {expected_count}"
        )
    ids = [str(case["case_id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("fixed multidistance case IDs are not unique")
    return attach_objective_weights(cases, weight_spec)
