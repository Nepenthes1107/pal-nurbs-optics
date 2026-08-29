from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from biot.e2e.pal_case_layout import (
    PERIPHERAL_BAND_COUNTS,
    TRAINING_GROUP_COUNTS,
    classify_partition_point,
    generate_dense_candidate_fields,
    select_training_cases,
    trace_candidate_fields,
    write_preoptimization_artifacts,
)


def _zones_payload() -> dict[str, object]:
    x = [float(value) for value in range(-30, 31)]
    y = [float(value) for value in range(-30, 31)]
    far = [[int(-14 <= xx <= 14 and yy >= 1) for xx in x] for yy in y]
    corridor = [[int(-14 <= xx <= 14 and -12 <= yy <= 0) for xx in x] for yy in y]
    near = [[int(-14 <= xx <= 14 and yy <= -13) for xx in x] for yy in y]
    left = [[int(xx <= -15) for xx in x] for _ in y]
    right = [[int(xx >= 15) for xx in x] for _ in y]
    monitored = [[1 for _ in x] for _ in y]
    return {
        "x_mm": x,
        "physical_y_mm": y,
        "masks": {
            "far": far,
            "corridor": corridor,
            "near": near,
            "peripheral_astig_left": left,
            "peripheral_astig_right": right,
            "monitored": monitored,
        },
        "rules": {"topology": {}},
        "statistics": {"corridor": {"physical_y_range_mm": [-9.0, -3.0]}},
    }


def _candidate(candidate_id: str, zone: str, fx: float, fy: float, x: float, y: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id, "field_x_deg": fx, "field_y_deg": fy,
        "reference_lens_x_mm": x, "reference_lens_physical_y_mm": y,
        "reference_partition_zone": zone, "trace_status": "ok", "eligible": True,
        "zone_boundary_clearance_mm": 2.0, "aperture_edge_clearance_mm": 2.0,
        "zone_boundary_safety_mm": 1.0,
    }


def _traced_candidates() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for zone, y0 in (("far", 20.0), ("near", -22.0)):
        for iy, y in enumerate((y0 - 6, y0 - 2, y0 + 2, y0 + 6)):
            for ix, x in enumerate((-12.0, -6.0, 0.0, 6.0, 12.0)):
                rows.append(_candidate(f"{zone}_{iy}_{ix}", zone, x, y, x, y))
    for iy, y in enumerate((-9.0, -7.0, -5.0, -3.0)):
        for x in (-4.0, 0.0, 4.0):
            rows.append(_candidate(f"mid_{iy}_{x}", "corridor", x, y, x, y))
    for band, y_values in (
        ("upper", (2.0, 4.25, 6.5, 8.75, 11.0)),
        ("middle", (-8.5, -7.25, -6.0, -4.75, -3.5)),
        ("lower", (-12.0, -14.4, -16.8, -19.2, -21.6, -24.0)),
    ):
        for index, y in enumerate(y_values):
            field_x = 20.0 + index
            rows.append(_candidate(f"pl_{band}_{index}", "astig_left", -field_x, y, -20.0 - index, y))
            rows.append(_candidate(f"pr_{band}_{index}", "astig_right", field_x, y, 20.0 + index, y))
    return rows


def _selected_cases(*, far_distance: float = 100000.0) -> list[dict[str, object]]:
    return select_training_cases(
        _traced_candidates(), far_object_distance_mm=far_distance,
        intermediate_object_distance_mm=2000.0, near_object_distance_mm=500.0,
        corridor_y_min_mm=-9.0, corridor_y_max_mm=-3.0,
    )


def _sampling_contract(*, far_distance: float = 100000.0) -> dict[str, object]:
    return {
        "method": "test dense-field FPS",
        "zone_boundary_safety_mm": {"default": 1.5, "corridor": 1.0},
        "aperture_edge_safety_mm": 1.5,
        "object_distance_mm": {
            "far": far_distance, "intermediate": 2000.0, "near": 500.0,
        },
        "peripheral_band_distance_mm": {
            "upper": far_distance, "middle": 2000.0, "lower": 500.0,
        },
    }


def test_dense_grid_is_deterministic_and_not_a_lens_plane_prescription() -> None:
    fields = generate_dense_candidate_fields(field_min_deg=-4.0, field_max_deg=4.0, field_step_deg=2.0)
    assert len(fields) == 25
    assert fields[0] == {"candidate_id": "cand_0001", "field_x_deg": -4.0, "field_y_deg": -4.0}
    assert "reference_lens_x_mm" not in fields[0]


def test_candidate_trace_rejects_invalid_sampling_margins_before_tracing() -> None:
    called = False

    def trace_reference(_fx: float, _fy: float) -> tuple[float, float]:
        nonlocal called
        called = True
        return 0.0, 10.0

    candidate = [{"candidate_id": "c1", "field_x_deg": 0.0, "field_y_deg": 0.0}]
    with pytest.raises(ValueError, match="finite and non-negative"):
        trace_candidate_fields(
            candidate,
            trace_reference=trace_reference,
            zones_payload=_zones_payload(),
            zone_boundary_safety_mm={"default": float("nan")},
            aperture_edge_safety_mm=1.5,
        )
    with pytest.raises(ValueError, match="missing zone-boundary"):
        trace_candidate_fields(
            candidate,
            trace_reference=trace_reference,
            zones_payload=_zones_payload(),
            zone_boundary_safety_mm={"corridor": 1.0},
            aperture_edge_safety_mm=1.5,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        trace_candidate_fields(
            candidate,
            trace_reference=trace_reference,
            zones_payload=_zones_payload(),
            zone_boundary_safety_mm={"default": 1.5},
            aperture_edge_safety_mm=-1.0,
        )
    assert called is False


def test_selects_fixed_80_cases_with_layered_corridor_and_mirrored_peripheral() -> None:
    cases = _selected_cases()
    counts = {group: sum(case["training_group"] == group for case in cases) for group in TRAINING_GROUP_COUNTS}
    assert counts == TRAINING_GROUP_COUNTS
    intermediate = [case for case in cases if case["training_group"] == "intermediate"]
    assert {case["corridor_vertical_layer"] for case in intermediate} == {1, 2, 3, 4}
    assert all(sum(case["corridor_vertical_layer"] == layer for case in intermediate) == 3 for layer in range(1, 5))
    left = {case["peripheral_pair_id"]: case for case in cases if case["training_group"] == "peripheral_left"}
    right = {case["peripheral_pair_id"]: case for case in cases if case["training_group"] == "peripheral_right"}
    assert left.keys() == right.keys()
    for pair_id in left:
        assert left[pair_id]["field_x_deg"] == -right[pair_id]["field_x_deg"]
        assert left[pair_id]["field_y_deg"] == right[pair_id]["field_y_deg"]
        assert left[pair_id]["distance_mm"] == right[pair_id]["distance_mm"]
    by_band = {band: {case["distance_mm"] for case in left.values() if case["peripheral_band"] == band} for band in ("upper", "middle", "lower")}
    assert by_band == {"upper": {100000.0}, "middle": {2000.0}, "lower": {500.0}}
    assert {
        band: sum(case["peripheral_band"] == band for case in left.values())
        for band in ("upper", "middle", "lower")
    } == PERIPHERAL_BAND_COUNTS


def test_select_training_cases_serializes_infinite_distance_without_overflow() -> None:
    cases = _selected_cases(far_distance=float("inf"))
    far_cases = [case for case in cases if case["training_group"] == "far"]
    assert far_cases
    assert all(case["case_id"].endswith("_Dinf") for case in far_cases)


def test_partition_classification_respects_physical_y_order() -> None:
    payload = _zones_payload()
    assert classify_partition_point(payload, x_mm=0.0, physical_y_mm=10.0) == "far"
    assert classify_partition_point(payload, x_mm=0.0, physical_y_mm=-6.0) == "corridor"
    assert classify_partition_point(payload, x_mm=0.0, physical_y_mm=-20.0) == "near"


def test_preoptimization_artifacts_record_candidates_and_five_groups(tmp_path) -> None:
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(json.dumps(_zones_payload()), encoding="utf-8")
    excel_path = tmp_path / "lens.xlsx"
    excel_path.write_bytes(b"test lens identity")
    cases = _selected_cases()
    for case in cases:
        case["case_lens_x_mm"] = case["reference_lens_x_mm"]
        case["case_lens_physical_y_mm"] = case["reference_lens_physical_y_mm"]
        case["case_position_partition_zone"] = case["reference_partition_zone"]
    output = write_preoptimization_artifacts(
        output_dir=tmp_path / "preoptimization", excel_path=excel_path,
        zones_json=zones_path,
        candidates=_traced_candidates(), cases=cases, reference_distance_mm=100000.0,
        sampling_contract=_sampling_contract(),
    )
    for name in (
        "candidate_fields.json", "case_manifest.json", "coverage_audit.json",
        "partition_map.png", "candidate_reachability_on_lens.png",
        "training_cases_on_lens.png",
        "training_cases_on_lens_assigned_distance.png",
        "training_cases_in_field.png",
    ):
        assert (output / name).is_file()
    manifest = json.loads((output / "case_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 5
    assert "posthoc_cases_json" not in manifest["source"]
    assert manifest["group_counts"] == TRAINING_GROUP_COUNTS
    assert manifest["objective_contract"]["J"] == "0.85 * J_functional + 0.15 * J_peripheral"
    assert manifest["case_geometry_audit"]["passed"] is True
    assert manifest["coverage_audit"]["overall_passed"] is True
    coverage = json.loads((output / "coverage_audit.json").read_text(encoding="utf-8"))
    assert coverage["schema_version"] == 3
    assert coverage["overall_passed"] is True
    for side in ("peripheral_astig_left", "peripheral_astig_right"):
        assert all(
            band["coverage_gate"]["passed"]
            for band in coverage["zones"][side]["peripheral_band_coverage"].values()
        )


def test_preoptimization_artifacts_accept_physical_infinity_for_far_distance(
    tmp_path,
) -> None:
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(json.dumps(_zones_payload()), encoding="utf-8")
    excel_path = tmp_path / "lens.xlsx"
    excel_path.write_bytes(b"test lens identity")
    cases = _selected_cases(far_distance=float("inf"))
    for case in cases:
        case["case_lens_x_mm"] = case["reference_lens_x_mm"]
        case["case_lens_physical_y_mm"] = case["reference_lens_physical_y_mm"]
        case["case_position_partition_zone"] = case["reference_partition_zone"]

    output = write_preoptimization_artifacts(
        output_dir=tmp_path / "preoptimization_infinity",
        excel_path=excel_path,
        zones_json=zones_path,
        candidates=_traced_candidates(),
        cases=cases,
        reference_distance_mm=float("inf"),
        sampling_contract=_sampling_contract(far_distance=float("inf")),
    )

    manifest = json.loads((output / "case_manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_geometry_audit"]["passed"] is True
    assert manifest["sampling_contract"]["object_distance_mm"]["far"] == float("inf")
    assert manifest["sampling_contract"]["peripheral_band_distance_mm"]["upper"] == float("inf")


def test_preoptimization_contract_rejects_task_distance_zone_crossing(tmp_path) -> None:
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(json.dumps(_zones_payload()), encoding="utf-8")
    excel_path = tmp_path / "lens.xlsx"
    excel_path.write_bytes(b"test lens identity")
    cases = _selected_cases()
    for case in cases:
        case["case_lens_x_mm"] = case["reference_lens_x_mm"]
        case["case_lens_physical_y_mm"] = case["reference_lens_physical_y_mm"]
        case["case_position_partition_zone"] = case["reference_partition_zone"]
    cases[0]["case_lens_x_mm"] = 20.0
    cases[0]["case_position_partition_zone"] = "astig_right"
    with pytest.raises(ValueError, match="case rear point"):
        write_preoptimization_artifacts(
            output_dir=tmp_path / "bad", excel_path=excel_path,
            zones_json=zones_path,
            candidates=_traced_candidates(), cases=cases,
            reference_distance_mm=100000.0,
            sampling_contract=_sampling_contract(),
        )


def test_preoptimization_contract_rejects_assigned_distance_fps_drift(tmp_path) -> None:
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(json.dumps(_zones_payload()), encoding="utf-8")
    excel_path = tmp_path / "lens.xlsx"
    excel_path.write_bytes(b"test lens identity")
    cases = _selected_cases()
    for case in cases:
        case["case_lens_x_mm"] = case["reference_lens_x_mm"]
        case["case_lens_physical_y_mm"] = case["reference_lens_physical_y_mm"]
        case["case_position_partition_zone"] = case["reference_partition_zone"]
    far_centre = next(
        case
        for case in cases
        if case["training_group"] == "far"
        and abs(float(case["reference_lens_x_mm"])) < 1.0e-9
    )
    far_centre["case_lens_x_mm"] = 0.1
    with pytest.raises(ValueError, match="reference-domain coverage cannot be transferred"):
        write_preoptimization_artifacts(
            output_dir=tmp_path / "fps_drift",
            excel_path=excel_path,
            zones_json=zones_path,
            candidates=_traced_candidates(),
            cases=cases,
            reference_distance_mm=100000.0,
            sampling_contract=_sampling_contract(),
        )


def test_preoptimization_contract_rejects_wrong_peripheral_distance(tmp_path) -> None:
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(json.dumps(_zones_payload()), encoding="utf-8")
    excel_path = tmp_path / "lens.xlsx"
    excel_path.write_bytes(b"test lens identity")
    cases = _selected_cases()
    for case in cases:
        case["case_lens_x_mm"] = case["reference_lens_x_mm"]
        case["case_lens_physical_y_mm"] = case["reference_lens_physical_y_mm"]
        case["case_position_partition_zone"] = case["reference_partition_zone"]
    pair_id = next(
        case["peripheral_pair_id"]
        for case in cases
        if case.get("peripheral_band") == "upper"
    )
    for case in cases:
        if case.get("peripheral_pair_id") == pair_id:
            case["distance_mm"] = 2000.0
    with pytest.raises(ValueError, match="expected 100000.0 mm"):
        write_preoptimization_artifacts(
            output_dir=tmp_path / "wrong_distance", excel_path=excel_path,
            zones_json=zones_path,
            candidates=_traced_candidates(), cases=cases,
            reference_distance_mm=100000.0,
            sampling_contract=_sampling_contract(),
        )


def test_preoptimization_contract_rejects_swapped_corridor_roles(tmp_path) -> None:
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(json.dumps(_zones_payload()), encoding="utf-8")
    excel_path = tmp_path / "lens.xlsx"
    excel_path.write_bytes(b"test lens identity")
    cases = _selected_cases()
    for case in cases:
        case["case_lens_x_mm"] = case["reference_lens_x_mm"]
        case["case_lens_physical_y_mm"] = case["reference_lens_physical_y_mm"]
        case["case_position_partition_zone"] = case["reference_partition_zone"]
    layer = [
        case for case in cases
        if case.get("training_group") == "intermediate"
        and case.get("corridor_vertical_layer") == 1
    ]
    left = next(case for case in layer if case["corridor_horizontal_role"] == "left_boundary")
    centre = next(case for case in layer if case["corridor_horizontal_role"] == "centre")
    left["corridor_horizontal_role"], centre["corridor_horizontal_role"] = (
        centre["corridor_horizontal_role"], left["corridor_horizontal_role"]
    )
    with pytest.raises(ValueError, match="roles do not match x order"):
        write_preoptimization_artifacts(
            output_dir=tmp_path / "swapped_roles", excel_path=excel_path,
            zones_json=zones_path,
            candidates=_traced_candidates(), cases=cases,
            reference_distance_mm=100000.0,
            sampling_contract=_sampling_contract(),
        )


def test_preoptimization_contract_rejects_tampered_candidate_membership(tmp_path) -> None:
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(json.dumps(_zones_payload()), encoding="utf-8")
    excel_path = tmp_path / "lens.xlsx"
    excel_path.write_bytes(b"test lens identity")
    cases = _selected_cases()
    for case in cases:
        case["case_lens_x_mm"] = case["reference_lens_x_mm"]
        case["case_lens_physical_y_mm"] = case["reference_lens_physical_y_mm"]
        case["case_position_partition_zone"] = case["reference_partition_zone"]
    cases[0]["candidate_id"] = "tampered_candidate"
    with pytest.raises(ValueError, match="unknown candidate"):
        write_preoptimization_artifacts(
            output_dir=tmp_path / "tampered_membership", excel_path=excel_path,
            zones_json=zones_path,
            candidates=_traced_candidates(), cases=cases,
            reference_distance_mm=100000.0,
            sampling_contract=_sampling_contract(),
        )


def test_plotter_uses_reference_and_assigned_rear_coordinates_separately(tmp_path) -> None:
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(json.dumps(_zones_payload()), encoding="utf-8")
    candidates_path = tmp_path / "candidate_fields.json"
    candidates_path.write_text(
        json.dumps({
            "candidates": [{
                "trace_status": "ok", "eligible": True,
                "reference_partition_zone": "far",
                "reference_lens_x_mm": 0.0,
                "reference_lens_physical_y_mm": 10.0,
            }],
        }),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "case_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "cases": [{
                "training_group": "far",
                "reference_lens_x_mm": 0.0,
                "reference_lens_physical_y_mm": 10.0,
                "case_lens_x_mm": 1.0,
                "case_lens_physical_y_mm": 11.0,
                "field_x_deg": 2.0,
                "field_y_deg": 3.0,
            }],
        }),
        encoding="utf-8",
    )

    calls_path = tmp_path / "scatter_calls.json"
    plotter_path = (
        Path(__file__).resolve().parents[1]
        / "biot" / "e2e" / "pal_case_layout_plotter.py"
    )
    script = """
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("pal_case_layout_plotter_probe", sys.argv[1])
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load PAL case-layout plotter")
plotter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)
from matplotlib.axes import Axes

calls = []
original_scatter = Axes.scatter

def record_scatter(self, x, y, *args, **kwargs):
    calls.append({
        "title": self.get_title(),
        "label": kwargs.get("label"),
        "x": [float(value) for value in x],
        "y": [float(value) for value in y],
    })
    return original_scatter(self, x, y, *args, **kwargs)

Axes.scatter = record_scatter
plotter._save = lambda fig, path: plotter.plt.close(fig)
plotter.render(
    zones_path=Path(sys.argv[2]),
    candidates_path=Path(sys.argv[3]),
    manifest_path=Path(sys.argv[4]),
    output=Path(sys.argv[5]),
)
Path(sys.argv[6]).write_text(json.dumps(calls), encoding="utf-8")
"""
    completed = subprocess.run(
        [
            sys.executable, "-c", script,
            str(plotter_path),
            str(zones_path), str(candidates_path), str(manifest_path),
            str(tmp_path / "plots"), str(calls_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"isolated plotter semantic probe failed: "
        f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
    )
    calls = json.loads(calls_path.read_text(encoding="utf-8"))

    reference = next(
        call for call in calls
        if call["title"] == "1 training cases over the reachable safe domain"
        and call["label"] == "Far (n=1)"
    )
    assigned = next(
        call for call in calls
        if call["title"] == "1 training cases at their assigned object distances"
        and call["label"] == "Far (n=1)"
    )
    assert (reference["x"], reference["y"]) == ([0.0], [10.0])
    assert (assigned["x"], assigned["y"]) == ([1.0], [11.0])
