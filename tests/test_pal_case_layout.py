from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

from biot.e2e.pal_case_layout_plotter import (
    PARTITION_PALETTE,
    PARTITION_ORDER as PLOT_PARTITION_ORDER,
    ZONE_COLORS as PLOT_ZONE_COLORS,
)

from biot.e2e.pal_case_layout import (
    PERIPHERAL_BAND_COUNTS,
    TRAINING_GROUP_COUNTS,
    _unique_physical_candidate_rows,
    _validate_selected_candidate_membership,
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
    empty = [[0 for _ in x] for _ in y]
    lower_edge_guard = [[int(-23 <= yy <= -18) for _ in x] for yy in y]
    power = []
    for yy in y:
        local_add = 0.35 if yy >= -3 else 0.8 if yy >= -9 else 1.6
        power.append([1.0 + local_add if -14 <= xx <= 14 and -12 <= yy <= 0 else 1.0 for xx in x])
    return {
        "x_mm": x,
        "physical_y_mm": y,
        "masks": {
            "far": far,
            "corridor": corridor,
            "corridor_flank": empty,
            "near": near,
            "peripheral_astig_left": left,
            "peripheral_astig_right": right,
            "monitored": monitored,
            "lower_edge_guard": lower_edge_guard,
        },
        "maps": {"power_D": power, "astigmatism_D": [[0.0 for _ in x] for _ in y]},
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
    for iy, y in enumerate((14.0, 18.0, 22.0, 26.0)):
        for ix, x in enumerate((-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0)):
            rows.append(_candidate(f"far_{iy}_{ix}", "far", x, y, x, y))
    near_rows = (
        ((-30.0, -14.0), (-35.0, -18.0), (-39.0, -21.0)),
        ((-43.0, -23.0), (-47.0, -25.0), (-51.0, -27.0),
         (-55.0, -29.0), (-57.0, -30.0)),
    )
    for stratum, definitions in zip(("core", "deep"), near_rows):
        for iy, (field_y, lens_y) in enumerate(definitions):
            for ix, x in enumerate((-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0)):
                rows.append(
                    _candidate(
                        f"near_{stratum}_{iy}_{ix}", "near", x, field_y, x, lens_y
                    )
                )
    for iy, y in enumerate((-12.0, -11.0, -10.0, -9.0, -7.0, -5.0, -3.0, -2.0, 0.0)):
        for x in (-8.0, -4.0, 0.0, 4.0, 8.0):
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
    zones = _zones_payload()
    return select_training_cases(
        _traced_candidates(), far_object_distance_mm=far_distance,
        intermediate_object_distance_mm=2000.0, near_object_distance_mm=500.0,
        corridor_y_min_mm=-9.0, corridor_y_max_mm=-3.0,
        power_map=np.asarray(zones["maps"]["power_D"], dtype=np.float64),
        pfar=1.0,
        zones_payload=zones,
    )


def _sampling_contract(*, far_distance: float = 100000.0) -> dict[str, object]:
    return {
        "method": "test dense-field FPS",
        "candidate_eligibility": (
            "trace_status=ok and reference_partition_zone is classified"
        ),
        "object_distance_mm": {
            "far": far_distance, "intermediate": 2000.0, "near": 500.0,
        },
        "peripheral_band_distance_mm": {
            "upper": far_distance, "middle": far_distance, "lower": far_distance,
        },
    }


def test_dense_grid_is_deterministic_and_not_a_lens_plane_prescription() -> None:
    fields = generate_dense_candidate_fields(field_min_deg=-4.0, field_max_deg=4.0, field_step_deg=2.0)
    assert len(fields) == 25
    assert fields[0] == {"candidate_id": "cand_00001", "field_x_deg": -4.0, "field_y_deg": -4.0}
    assert "reference_lens_x_mm" not in fields[0]


def test_candidate_trace_ignores_clearance_filter_arguments() -> None:
    calls = 0

    def trace_reference(_fx: float, _fy: float) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return 0.0, 10.0

    candidate = [{"candidate_id": "c1", "field_x_deg": 0.0, "field_y_deg": 0.0}]
    for zone_margin, aperture_margin in (
        ({"default": float("nan")}, -1.0),
        ({"corridor": 1.0}, float("nan")),
        ({"default": -100.0}, 1.5),
    ):
        traced = trace_candidate_fields(
            candidate,
            trace_reference=trace_reference,
            zones_payload=_zones_payload(),
            zone_boundary_safety_mm=zone_margin,
            aperture_edge_safety_mm=aperture_margin,
        )
        assert traced[0]["trace_status"] == "ok"
        assert traced[0]["reference_partition_zone"] == "far"
        assert traced[0]["trace_eligible"] is True
        assert traced[0]["eligible"] is True
    assert calls == 3


def test_candidate_trace_progress_is_portable_across_checkout_paths(tmp_path) -> None:
    candidates = [
        {"candidate_id": "c1", "field_x_deg": 0.0, "field_y_deg": 0.0}
    ]
    progress = tmp_path / "candidate_trace_progress.json"
    common = {
        "candidates": candidates,
        "trace_reference": lambda _fx, _fy: (0.0, 10.0),
        "zones_payload": _zones_payload(),
        "zone_boundary_safety_mm": {"default": 1.5, "corridor": 1.0},
        "aperture_edge_safety_mm": 1.5,
        "progress_path": progress,
    }
    first = trace_candidate_fields(
        **common,
        trace_identity={
            "excel_path": "/root/cloud/repo/lens.xlsx",
            "excel_sha256": "a" * 64,
        },
    )
    second = trace_candidate_fields(
        **common,
        trace_identity={
            "excel_path": r"D:\local\repo\lens.xlsx",
            "excel_sha256": "a" * 64,
        },
    )

    assert second == first


def test_candidate_trace_progress_path_is_identity_without_content_hash(tmp_path) -> None:
    candidates = [
        {"candidate_id": "c1", "field_x_deg": 0.0, "field_y_deg": 0.0}
    ]
    progress = tmp_path / "candidate_trace_progress.json"
    common = {
        "candidates": candidates,
        "trace_reference": lambda _fx, _fy: (0.0, 10.0),
        "zones_payload": _zones_payload(),
        "zone_boundary_safety_mm": {"default": 1.5, "corridor": 1.0},
        "aperture_edge_safety_mm": 1.5,
        "progress_path": progress,
    }
    trace_candidate_fields(
        **common,
        trace_identity={"excel_path": "/root/cloud/repo/lens.xlsx"},
    )

    with pytest.raises(ValueError, match="candidate progress identity mismatch"):
        trace_candidate_fields(
            **common,
            trace_identity={"excel_path": r"D:\local\repo\lens.xlsx"},
        )


def test_selects_fixed_121_cases_with_stratified_corridor_near_and_mirrored_peripheral() -> None:
    cases = _selected_cases()
    counts = {group: sum(case["training_group"] == group for case in cases) for group in TRAINING_GROUP_COUNTS}
    assert counts == TRAINING_GROUP_COUNTS
    assert len(cases) == 121
    assert {
        group: {
            stratum: sum(
                case["training_group"] == group
                and case.get("near_spatial_stratum") == stratum
                for case in cases
            )
            for stratum in ("core", "deep")
        }
        for group in ("near", "near_robustness")
    } == {
        "near": {"core": 8, "deep": 20},
        "near_robustness": {"core": 4, "deep": 8},
    }
    for group in TRAINING_GROUP_COUNTS:
        weights = [
            case["spatial_weight"]
            for case in cases
            if case["training_group"] == group
        ]
        assert all(value > 0.0 for value in weights)
        assert sum(weights) == pytest.approx(1.0, abs=1.0e-12)
    expected_add_ranges = {
        "corridor_upper": (0.2, 0.5),
        "corridor_middle": (0.5, 1.3),
        "corridor_lower": (1.3, 2.0),
    }
    for group, (lower, upper) in expected_add_ranges.items():
        values = [float(case["corridor_local_add_D"]) for case in cases if case["training_group"] == group]
        assert len(values) == 5
        assert all(lower <= value <= upper for value in values)
    left = {case["peripheral_pair_id"]: case for case in cases if case["training_group"] == "peripheral_left"}
    right = {case["peripheral_pair_id"]: case for case in cases if case["training_group"] == "peripheral_right"}
    assert left.keys() == right.keys()
    for pair_id in left:
        assert left[pair_id]["field_x_deg"] == -right[pair_id]["field_x_deg"]
        assert left[pair_id]["field_y_deg"] == right[pair_id]["field_y_deg"]
        assert left[pair_id]["distance_mm"] == right[pair_id]["distance_mm"]
    by_band = {band: {case["distance_mm"] for case in left.values() if case["peripheral_band"] == band} for band in ("upper", "middle", "lower")}
    assert by_band == {"upper": {100000.0}, "middle": {100000.0}, "lower": {100000.0}}
    assert {
        band: sum(case["peripheral_band"] == band for case in left.values())
        for band in ("upper", "middle", "lower")
    } == PERIPHERAL_BAND_COUNTS


def test_select_training_cases_serializes_infinite_distance_without_overflow() -> None:
    cases = _selected_cases(far_distance=float("inf"))
    far_cases = [case for case in cases if case["training_group"] == "far"]
    assert far_cases
    assert all(case["case_id"].endswith("_Dinf") for case in far_cases)
    zones = _zones_payload()
    reselected = select_training_cases(
        cases,
        far_object_distance_mm=float("inf"),
        intermediate_object_distance_mm=2000.0,
        near_object_distance_mm=500.0,
        corridor_y_min_mm=-9.0,
        corridor_y_max_mm=-3.0,
        power_map=np.asarray(zones["maps"]["power_D"], dtype=np.float64),
        pfar=1.0,
        zones_payload=zones,
    )
    assert sum(case["training_group"] == "far" for case in reselected) == 28
    assert all(
        math.isinf(float(case["distance_mm"]))
        for case in reselected
        if case["training_group"] == "far"
    )


def test_final_fps_retains_each_qualified_pool_training_group() -> None:
    zones = _zones_payload()
    qualified_pool = _selected_cases(far_distance=float("inf"))
    rogue = next(
        dict(case)
        for case in qualified_pool
        if case["training_group"] == "near_robustness"
    )
    rogue.update(
        {
            "candidate_id": "robustness_only_extreme",
            "case_id": "near_robustness_pool_extreme_D2000",
            "field_x_deg": 99.0,
            "reference_lens_x_mm": 13.5,
            "case_lens_x_mm": 13.5,
            "distance_mm": 2000.0,
        }
    )
    qualified_pool.append(rogue)

    final_cases = select_training_cases(
        qualified_pool,
        far_object_distance_mm=float("inf"),
        intermediate_object_distance_mm=2000.0,
        near_object_distance_mm=500.0,
        corridor_y_min_mm=-9.0,
        corridor_y_max_mm=-3.0,
        power_map=np.asarray(zones["maps"]["power_D"], dtype=np.float64),
        pfar=1.0,
        zones_payload=zones,
    )

    qualified_keys = {
        (
            str(case["training_group"]),
            str(case["candidate_id"]),
            float(case["distance_mm"]),
            float(case["field_x_deg"]),
            float(case["field_y_deg"]),
        )
        for case in qualified_pool
    }
    assert all(
        (
            str(case["training_group"]),
            str(case["candidate_id"]),
            float(case["distance_mm"]),
            float(case["field_x_deg"]),
            float(case["field_y_deg"]),
        )
        in qualified_keys
        for case in final_cases
    )
    far_cases = [case for case in final_cases if case["training_group"] == "far"]
    assert far_cases
    assert all(math.isinf(float(case["distance_mm"])) for case in far_cases)
    assert all(case["candidate_id"] != "robustness_only_extreme" for case in far_cases)


def test_partition_classification_respects_physical_y_order() -> None:
    payload = _zones_payload()
    assert classify_partition_point(payload, x_mm=0.0, physical_y_mm=10.0) == "far"
    assert classify_partition_point(payload, x_mm=0.0, physical_y_mm=-6.0) == "corridor"
    assert classify_partition_point(payload, x_mm=0.0, physical_y_mm=-20.0) == "near"


def test_preoptimization_artifacts_record_candidates_and_nine_groups(tmp_path) -> None:
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
        "dense_candidate_fields.json", "dense_candidate_grid_on_lens.png",
        "partition_map.png", "candidate_reachability_on_lens.png",
        "training_cases_on_lens.png",
        "training_cases_on_lens_assigned_distance.png",
        "training_cases_in_field.png",
    ):
        assert (output / name).is_file()
    manifest = json.loads((output / "case_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 10
    assert "posthoc_cases_json" not in manifest["source"]
    assert manifest["group_counts"] == TRAINING_GROUP_COUNTS
    assert "group_weight" in manifest["objective_contract"]["J"]
    assert manifest["case_geometry_audit"]["passed"] is True
    assert manifest["coverage_audit"]["overall_passed"] is True
    coverage = json.loads((output / "coverage_audit.json").read_text(encoding="utf-8"))
    assert coverage["schema_version"] == 6
    assert coverage["overall_passed"] is True
    for side in ("peripheral_astig_left", "peripheral_astig_right"):
        assert all(
            band["coverage_gate"]["passed"]
            for band in coverage["zones"][side]["peripheral_band_coverage"].values()
        )


def test_partition_plot_palette_has_one_color_per_partition() -> None:
    assert len(PARTITION_PALETTE) == len(PLOT_PARTITION_ORDER) + 1
    assert PARTITION_PALETTE[1:] == tuple(
        PLOT_ZONE_COLORS[zone] for zone in PLOT_PARTITION_ORDER
    )


def test_preoptimization_artifacts_accept_group_qualified_pool_candidate_reuse(
    tmp_path,
) -> None:
    zones = _zones_payload()
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(json.dumps(zones), encoding="utf-8")
    excel_path = tmp_path / "lens.xlsx"
    excel_path.write_bytes(b"test lens identity")
    qualified_pool = _selected_cases()
    assert len({case["candidate_id"] for case in qualified_pool}) < len(
        qualified_pool
    )
    cases = select_training_cases(
        qualified_pool,
        far_object_distance_mm=100000.0,
        intermediate_object_distance_mm=2000.0,
        near_object_distance_mm=500.0,
        corridor_y_min_mm=-9.0,
        corridor_y_max_mm=-3.0,
        power_map=np.asarray(zones["maps"]["power_D"], dtype=np.float64),
        pfar=1.0,
        zones_payload=zones,
    )
    for case in cases:
        case["case_lens_x_mm"] = case["reference_lens_x_mm"]
        case["case_lens_physical_y_mm"] = case[
            "reference_lens_physical_y_mm"
        ]
        case["case_position_partition_zone"] = case[
            "reference_partition_zone"
        ]

    output = write_preoptimization_artifacts(
        output_dir=tmp_path / "qualified_pool_preoptimization",
        excel_path=excel_path,
        zones_json=zones_path,
        candidates=qualified_pool,
        cases=cases,
        reference_distance_mm=100000.0,
        sampling_contract=_sampling_contract(),
    )

    manifest = json.loads(
        (output / "case_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 10
    membership = manifest["candidate_membership_audit"]
    assert membership["candidate_domain"] == "group_qualified_pool"
    assert membership["candidate_record_count"] == len(qualified_pool)
    assert membership["unique_source_candidate_count"] < len(qualified_pool)
    coverage = json.loads(
        (output / "coverage_audit.json").read_text(encoding="utf-8")
    )
    assert coverage["schema_version"] == 6
    assert coverage["candidate_reuse_count"] > 0
    assert coverage["overall_passed"] is True
    far_cases = [
        case for case in cases
        if case["training_group"] == "far"
    ]
    assert far_cases
    selection_audit = far_cases[0]["coverage_selection_audit"]
    assert selection_audit["method"] == "group_fps_then_deterministic_coverage_swap"
    assert selection_audit["coverage_gate"]["passed"] is True


def test_group_qualified_pool_rejects_conflicting_physical_candidate_identity() -> None:
    source = _selected_cases()[0]
    conflict = dict(source)
    conflict["field_x_deg"] = float(source["field_x_deg"]) + 1.0

    with pytest.raises(ValueError, match="conflicting physical source records"):
        _validate_selected_candidate_membership([source, conflict], [])


def test_physical_candidate_coverage_aggregates_group_specific_eligibility() -> None:
    candidates = _selected_cases()
    by_id: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        by_id.setdefault(str(candidate["candidate_id"]), []).append(candidate)
    reused = next(rows for rows in by_id.values() if len(rows) > 1)
    records = [dict(row) for row in reused[:2]]
    records[0]["eligible"] = False
    records[1]["eligible"] = True

    physical = _unique_physical_candidate_rows(records)

    assert len(physical) == 1
    assert physical[0]["eligible"] is True


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


def test_preoptimization_contract_rejects_corridor_add_outside_stratum(tmp_path) -> None:
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(json.dumps(_zones_payload()), encoding="utf-8")
    excel_path = tmp_path / "lens.xlsx"
    excel_path.write_bytes(b"test lens identity")
    cases = _selected_cases()
    for case in cases:
        case["case_lens_x_mm"] = case["reference_lens_x_mm"]
        case["case_lens_physical_y_mm"] = case["reference_lens_physical_y_mm"]
        case["case_position_partition_zone"] = case["reference_partition_zone"]
    target = next(case for case in cases if case["training_group"] == "corridor_upper")
    target["corridor_local_add_D"] = 0.8
    target["distance_mm"] = 1250.0
    with pytest.raises(ValueError, match="outside corridor_upper band"):
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
        if call["title"] == "1 training cases over the reachable eligible domain"
        and call["label"] == "Far (n=1)"
    )
    assigned = next(
        call for call in calls
        if call["title"] == "1 training cases at their assigned object distances"
        and call["label"] == "Far (n=1)"
    )
    assert (reference["x"], reference["y"]) == ([0.0], [10.0])
    assert (assigned["x"], assigned["y"]) == ([1.0], [11.0])
