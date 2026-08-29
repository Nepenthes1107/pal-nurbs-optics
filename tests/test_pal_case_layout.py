from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from biot.e2e.pal_case_layout import (
    DISTANCE_LABELS,
    PARTITION_ZONES,
    PartitionMap,
    attach_objective_weights,
    build_multidistance_layout,
    generate_fov_grid,
    load_weight_spec,
)


def _weight_spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "zone_total_weight": {
            "far": 0.25,
            "corridor": 0.25,
            "near": 0.25,
            "astig_left": 0.125,
            "astig_right": 0.125,
        },
        "distance_fraction_by_zone": {
            zone: {label: 1.0 / 3.0 for label in DISTANCE_LABELS}
            for zone in PARTITION_ZONES
        },
    }


def _synthetic_partition() -> PartitionMap:
    # physical_y is deliberately descending, matching zones.json's convention.
    x = np.arange(-2.0, 3.0)
    y = np.arange(2.0, -3.0, -1.0)
    masks = {
        "far": np.zeros((5, 5), dtype=bool),
        "corridor": np.zeros((5, 5), dtype=bool),
        "near": np.zeros((5, 5), dtype=bool),
        "peripheral_astig_left": np.zeros((5, 5), dtype=bool),
        "peripheral_astig_right": np.zeros((5, 5), dtype=bool),
        "monitored": np.ones((5, 5), dtype=bool),
    }
    masks["far"][0, 1:4] = True
    masks["corridor"][2, 1:4] = True
    masks["near"][4, 1:4] = True
    masks["peripheral_astig_left"][1:4, 0] = True
    masks["peripheral_astig_right"][1:4, 4] = True
    return PartitionMap(x, y, masks)


def test_fov_grid_is_exactly_11_by_11_and_deterministic() -> None:
    grid = generate_fov_grid(field_min_deg=-10.0, field_max_deg=10.0, count=11)
    assert len(grid) == 121
    assert grid[0] == {
        "grid_row": 0,
        "grid_column": 0,
        "field_x_deg": -10.0,
        "field_y_deg": -10.0,
    }
    assert grid[-1]["field_x_deg"] == 10.0
    assert grid[-1]["field_y_deg"] == 10.0
    assert grid[60]["field_x_deg"] == 0.0
    assert grid[60]["field_y_deg"] == 0.0
    with pytest.raises(ValueError, match="11x11"):
        generate_fov_grid(field_min_deg=-1.0, field_max_deg=1.0, count=9)


def test_multidistance_training_fov_contract_is_minus55_to_plus55_step11() -> None:
    grid = generate_fov_grid(field_min_deg=-55.0, field_max_deg=55.0, count=11)
    expected = [-55.0, -44.0, -33.0, -22.0, -11.0, 0.0, 11.0, 22.0, 33.0, 44.0, 55.0]
    assert [row["field_x_deg"] for row in grid[:11]] == expected
    assert [row["field_y_deg"] for row in grid[::11]] == expected
    assert len(grid) == 121
    assert [(row["grid_row"], row["grid_column"]) for row in grid[:3]] == [(0, 0), (0, 1), (0, 2)]


def test_partition_map_uses_stored_masks_and_explicit_nearest_extension() -> None:
    partition = _synthetic_partition()
    stored = partition.classify(x_mm=0.0, physical_y_mm=2.0)
    assert stored.zone == "far"
    assert stored.mode == "stored_mask"

    # (0, 1.4) maps to a monitored but intentionally unlabelled safety-band cell;
    # it is assigned to the nearest partition and the mode is recorded.
    nearest = partition.classify(x_mm=0.0, physical_y_mm=1.4)
    assert nearest.zone == "far"
    assert nearest.mode == "nearest_partition_cell"
    assert nearest.nearest_partition_distance_mm == pytest.approx(0.6)

    outside = PartitionMap(
        partition.x_mm,
        partition.physical_y_mm,
        {**partition.masks, "monitored": np.zeros((5, 5), dtype=bool)},
    )
    with pytest.raises(ValueError, match="outside monitored"):
        outside.classify(x_mm=0.0, physical_y_mm=0.0)


def test_attach_objective_weights_preserves_zone_masses() -> None:
    cases = [
        {"case_id": f"{zone}_{label}_{i}", "zone": zone, "distance_label": label}
        for zone in PARTITION_ZONES
        for label in DISTANCE_LABELS
        for i in range(2)
    ]
    weighted = attach_objective_weights(cases, _weight_spec())
    assert len(weighted) == len(cases)
    assert math.isclose(sum(row["objective_weight"] for row in weighted), 1.0)
    for zone, expected in {
        "far": 0.25,
        "corridor": 0.25,
        "near": 0.25,
        "astig_left": 0.125,
        "astig_right": 0.125,
    }.items():
        actual = sum(row["objective_weight"] for row in weighted if row["zone"] == zone)
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)


def test_attach_objective_weights_fails_if_a_zone_distance_cell_is_missing() -> None:
    cases = [
        {"case_id": "near_D500", "zone": "near", "distance_label": "D500"},
    ]
    with pytest.raises(ValueError, match="positive-count zone/distance"):
        attach_objective_weights(cases, _weight_spec())


def test_build_multidistance_layout_has_three_complete_shared_grids() -> None:
    partition = _synthetic_partition()

    def trace_reference(distance: float, field_x: float, field_y: float) -> tuple[float, float]:
        # The synthetic map deliberately makes the three distance mappings
        # identical; the builder still calls the callback once per case.
        assert distance in (500.0, 1000.0) or math.isinf(distance)
        return field_x / 5.0, -field_y / 5.0

    cases = build_multidistance_layout(
        field_min_deg=-10.0,
        field_max_deg=10.0,
        partition_map=partition,
        weight_spec=_weight_spec(),
        trace_reference=trace_reference,
    )
    assert len(cases) == 363
    assert [cases[i]["distance_label"] for i in (0, 121, 242)] == list(DISTANCE_LABELS)
    assert [
        (row["field_x_deg"], row["field_y_deg"])
        for row in cases[:121]
    ] == [
        (row["field_x_deg"], row["field_y_deg"])
        for row in cases[121:242]
    ]
    assert [
        (row["field_x_deg"], row["field_y_deg"])
        for row in cases[:121]
    ] == [
        (row["field_x_deg"], row["field_y_deg"])
        for row in cases[242:]
    ]
    assert all(row["zone"] in PARTITION_ZONES for row in cases)
    assert math.isclose(sum(row["objective_weight"] for row in cases), 1.0)


def test_repository_weight_configuration_is_machine_readable() -> None:
    path = Path("inputs/pal/multidistance_weights.json")
    payload = load_weight_spec(path)
    assert payload["schema_version"] == 1
    assert set(payload["zone_total_weight"]) == set(PARTITION_ZONES)
    assert set(payload["distance_fraction_by_zone"]["far"]) == set(DISTANCE_LABELS)


def test_repository_weight_configuration_matches_fixed_objective_masses() -> None:
    payload = load_weight_spec(Path("inputs/pal/multidistance_weights.json"))
    totals = payload["zone_total_weight"]
    fractions = payload["distance_fraction_by_zone"]
    expected = {
        "far": {"D500": 0.0, "D1000": 0.05, "Dinf": 0.2},
        "corridor": {"D500": 0.025, "D1000": 0.2, "Dinf": 0.025},
        "near": {"D500": 0.2, "D1000": 0.05, "Dinf": 0.0},
    }
    for zone, cells in expected.items():
        for distance, mass in cells.items():
            assert fractions[zone][distance] * totals[zone] == pytest.approx(mass, abs=1.0e-12)
    assert fractions["far"]["D500"] == 0.0
    assert fractions["near"]["Dinf"] == 0.0
