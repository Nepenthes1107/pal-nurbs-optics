from pathlib import Path

import numpy as np

from averfang import compute_averfang_maps, load_sag_xlsx, physical_display_map


def test_averfang_map_current_pal_uses_full_80mm_grid():
    repo_root = Path(__file__).resolve().parents[1]
    sag = load_sag_xlsx(repo_root / "USSTbc1.xlsx", grid_shape=(81, 81))

    result = compute_averfang_maps(
        sag,
        semi_dia_mm=40.0,
        refractive_index=1.76,
        front_radius_mm=253.3,
        center_thickness_mm=2.3,
        crib_diameter_mm=80.0,
    )

    x = result["x_mm"]
    y = result["y_mm"]
    power = result["power_D"]
    astigmatism = result["astigmatism_D"]

    assert power.shape == (81, 81)
    assert astigmatism.shape == (81, 81)
    assert x[0] == -40.0
    assert x[-1] == 40.0
    assert y[0] == -40.0
    assert y[-1] == 40.0
    assert np.isfinite(power[40, 40])
    assert abs(power[40, 40] + 2.38) < 0.02
    assert np.nanmean(power[39:42, 39:42]) > np.nanmean(power[25:30, 39:42])
    assert np.nanmin(astigmatism) >= 0.0
    assert result["metadata"]["grid_pitch_mm"] == 1.0


def test_physical_display_places_raw_first_row_at_positive_y():
    x = np.array([-1.0, 0.0, 1.0])
    physical_y_by_raw_row = np.array([1.0, 0.0, -1.0])
    raw_rows = np.array(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ]
    )

    x_display, y_display, display = physical_display_map(
        x,
        physical_y_by_raw_row,
        raw_rows,
    )

    np.testing.assert_array_equal(x_display, x)
    np.testing.assert_array_equal(y_display, [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(display[0], raw_rows[-1])
    np.testing.assert_array_equal(display[-1], raw_rows[0])
