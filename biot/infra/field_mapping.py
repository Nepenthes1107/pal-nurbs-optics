from __future__ import annotations


def field_angles_to_cb_excel_tilts(field_x_deg: float, field_y_deg: float) -> tuple[float, float]:
    """Map requested field angles to Excel CoordinateBreak tilt cells.

    Inputs:
    - field_x_deg, field_y_deg: requested visual field angles in degree.

    Outputs:
    - tuple `(h7_tilt_x_deg, i7_tilt_y_deg)` written to Excel cells H7/I7.

    Coordinate meaning:
    - Excel H7 is the CoordinateBreak rotation about the local X axis, which
      produces a Y-field eye rotation.
    - Excel I7 is the CoordinateBreak rotation about the local Y axis, which
      produces an X-field eye rotation.

    This helper is CPU-only, creates no tensors, and has no autograd behavior.
    """

    return float(field_y_deg), float(field_x_deg)
