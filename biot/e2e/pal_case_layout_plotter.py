"""Render Phase 16 case-layout plots in a clean process without importing torch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# These are the five disjoint physical partitions used by case selection.
# corridor_flank is a diagnostic submask wholly contained in corridor and is
# deliberately rendered with the corridor color.
PARTITION_ORDER = (
    "far", "corridor", "near", "peripheral_astig_left", "peripheral_astig_right",
)
ZONE_COLORS = {
    "far": "#4c78a8",
    "corridor": "#b279a2",
    "near": "#59a14f",
    "peripheral_astig_left": "#e15759",
    "peripheral_astig_right": "#f28e2b",
}
ZONE_LABELS = {
    "far": "Far",
    "corridor": "Intermediate corridor",
    "near": "Near",
    "peripheral_astig_left": "Peripheral-left",
    "peripheral_astig_right": "Peripheral-right",
}
PARTITION_PALETTE = ("#f5f5f5",) + tuple(
    ZONE_COLORS[zone] for zone in PARTITION_ORDER
)
ZONE_REFERENCE_NAMES = {
    "far": "far",
    "corridor": "corridor",
    "near": "near",
    "peripheral_astig_left": "astig_left",
    "peripheral_astig_right": "astig_right",
}
GROUP_STYLES = {
    "far": ("#174a7e", "o", "Far"),
    "far_robustness": ("#5b8db8", "o", "Far robustness"),
    "corridor_upper": ("#9467bd", "s", "Corridor upper"),
    "corridor_middle": ("#6f3b74", "s", "Corridor middle"),
    "corridor_lower": ("#b279a2", "s", "Corridor lower"),
    "near": ("#27632a", "^", "Near"),
    "near_robustness": ("#59a14f", "^", "Near robustness"),
    "near_edge_astig": ("#8cd17d", "D", "Near edge astig"),
    "peripheral_left": ("#a72d34", "<", "Peripheral-left"),
    "peripheral_right": ("#b96500", ">", "Peripheral-right"),
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _zone_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    x = np.asarray(payload["x_mm"], dtype=np.float64)
    y = np.asarray(payload["physical_y_mm"], dtype=np.float64)
    masks = {name: np.asarray(value, dtype=bool) for name, value in payload["masks"].items()}
    expected = (y.size, x.size)
    if any(mask.shape != expected for mask in masks.values()):
        raise ValueError("zone mask shape mismatch")
    return x, y, masks


def _plot_base(ax, payload: dict[str, Any], title: str) -> None:
    x, y, masks = _zone_arrays(payload)
    labels = np.zeros((y.size, x.size), dtype=np.int16)
    for number, zone in enumerate(PARTITION_ORDER, 1):
        labels[masks[zone]] = number
    # labels are 0 for background and 1..N in PARTITION_ORDER.  Keep this
    # generated from the same order as the legend; a missing entry silently
    # clips the last zone to the previous color in matplotlib.
    palette = PARTITION_PALETTE
    xx, yy = np.meshgrid(x, y)
    ax.pcolormesh(
        xx, yy, labels,
        cmap=colors.ListedColormap(palette), shading="nearest", alpha=0.78,
    )
    monitored = masks["monitored"]
    if bool(monitored.any()) and not bool(monitored.all()):
        ax.contour(
            xx, yy, monitored.astype(float), levels=[0.5],
            colors="#333333", linewidths=1,
        )
    pitch_x = float(np.median(abs(np.diff(x))))
    pitch_y = float(np.median(abs(np.diff(y))))
    ax.set_xlim(float(x.min()) - 0.5 * pitch_x, float(x.max()) + 0.5 * pitch_x)
    ax.set_ylim(float(y.min()) - 0.5 * pitch_y, float(y.max()) + 0.5 * pitch_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set(
        xlabel="PAL rear-surface x (mm)",
        ylabel="PAL rear-surface physical y (mm)",
        title=title,
    )
    ax.grid(alpha=0.15, linewidth=0.5)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render(
    *, zones_path: Path, candidates_path: Path, manifest_path: Path,
    dense_candidates_path: Path | None = None, output: Path,
) -> None:
    zones = _read(zones_path)
    candidate_payload = _read(candidates_path)
    manifest = _read(manifest_path)
    candidates = list(candidate_payload["candidates"])
    dense_candidates = candidates
    if dense_candidates_path is not None:
        dense_payload = _read(dense_candidates_path)
        dense_candidates = list(dense_payload["candidates"])
    cases = list(manifest["cases"])
    total = len(cases)
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 7.3))
    _plot_base(ax, zones, "PAL five-zone partition before optimization")
    ax.legend(
        handles=[
            Line2D([], [], color=ZONE_COLORS[zone], marker="s", linestyle="None", markersize=8, label=ZONE_LABELS[zone])
            for zone in PARTITION_ORDER
        ],
        loc="upper right", fontsize=8, frameon=False,
    )
    _save(fig, output / "partition_map.png")

    traceable = [
        row for row in candidates
        if row.get("trace_status") == "ok" and row.get("reference_partition_zone") is not None
    ]
    eligible = [row for row in traceable if bool(row.get("eligible"))]
    fig, ax = plt.subplots(figsize=(8.4, 7.3))
    _plot_base(ax, zones, "Full masks and Original-PAL reachable training domain")
    ax.scatter(
        [row["reference_lens_x_mm"] for row in traceable],
        [row["reference_lens_physical_y_mm"] for row in traceable],
        s=16, c="#5f6368", alpha=0.45, linewidths=0.6, marker="x",
        label=f"True-traceable (n={len(traceable)})", zorder=3,
    )
    ax.scatter(
        [row["reference_lens_x_mm"] for row in eligible],
        [row["reference_lens_physical_y_mm"] for row in eligible],
        s=26, facecolors="none", edgecolors="#111111", alpha=0.85,
        linewidths=0.8, marker="o",
        label=f"Eligible (traceable + classified) (n={len(eligible)})", zorder=4,
    )
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, output / "candidate_reachability_on_lens.png")

    fig, ax = plt.subplots(figsize=(8.4, 7.3))
    dense_ok = sum(row.get("trace_status") == "ok" for row in dense_candidates)
    dense_failed = len(dense_candidates) - dense_ok
    _plot_base(
        ax,
        zones,
        "Dense object-field grid mapped to PAL rear-surface coordinates "
        f"(ok={dense_ok}, failed={dense_failed}; failed points omitted)",
    )
    dense_by_zone = {
        zone: [
            row for row in dense_candidates
            if row.get("trace_status") == "ok"
            and row.get("reference_partition_zone") == ZONE_REFERENCE_NAMES[zone]
        ]
        for zone in PARTITION_ORDER
    }
    for zone in PARTITION_ORDER:
        rows = dense_by_zone[zone]
        if not rows:
            continue
        ax.scatter(
            [row["reference_lens_x_mm"] for row in rows],
            [row["reference_lens_physical_y_mm"] for row in rows],
            s=5, c=ZONE_COLORS[zone], alpha=0.55, linewidths=0,
            label=f"{ZONE_LABELS[zone]} (n={len(rows)})", zorder=4,
        )
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, output / "dense_candidate_grid_on_lens.png")

    fig, ax = plt.subplots(figsize=(8.4, 7.3))
    _plot_base(ax, zones, f"{total} training cases over the reachable eligible domain")
    ax.scatter(
        [row["reference_lens_x_mm"] for row in eligible],
        [row["reference_lens_physical_y_mm"] for row in eligible],
        s=3, c="#242424", alpha=0.16, linewidths=0, zorder=3,
    )
    for group, (color, marker, label) in GROUP_STYLES.items():
        rows = [case for case in cases if case["training_group"] == group]
        ax.scatter(
            [row["reference_lens_x_mm"] for row in rows],
            [row["reference_lens_physical_y_mm"] for row in rows],
            c=color, marker=marker, s=42, edgecolors="white", linewidths=0.7,
            label=f"{label} (n={len(rows)})", zorder=5,
        )
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, output / "training_cases_on_lens.png")

    fig, ax = plt.subplots(figsize=(8.4, 7.3))
    _plot_base(
        ax,
        zones,
        f"{total} training cases at their assigned object distances",
    )
    for group, (color, marker, label) in GROUP_STYLES.items():
        rows = [case for case in cases if case["training_group"] == group]
        ax.scatter(
            [row["case_lens_x_mm"] for row in rows],
            [row["case_lens_physical_y_mm"] for row in rows],
            c=color, marker=marker, s=42, edgecolors="white", linewidths=0.7,
            label=f"{label} (n={len(rows)})", zorder=5,
        )
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, output / "training_cases_on_lens_assigned_distance.png")

    fig, ax = plt.subplots(figsize=(7.7, 7.1))
    for group, (color, marker, label) in GROUP_STYLES.items():
        rows = [case for case in cases if case["training_group"] == group]
        ax.scatter(
            [row["field_x_deg"] for row in rows],
            [row["field_y_deg"] for row in rows],
            c=color, marker=marker, s=44, edgecolors="white", linewidths=0.7,
            label=f"{label} (n={len(rows)})",
        )
    ax.set(
        aspect="equal", xlabel="Field x (deg)", ylabel="Field y (deg)",
        title=f"{total} traceable training cases in object-field coordinates",
    )
    ax.axhline(0, color="#777", linewidth=0.6)
    ax.axvline(0, color="#777", linewidth=0.6)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, output / "training_cases_in_field.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dense-candidates", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(
        zones_path=args.zones,
        candidates_path=args.candidates,
        manifest_path=args.manifest,
        dense_candidates_path=args.dense_candidates,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
