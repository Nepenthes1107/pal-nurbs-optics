from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(7, 5), tight_layout=True)
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_axis_off()

    def show_psf(self, psf: np.ndarray) -> None:
        """Render a display-normalized PSF without changing computational data."""

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.axes.set_axis_off()
        image = np.asarray(psf, dtype=float)
        self.axes.imshow(image, cmap="magma")
        self.draw_idle()

    def show_image(self, image: np.ndarray, title: str = "", cmap: str = "gray") -> None:
        """Render a display image on the full canvas."""

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.axes.set_axis_off()
        self.axes.imshow(np.asarray(image, dtype=float), cmap=cmap)
        if title:
            self.axes.set_title(title)
        self.draw_idle()

    def show_mtf_curve(self, mtf_curve: np.ndarray) -> None:
        """Render a 1D MTF curve array with columns [freq, sagittal, tangential]."""

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        arr = np.asarray(mtf_curve, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            self.axes.text(0.5, 0.5, "无 MTF 曲线数据", ha="center", va="center")
            self.axes.set_axis_off()
            self.draw_idle()
            return
        self.axes.plot(arr[:, 0], arr[:, 1], label="Sagittal", color="#d95f02", linewidth=2)
        self.axes.plot(arr[:, 0], arr[:, 2], label="Tangential", color="#1f77b4", linewidth=2)
        self.axes.set_xlabel("Frequency (cycles/mm)")
        self.axes.set_ylabel("MTF")
        self.axes.set_ylim(0.0, 1.05)
        self.axes.grid(True, alpha=0.25)
        self.axes.legend(loc="best")
        self.draw_idle()

    def show_power_astigmatism(self, columns: list[str], data: np.ndarray) -> None:
        """Render power/astigmatism curves from lens metric table data."""

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        arr = np.asarray(data, dtype=float)
        required = [
            "theta_deg",
            "local_sagittal_error_D",
            "local_meridional_error_D",
            "local_mean_error_D",
            "local_astigmatism_D",
        ]
        if arr.ndim != 2 or not all(name in columns for name in required):
            self.axes.text(0.5, 0.5, "无光焦度/像散数据", ha="center", va="center")
            self.axes.set_axis_off()
            self.draw_idle()
            return
        idx = {name: columns.index(name) for name in required}
        x = arr[:, idx["theta_deg"]]
        self.axes.plot(x, arr[:, idx["local_sagittal_error_D"]], label="Local sagittal error", color="#d95f02", linewidth=2)
        self.axes.plot(x, arr[:, idx["local_meridional_error_D"]], label="Local meridional error", color="#1f77b4", linewidth=2)
        self.axes.plot(x, arr[:, idx["local_mean_error_D"]], label="Local mean error", color="#333333", linewidth=2)
        self.axes.plot(x, arr[:, idx["local_astigmatism_D"]], label="Local astigmatism", color="#d62728", linewidth=2, linestyle="--")
        self.axes.set_xlabel("Field angle (degree)")
        self.axes.set_ylabel("Power / astigmatism (D)")
        self.axes.grid(True, alpha=0.25)
        self.axes.legend(loc="best")
        self.draw_idle()

    def show_distortion_curve(self, columns: list[str], data: np.ndarray) -> None:
        """Render magnification and distortion-percent subplots."""

        self.figure.clear()
        arr = np.asarray(data, dtype=float)
        required = ["theta_deg", "magnification", "distortion_percent"]
        if arr.ndim != 2 or not all(name in columns for name in required):
            self.axes = self.figure.add_subplot(111)
            self.axes.text(0.5, 0.5, "无畸变曲线数据", ha="center", va="center")
            self.axes.set_axis_off()
            self.draw_idle()
            return
        idx = {name: columns.index(name) for name in required}
        x = arr[:, idx["theta_deg"]]
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212, sharex=ax1)
        ax1.plot(x, arr[:, idx["magnification"]], color="#d62728", linewidth=2)
        ax1.set_ylabel("Magnification")
        ax1.grid(True, alpha=0.25)
        ax2.plot(x, arr[:, idx["distortion_percent"]], color="#d62728", linewidth=2)
        ax2.set_xlabel("Field angle (degree)")
        ax2.set_ylabel("Distortion (%)")
        ax2.grid(True, alpha=0.25)
        self.axes = ax1
        self.draw_idle()

    def show_distortion_grid(self, regular: np.ndarray, distorted: np.ndarray, *, unit: str = "mm") -> None:
        """Render regular and distorted grid wireframes."""

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        regular_arr = np.asarray(regular, dtype=float)
        distorted_arr = np.asarray(distorted, dtype=float)
        if regular_arr.ndim != 3 or distorted_arr.ndim != 3 or regular_arr.shape[-1] < 2 or distorted_arr.shape[-1] < 2:
            self.axes.text(0.5, 0.5, "无畸变网格数据", ha="center", va="center")
            self.axes.set_axis_off()
            self.draw_idle()
            return

        def draw_grid(grid: np.ndarray, color: str, linewidth: float) -> None:
            for row in range(grid.shape[0]):
                self.axes.plot(grid[row, :, 0], grid[row, :, 1], color=color, linewidth=linewidth)
            for col in range(grid.shape[1]):
                self.axes.plot(grid[:, col, 0], grid[:, col, 1], color=color, linewidth=linewidth)

        draw_grid(regular_arr, "#1f77b4", 1.0)
        draw_grid(distorted_arr, "#d62728", 1.4)
        label_unit = "degree" if unit == "deg" else unit
        self.axes.set_xlabel(f"X ({label_unit})")
        self.axes.set_ylabel(f"Y ({label_unit})")
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.grid(True, alpha=0.2)
        self.axes.legend(["Regular", "Distorted"], loc="best")
        self.draw_idle()

    def show_mtf_grid(self, mtf_grid: np.ndarray) -> None:
        """Render sagittal/tangential cutoff MTF grids side by side."""

        self.figure.clear()
        arr = np.asarray(mtf_grid, dtype=float)
        if arr.ndim != 3 or arr.shape[2] < 2:
            self.axes = self.figure.add_subplot(111)
            self.axes.text(0.5, 0.5, "无 MTF 网格数据", ha="center", va="center")
            self.axes.set_axis_off()
            self.draw_idle()
            return
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        im1 = ax1.imshow(arr[:, :, 0], cmap="viridis", vmin=0.0, vmax=1.0)
        im2 = ax2.imshow(arr[:, :, 1], cmap="viridis", vmin=0.0, vmax=1.0)
        ax1.set_title("Sagittal cutoff MTF")
        ax2.set_title("Tangential cutoff MTF")
        ax1.set_xlabel("Field X index")
        ax1.set_ylabel("Field Y index")
        ax2.set_xlabel("Field X index")
        ax2.set_ylabel("Field Y index")
        self.figure.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        self.figure.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        self.axes = ax1
        self.draw_idle()

    @staticmethod
    def _five_degree_ticks(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        low = int(np.ceil(float(values.min()) / 5.0) * 5)
        high = int(np.floor(float(values.max()) / 5.0) * 5)
        ticks = np.arange(low, high + 1, 5, dtype=int)
        if ticks.size == 0:
            return values
        return ticks

    @staticmethod
    def _normalize_display(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax <= vmin:
            return np.zeros_like(arr)
        return (arr - vmin) / (vmax - vmin)

    def _style_field_axis(self, ax, field_x_values: np.ndarray, field_y_values: np.ndarray) -> None:
        ax.set_xlabel("field X (Degrees)", fontfamily="Times New Roman", fontsize=13)
        ax.set_ylabel("field Y (Degrees)", fontfamily="Times New Roman", fontsize=13)
        ax.set_xticks(self._five_degree_ticks(field_x_values))
        ax.set_yticks(self._five_degree_ticks(field_y_values))
        ax.tick_params(direction="in", top=True, right=True, labelsize=10)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily("Times New Roman")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    @staticmethod
    def _field_extent(field_x_values: np.ndarray, field_y_values: np.ndarray) -> list[float]:
        x_min = float(np.min(field_x_values))
        x_max = float(np.max(field_x_values))
        y_min = float(np.min(field_y_values))
        y_max = float(np.max(field_y_values))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        return [x_min, x_max, y_min, y_max]

    def show_field_stitch(
        self,
        image: np.ndarray,
        field_x_values: np.ndarray,
        field_y_values: np.ndarray,
        *,
        kind: str,
    ) -> None:
        """Render a field stitch with reference-style field axes.

        `kind` is "psf" or "chart". Inputs are display-only images; physical
        PSF and MTF calculations are not performed in the GUI layer.
        """

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        arr = np.asarray(image, dtype=float)
        field_x = np.asarray(field_x_values, dtype=float)
        field_y = np.asarray(field_y_values, dtype=float)
        if arr.size == 0 or field_x.size == 0 or field_y.size == 0:
            self.axes.text(0.5, 0.5, "无拼接图数据", ha="center", va="center")
            self.axes.set_axis_off()
            self.draw_idle()
            return

        extent = self._field_extent(field_x, field_y)
        if kind == "chart":
            display = 1.0 - self._normalize_display(arr)
            im = self.axes.imshow(
                display,
                extent=extent,
                origin="lower",
                cmap="gray_r",
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
            )
            cbar = self.figure.colorbar(im, ax=self.axes, fraction=0.046, pad=0.04)
            cbar.set_ticks([0.0, 0.5, 1.0])
            cbar.ax.invert_yaxis()
        else:
            vmax = float(np.nanmax(arr))
            im = self.axes.imshow(
                arr,
                extent=extent,
                origin="lower",
                cmap="jet",
                vmin=0.0,
                vmax=vmax if vmax > 0.0 else 1.0,
                aspect="auto",
            )
            cbar = self.figure.colorbar(im, ax=self.axes, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(direction="in", labelsize=10)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontfamily("Times New Roman")
        self._style_field_axis(self.axes, field_x, field_y)
        self.draw_idle()

    def show_mtf_value_grid(
        self,
        mtf_grid: np.ndarray,
        field_x_values: np.ndarray,
        field_y_values: np.ndarray,
    ) -> None:
        """Render a MATLAB-style two-number MTF grid.

        Each cell shows sagittal/horizontal in black and tangential/vertical in
        red, matching `build_mtf_weighted_grid.py` visual semantics.
        """

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        arr = np.asarray(mtf_grid, dtype=float)
        field_x = np.asarray(field_x_values, dtype=float)
        field_y = np.asarray(field_y_values, dtype=float)
        if arr.ndim != 3 or arr.shape[2] < 2:
            self.axes.text(0.5, 0.5, "无 MTF 网格数据", ha="center", va="center")
            self.axes.set_axis_off()
            self.draw_idle()
            return

        rows, cols = arr.shape[:2]
        self.axes.set_xlim(0, cols)
        self.axes.set_ylim(0, rows)
        self.axes.invert_yaxis()
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.set_facecolor("white")

        for col in range(cols + 1):
            self.axes.plot([col, col], [0, rows], color="black", linewidth=0.8)
        for row in range(rows + 1):
            self.axes.plot([0, cols], [row, row], color="black", linewidth=0.8)

        for row in range(rows):
            for col in range(cols):
                sag = arr[row, col, 0]
                tan = arr[row, col, 1]
                cx = col + 0.5
                cy = row + 0.5
                sag_text = "nan" if not np.isfinite(sag) else f"{sag:.4f}"
                tan_text = "nan" if not np.isfinite(tan) else f"{tan:.4f}"
                self.axes.text(cx, cy - 0.14, sag_text, ha="center", va="center", color="black", fontsize=9)
                self.axes.text(cx, cy + 0.18, tan_text, ha="center", va="center", color="#d20000", fontsize=9)

        if field_x.size == cols:
            self.axes.set_xticks(np.arange(cols) + 0.5)
            self.axes.set_xticklabels([f"{x:g}" for x in field_x], fontfamily="Times New Roman")
        else:
            self.axes.set_xticks(np.arange(cols) + 0.5)
        if field_y.size == rows:
            self.axes.set_yticks(np.arange(rows) + 0.5)
            self.axes.set_yticklabels([f"{y:g}" for y in field_y], fontfamily="Times New Roman")
        else:
            self.axes.set_yticks(np.arange(rows) + 0.5)
        self.axes.set_xlabel("field X (Degrees)", fontfamily="Times New Roman", fontsize=13)
        self.axes.set_ylabel("field Y (Degrees)", fontfamily="Times New Roman", fontsize=13)
        self.axes.tick_params(direction="in", top=True, right=True, labelsize=10)
        self.axes.set_title("Cutoff MTF: black=Sagittal, red=Tangential", fontsize=12)
        self.draw_idle()
