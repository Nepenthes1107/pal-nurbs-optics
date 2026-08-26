import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None or importlib.util.find_spec("pytestqt") is None,
    reason="GUI dependencies are not installed",
)


def test_main_window_smoke(qtbot):
    from biot.gui.main_window import MainWindow

    repo_root = Path(__file__).resolve().parents[1]
    window = MainWindow(default_excel=repo_root / "eye_image_glass.xlsx")
    qtbot.addWidget(window)

    assert window.tabs.count() == 6
    assert window.system_tab.config is not None
    assert window.single_field_tab.system_config is not None
    assert window.sweep_tab.system_config is not None
    assert window.power_tab.system_config is not None
    assert window.distortion_tab.system_config is not None
    assert "系统配置" == window.tabs.tabText(0)
    assert "单视角分析" == window.tabs.tabText(1)
    assert "范围扫描" == window.tabs.tabText(2)
    assert "光焦度像散" == window.tabs.tabText(3)
    assert "畸变分析" == window.tabs.tabText(4)
    assert window.tabs.isTabEnabled(2)
    assert window.tabs.isTabEnabled(3)
    assert window.tabs.isTabEnabled(4)
    assert window.tabs.isTabEnabled(5)
    assert window.export_tab is not None
    assert not hasattr(window.distortion_tab, "fix_grid_axis_bug_check")
