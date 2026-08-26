"""Lock the Unicode-path image IO in ``biot/infra/image_io.py``.

``cv2.imwrite``/``cv2.imread`` hand the path to OpenCV as a UTF-8 byte string,
which Windows converts with the ANSI code page. Under cp936 a lead byte
0x81..0xFE claims the next byte 0x40..0xFE as its trail, so a directory name
whose UTF-8 form has an ODD byte count leaves a dangling lead byte that swallows
the following separator 0x5C. This repository lives under a 7-character CJK
directory name -- 21 UTF-8 bytes -- so every OpenCV path write inside the
project tree lost a directory level: it either returned False (under
``results/``) or, worse, returned True after putting the bytes in the parent
directory under a mojibake name (at the project root).

``write_cv_image`` has existed since ``c242608`` (2026-07-22) for exactly this
reason, but ``cebda87`` (2026-08-05) reverted ``save_display_png`` to a bare
``cv2.imwrite`` while syncing from ``BIOT_vis``. These tests therefore lock both
the behaviour and the absence of the raw calls at every live call site, so the
next sync cannot silently reintroduce it.

The failure geometry is exercised directly (odd-byte CJK directory, nested
subdirectory), so the tests stay meaningful on a POSIX runner too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biot.infra.image_io import read_cv_image, write_cv_image  # noqa: E402
from biot.services.visualization_utils import save_display_png  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# 7 个汉字 = 21 个 UTF-8 字节（奇数），正是触发 GBK 吞掉分隔符的结构。
ODD_BYTE_CJK_DIR = "端到端光学设计"

# 所有仍在使用 OpenCV 图像读写的活代码；冻结副本 results/ 不在其中。
LIVE_CALL_SITES = (
    Path("biot/services/visualization_utils.py"),
    Path("biot/infra/image_io.py"),
    Path("mtf_utils.py"),
)


def test_the_directory_name_that_broke_opencv_still_has_an_odd_utf8_length() -> None:
    """The bug needs an odd UTF-8 byte count; assert the premise, not the symptom."""

    encoded = ODD_BYTE_CJK_DIR.encode("utf-8")
    assert len(encoded) == 21
    assert len(encoded) % 2 == 1
    # 分隔符不在名字里；被吞掉的是名字之后的那个分隔符。
    assert 0x5C not in encoded


def test_write_cv_image_writes_into_an_odd_byte_cjk_directory(tmp_path: Path) -> None:
    target = tmp_path / ODD_BYTE_CJK_DIR / "results" / "probe.png"
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)

    returned = write_cv_image(target, image)

    assert returned == target
    assert target.is_file()
    assert target.stat().st_size > 0
    # 关键回归：字节没有落到上一层去（曾经的静默错位写）。
    assert sorted(p.name for p in target.parent.parent.iterdir()) == ["results"]


def test_the_written_png_round_trips_bit_exactly(tmp_path: Path) -> None:
    target = tmp_path / ODD_BYTE_CJK_DIR / "results" / "round_trip.png"
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)

    write_cv_image(target, image)
    decoded = read_cv_image(target, cv2.IMREAD_GRAYSCALE)

    assert decoded.dtype == np.uint8
    assert decoded.shape == image.shape
    assert np.array_equal(decoded, image)


def test_a_deeper_nesting_under_the_cjk_directory_also_survives(tmp_path: Path) -> None:
    """The swallowed separator is the one after the CJK name, so depth must not matter."""

    target = tmp_path / ODD_BYTE_CJK_DIR / "results" / "test_service_single_field" / "chart.png"

    write_cv_image(target, np.full((4, 4), 200, dtype=np.uint8))

    assert target.is_file()
    assert read_cv_image(target, cv2.IMREAD_GRAYSCALE).shape == (4, 4)


def test_save_display_png_normalizes_and_writes_under_a_cjk_path(tmp_path: Path) -> None:
    target = tmp_path / ODD_BYTE_CJK_DIR / "display.png"
    image = np.array([[0.0, 0.5], [1.0, np.nan]], dtype=np.float64)

    save_display_png(image, target)
    decoded = read_cv_image(target, cv2.IMREAD_GRAYSCALE)

    assert decoded.shape == (2, 2)
    # nan 归零、最大值到 255：显示归一化仍然生效，不是被写盘路径改写绕过。
    assert int(decoded[1, 0]) == 255
    assert int(decoded[0, 0]) == 0
    assert int(decoded[1, 1]) == 0


def test_invert_is_applied_before_encoding(tmp_path: Path) -> None:
    target = tmp_path / ODD_BYTE_CJK_DIR / "inverted.png"

    save_display_png(np.array([[0.0, 1.0]], dtype=np.float64), target, invert=True)
    decoded = read_cv_image(target, cv2.IMREAD_GRAYSCALE)

    assert int(decoded[0, 0]) == 255
    assert int(decoded[0, 1]) == 0


def test_a_missing_extension_is_rejected_before_any_write(tmp_path: Path) -> None:
    target = tmp_path / ODD_BYTE_CJK_DIR / "no_extension"
    with pytest.raises(ValueError):
        write_cv_image(target, np.zeros((4, 4), dtype=np.uint8))
    assert not target.exists()


def test_an_unencodable_target_fails_loudly_instead_of_writing_a_stub(tmp_path: Path) -> None:
    """A failed encode must raise, not leave a truncated or empty file behind."""

    # 后缀选择编码器；未知后缀没有编码器可用。
    target = tmp_path / ODD_BYTE_CJK_DIR / "bad.notanimage"
    with pytest.raises((RuntimeError, cv2.error)):
        write_cv_image(target, np.zeros((4, 4), dtype=np.uint8))
    assert not target.exists()


def test_a_missing_file_is_reported_rather_than_returning_none(tmp_path: Path) -> None:
    missing = tmp_path / ODD_BYTE_CJK_DIR / "absent.png"
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        read_cv_image(missing)


@pytest.mark.parametrize("relative_path", LIVE_CALL_SITES, ids=lambda p: p.name)
def test_no_live_module_calls_the_ansi_path_api(relative_path: Path) -> None:
    """Guard the exact regression cebda87 introduced when syncing from BIOT_vis."""

    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert "cv2.imwrite(" not in text, f"{relative_path} reintroduced cv2.imwrite"
    assert "cv2.imread(" not in text, f"{relative_path} reintroduced cv2.imread"


def test_the_shared_helper_is_the_only_place_that_encodes() -> None:
    """Keep one implementation, so the reason is documented in one place."""

    helper = (REPO_ROOT / "biot/infra/image_io.py").read_text(encoding="utf-8")
    assert "cv2.imencode(" in helper
    assert "cv2.imdecode(" in helper
    # 根因必须留在代码里：上一次回退正是因为这里只写了"怎么做"没写"为什么"。
    assert "0x5C" in helper
    for other in LIVE_CALL_SITES:
        if other.name == "image_io.py":
            continue
        text = (REPO_ROOT / other).read_text(encoding="utf-8")
        assert "cv2.imencode(" not in text, f"{other} should use write_cv_image"
