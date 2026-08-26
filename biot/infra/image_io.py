"""Image file IO that survives non-ASCII paths.

Why this module exists, in full, because the reason is not self-evident and a
previous sync from ``BIOT_vis`` (commit ``cebda87``) silently reverted
``visualization_utils.save_display_png`` back to a bare ``cv2.imwrite``:

``cv2.imwrite``/``cv2.imread`` take the path as a UTF-8 ``std::string`` and
convert it to wide characters with the Windows ANSI code page. Under cp936 (GBK)
that conversion pairs bytes greedily: a lead byte 0x81..0xFE claims the next
byte 0x40..0xFE as its trail. A directory name whose UTF-8 form has an ODD byte
count therefore leaves one lead byte over, which swallows the following path
separator 0x5C -- itself a legal GBK trail byte -- and the path loses a
directory level.

This repository lives under a 7-character CJK directory name: 21 UTF-8 bytes,
odd. Measured with ``MultiByteToWideChar(CP_ACP, ...)``, the asked-for
5-component ``D:\\VSCODE\\<cjk>\\results\\x.png`` arrives at OpenCV as a
4-component path with the mojibake name and ``results`` fused into one
component. The consequence depends on whether that fused parent happens to
exist, which is why the bug looked erratic:

- writing to the project root returned **True** and put the bytes in
  ``D:\\VSCODE\\`` under a mojibake name -- a silent wrong-location write;
- writing anywhere under ``results/`` returned **False**.

Encoding in memory and letting Python write the bytes keeps the path on the
wide-character API end to end. The pixel data and the OpenCV codecs are
untouched, so this is a path-handling fix, not a change to any image content.

Display/GUI artifacts only. No GPU or autograd support.
"""

from pathlib import Path

import cv2
import numpy as np


def write_cv_image(output_path: Path, image: np.ndarray) -> Path:
    """Encode an OpenCV image and write it through Python's Unicode path API.

    Inputs:
    - output_path: destination file; the suffix selects the encoder and must be
      present.
    - image: array in the layout the chosen encoder expects (uint8 grayscale or
      BGR for PNG).
    Output is ``output_path``. Raises rather than leaving a partial file when the
    encoder rejects the input. See the module docstring for why this must not be
    replaced by ``cv2.imwrite``.
    """

    output_path = Path(output_path)
    extension = output_path.suffix.lower()
    if not extension:
        raise ValueError(f"Image output path has no extension: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(extension, np.asarray(image))
    if not encoded_ok:
        raise RuntimeError(f"Failed to encode image for: {output_path}")
    output_path.write_bytes(encoded.tobytes())
    return output_path


def read_cv_image(image_path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image through Python's Unicode path API and ``cv2.imdecode``.

    ``cv2.imread`` corrupts non-ASCII paths exactly as ``cv2.imwrite`` does, but
    reports the failure only by returning ``None``.

    Inputs:
    - image_path: source file.
    - flags: OpenCV imread flag, e.g. ``cv2.IMREAD_COLOR``.
    Output is the decoded array. Raises when the file cannot be decoded, so a
    read failure never propagates as a ``None`` image.
    """

    image_path = Path(image_path)
    encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise RuntimeError(f"Failed to decode image: {image_path}")
    return image
