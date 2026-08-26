from __future__ import annotations

import json
from pathlib import Path


def save_manifest(result, output_dir: Path | None = None) -> Path:
    """Save a minimal result manifest next to service artifacts."""

    target_dir = Path(output_dir or result.output_dir or ".")
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def load_manifest(path: Path, result_cls):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return result_cls.from_dict(data)

