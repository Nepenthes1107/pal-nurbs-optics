"""Helpers for running subprocesses inside the BIOT conda environment.

GUI and batch scripts may be launched from an unactivated shell.  Importing
PyTorch from that state can load the wrong DLL search path on Windows.  This
module constructs an environment equivalent to ``conda activate myenv`` before
calling subprocesses.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_CONDA_BASE = Path(r"D:\Anaconda")
_ENV_NAME = "myenv"


def _find_env_dir(env_name: str = _ENV_NAME) -> Optional[Path]:
    candidates = []
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        candidates.append(Path(prefix))
    candidates.append(_CONDA_BASE / "envs" / env_name)

    for path in candidates:
        if path.exists():
            return path
    return None


def python_exe(env_name: str = _ENV_NAME) -> str:
    """Return the requested conda environment's python.exe, or sys.executable."""
    env_dir = _find_env_dir(env_name)
    if env_dir and (env_dir / "python.exe").exists():
        return str(env_dir / "python.exe")
    return sys.executable


def _conda_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build environment variables with conda activation paths first."""
    env = dict(os.environ if base_env is None else base_env)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    env_dir = _find_env_dir()
    if env_dir is not None:
        lib_bin = env_dir / "Library" / "bin"
        scripts = env_dir / "Scripts"
        condabin = _CONDA_BASE / "condabin"
        prepend = [str(env_dir), str(lib_bin), str(scripts), str(condabin)]
        old_path = env.get("PATH", "")
        env["PATH"] = os.pathsep.join(prepend + [old_path])
        env["CONDA_PREFIX"] = str(env_dir)
        env["CONDA_DEFAULT_ENV"] = _ENV_NAME
    return env


def run(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run a subprocess with an environment equivalent to ``conda activate``."""
    user_env = kwargs.pop("env", None)
    kwargs["env"] = _conda_env(user_env)
    return subprocess.run(cmd, **kwargs)
