"""PAL-NURBS 使用的可微真实光学核心。

该包只公开与镜片/人眼追迹、OPD、物理 PSF 和 B-spline/NURBS 表面有关的
基础能力。实验 runner、恢复网络和历史 Phase 框架不在公共接口中。
"""

from .config import E2EConfig, as_torch_device_dtype, set_random_seed
from .psf_fft import TorchFFTPSFResult, torch_fft_psf_from_phase
from .rays import RayBundle, make_pupil_rays, pupil_disk_grid
from .regional_nurbs import FixedWeightNURBSPerturbation
from .tracing import PALSurface, PlaneSurface, SphericalSurface, TraceResult, trace_to_image_plane

__all__ = [
    "E2EConfig",
    "FixedWeightNURBSPerturbation",
    "PALSurface",
    "PlaneSurface",
    "RayBundle",
    "SphericalSurface",
    "TorchFFTPSFResult",
    "TraceResult",
    "as_torch_device_dtype",
    "make_pupil_rays",
    "pupil_disk_grid",
    "set_random_seed",
    "torch_fft_psf_from_phase",
    "trace_to_image_plane",
]
