"""torch-rcc8: Differentiable Raster & Continuous-SDF RCC-8 in PyTorch."""

from .rcc8 import (
    RCC8,
    classify_rcc8,
)
from .mask_utils import (
    mask_to_polygon,
)
from .diagnostics import (
    kuratowski_closure,
    kuratowski_interior,
    kuratowski_opening,
    classify_rcc8_grid_jepd,
    RCC8_NAMES,
    morph_dilation,
    morph_erosion,
    compute_rcc8_matrix,
    fuzzy_rcc8_loss,
    DifferentiableFuzzyRCC8Loss,
    FuzzyRCC8Loss,
    ConformalRiskCalibrator,
)
from .modules import (
    compute_rcc8_ec_matrix,
    TopoRCC8Transformer,
)
from .persistent import (
    morph_open_clean,
    morph_close_clean,
    granulometric_persistence,
    compute_persistent_rcc8_matrix,
    PersistentAnchoredRCC8Loss,
)
from .sdf_solver import (
    exact_edt_2d,
    compute_sdf_2d,
    stl_and,
    stl_or,
    stl_always,
    stl_eventually,
    SDFShape,
    render_sdf_shape,
    SDFRCC8Loss,
    RigidToolTrajectoryPlanner,
    TopologicalTTAAdapter,
    RCC8SceneLayoutSolver,
)
from .sdf_rcc8 import (
    box_analytical_sdf,
    sample_sdf_2d,
    sdf_dc_loss,
    sdf_ntppi_stl_loss,
)

__all__ = [
    "RCC8",
    "classify_rcc8",
    "mask_to_polygon",
    "kuratowski_closure",
    "kuratowski_interior",
    "kuratowski_opening",
    "classify_rcc8_grid_jepd",
    "RCC8_NAMES",
    "morph_dilation",
    "morph_erosion",
    "compute_rcc8_matrix",
    "fuzzy_rcc8_loss",
    "DifferentiableFuzzyRCC8Loss",
    "FuzzyRCC8Loss",
    "ConformalRiskCalibrator",
    "compute_rcc8_ec_matrix",
    "TopoRCC8Transformer",
    "morph_open_clean",
    "morph_close_clean",
    "granulometric_persistence",
    "compute_persistent_rcc8_matrix",
    "PersistentAnchoredRCC8Loss",
    "exact_edt_2d",
    "compute_sdf_2d",
    "stl_and",
    "stl_or",
    "stl_always",
    "stl_eventually",
    "SDFShape",
    "render_sdf_shape",
    "SDFRCC8Loss",
    "RigidToolTrajectoryPlanner",
    "TopologicalTTAAdapter",
    "RCC8SceneLayoutSolver",
    "box_analytical_sdf",
    "sample_sdf_2d",
    "sdf_dc_loss",
    "sdf_ntppi_stl_loss",
]
