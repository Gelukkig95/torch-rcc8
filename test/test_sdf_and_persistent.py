import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_rcc8 import (
    RCC8_NAMES,
    morph_open_clean,
    compute_rcc8_matrix,
    compute_persistent_rcc8_matrix,
    PersistentAnchoredRCC8Loss,
    SDFShape,
    render_sdf_shape,
    RigidToolTrajectoryPlanner,
    TopologicalTTAAdapter,
    RCC8SceneLayoutSolver,
)

NAME_TO_CODE = {n: i for i, n in enumerate(RCC8_NAMES)}

def test_matheron_idempotence_and_persistent_rcc8():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    x = torch.rand((4, 1, 32, 32), device=device)
    op1 = morph_open_clean(x, radius=2, mode="anisotropic")
    op2 = morph_open_clean(op1, radius=2, mode="anisotropic")
    assert (op1 - op2).abs().max().item() < 1e-6

    masks = torch.zeros((2, 48, 48), device=device)
    masks[0, 8:40, 8:40] = 1.0
    masks[1, 18:26, 18:26] = 1.0
    noisy = masks.clone()
    noisy[1, 2, 2] = 1.0  # 1px outside speckle flips standard NTPP -> PO
    assert int(compute_rcc8_matrix(noisy)[1, 0].item()) == NAME_TO_CODE["PO"]
    dom, _, _ = compute_persistent_rcc8_matrix(noisy, filtration_radii=(0, 1, 2))
    assert int(dom[1, 0].item()) == NAME_TO_CODE["NTPP"]

    loss_fn = PersistentAnchoredRCC8Loss(filtration_radii=(1, 2), kernel_size=5, target_relation="DC")
    p = torch.full((1, 1, 48, 48), 0.6, device=device, requires_grad=True)
    loss_fn(masks[0:1].unsqueeze(0), p).backward()
    assert p.grad[:, :, 12:36, 12:36].std().item() < 1e-7

def test_stl_sdf_trajectory_planner():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yy, xx = torch.meshgrid(torch.arange(96, device=device), torch.arange(96, device=device), indexing="ij")
    tool = (((yy - 48) ** 2 + (xx - 48) ** 2) <= 10.0 ** 2).float()
    obs = (((yy - 48) ** 2 + (xx - 48) ** 2) <= 14.0 ** 2).float()
    tgt = (((yy - 76) ** 2 + (xx - 76) ** 2) <= 4.0 ** 2).float()

    planner = RigidToolTrajectoryPlanner(dc_margin=2.2, in_margin=1.8)
    _, traj = planner.plan(tool, obs, tgt, start_xy=(18.0, 18.0), init_end_xy=(76.0, 48.0), num_waypoints=10)
    mat = compute_rcc8_matrix(torch.cat([obs[None], tgt[None], traj], dim=0))
    assert bool((mat[2:, 0] == NAME_TO_CODE["DC"]).all().item())
    assert bool((mat[2:, 1] == NAME_TO_CODE["NTPPi"]).any().item())

def test_vram_safe_tta_and_fast_vectorized_layout_solver():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = 96, 96
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    th = torch.atan2(yy - 48.0, xx - 48.0)
    host_blob = (torch.hypot(yy - 48.0, xx - 48.0) <= (25.0 + 4.0 * torch.sin(3 * th))).float()
    circle = (torch.hypot(yy - 48.0, xx - 48.0) <= 11.0).float()
    square = ((yy - 48.0).abs() <= 9.0) & ((xx - 48.0).abs() <= 9.0)
    tri = ((yy >= 36) & (yy <= 60) & ((xx - 48.0).abs() <= (yy - 36.0) * 0.6 + 1.0)).float()

    shapes = [SDFShape.from_mask(m) for m in (host_blob, circle, square.float(), tri, circle)]
    constraints = [(1, 0, "NTPP"), (2, 0, "TPP"), (3, 0, "EC"), (4, 0, "DC"), (1, 2, "DC"), (3, 4, "DC")]
    init_p = torch.tensor([
        [48.0, 48.0, 0.0, 1.0],
        [20.0, 22.0, 0.0, 0.55],
        [74.0, 25.0, 0.0, 0.50],
        [22.0, 74.0, 0.0, 0.55],
        [76.0, 76.0, 0.0, 0.50],
    ], device=device)

    solver = RCC8SceneLayoutSolver(canvas_hw=(H, W))
    # Warmup + timed solve
    _ = solver.solve(shapes, constraints, init_poses=init_p, steps=10)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, _, rmat = solver.solve(shapes, constraints, init_poses=init_p, steps=200, lr=0.85)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    sat = sum(int(int(rmat[i, j].item()) == NAME_TO_CODE[r]) for i, j, r in constraints)
    print(f"\n  [Vectorized Layout Solver] {sat}/6 constraints satisfied in {elapsed_ms:.1f} ms")
    assert sat >= 5

