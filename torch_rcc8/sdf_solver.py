r"""
torch_rcc8/sdf_solver.py
------------------------------
Pure-PyTorch GPU Signed Distance Field (SDF) RCC-8 Spatial Solver,
Signal Temporal Logic (STL) Trajectory Planner, Zero-Label Containment-Healing TTA,
and Vectorized Multi-Object Qualitative Scene Layout Engine.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .diagnostics import compute_rcc8_matrix, morph_dilation, morph_erosion
    from .persistent import granulometric_persistence
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from torch_rcc8.diagnostics import compute_rcc8_matrix, morph_dilation, morph_erosion
    from torch_rcc8.persistent import granulometric_persistence

def exact_edt_2d(mask: torch.Tensor, chunk_size: int = 4) -> torch.Tensor:
    """Exact separable 2D Euclidean Distance Transform on GPU/CPU in pure PyTorch."""
    orig_shape = mask.shape
    t = (mask > 0.5)
    while t.ndim < 4:
        t = t.unsqueeze(0)

    B, C, H, W = t.shape
    flat = t.reshape(B * C, H, W)
    device = mask.device
    max_d2 = float(H * H + W * W)
    inf_val = max_d2 * 4.0 + 1e6

    gx = torch.arange(W, device=device, dtype=torch.float32)
    gy = torch.arange(H, device=device, dtype=torch.float32)
    dx2 = (gx[:, None] - gx[None, :]) ** 2
    dy2 = (gy[:, None] - gy[None, :]) ** 2

    outs = []
    for start in range(0, B * C, chunk_size):
        sub = flat[start : start + chunk_size]
        has_fg = sub.any(dim=-1).any(dim=-1)
        f = torch.where(sub, 0.0, inf_val)
        d1 = (f.unsqueeze(-2) + dx2).amin(dim=-1)
        d2 = (d1.transpose(-2, -1).unsqueeze(-2) + dy2).amin(dim=-1).transpose(-2, -1)
        dist = torch.sqrt(d2.clamp(min=0.0, max=max_d2))
        if (~has_fg).any():
            dist = torch.where(has_fg[:, None, None], dist, math.hypot(H, W))
        outs.append(dist)

    return torch.cat(outs, dim=0).reshape(orig_shape)

def compute_sdf_2d(mask: torch.Tensor, border_is_outside: bool = True) -> torch.Tensor:
    """Half-pixel calibrated 2D Signed Distance Field (<0 inside, 0 on pixel edge, >0 outside)."""
    orig_shape = mask.shape
    t = (mask > 0.5).float()
    while t.ndim < 4:
        t = t.unsqueeze(0)

    d_out = exact_edt_2d(t)
    if border_is_outside:
        t_pad = F.pad(t, (1, 1, 1, 1), mode="constant", value=0.0)
        d_in = exact_edt_2d(1.0 - t_pad)[:, :, 1:-1, 1:-1]
    else:
        d_in = exact_edt_2d(1.0 - t)

    sdf = F.relu(d_out - 0.5) - F.relu(d_in - 0.5)
    return sdf.reshape(orig_shape)

def stl_and(r1: torch.Tensor, r2: torch.Tensor, tau: float = 0.15) -> torch.Tensor:
    stack = torch.stack([r1, r2], dim=-1)
    w = torch.softmax(-stack.detach() / tau, dim=-1)
    return (w * stack).sum(dim=-1)

def stl_or(r1: torch.Tensor, r2: torch.Tensor, tau: float = 0.15) -> torch.Tensor:
    stack = torch.stack([r1, r2], dim=-1)
    w = torch.softmax(stack.detach() / tau, dim=-1)
    return (w * stack).sum(dim=-1)

def stl_always(rob_seq: torch.Tensor, dim: int = -1, tau: float = 0.15) -> torch.Tensor:
    w = torch.softmax(-rob_seq.detach() / tau, dim=dim)
    return (w * rob_seq).sum(dim=dim)

def stl_eventually(rob_seq: torch.Tensor, dim: int = -1, tau: float = 0.15) -> torch.Tensor:
    w = torch.softmax(rob_seq.detach() / tau, dim=dim)
    return (w * rob_seq).sum(dim=dim)

@dataclass
class SDFShape:
    mask: torch.Tensor
    sdf: torch.Tensor       # (1, 1, H, W)
    centroid: torch.Tensor  # (2,)
    rel_bd: torch.Tensor    # (M_bd, 2)
    rel_all: torch.Tensor   # (M_bd + M_in, 2) where [:M_bd] == rel_bd

    @classmethod
    def from_mask(
        cls,
        mask_2d: torch.Tensor,
        max_bd_points: int = 256,
        max_in_points: int = 128,
        border_is_outside: bool = True,
    ) -> "SDFShape":
        if mask_2d.ndim != 2:
            raise ValueError(f"Expected 2D (H, W) mask, got {mask_2d.shape}")
        device = mask_2d.device
        m = (mask_2d > 0.5).float()
        sdf = compute_sdf_2d(m, border_is_outside=border_is_outside).unsqueeze(0).unsqueeze(0)

        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        if ys.numel() == 0:
            c = torch.tensor([mask_2d.shape[1] / 2.0, mask_2d.shape[0] / 2.0], device=device)
            z_bd = torch.zeros((max_bd_points, 2), device=device)
            z_all = torch.zeros((max_bd_points + max_in_points, 2), device=device)
            return cls(m, sdf, c, z_bd, z_all)

        pts_fg = torch.stack([xs.float(), ys.float()], dim=-1)
        centroid = pts_fg.mean(dim=0)

        hy, hx = torch.nonzero(m[:, :-1] != m[:, 1:], as_tuple=True)
        vy, vx = torch.nonzero(m[:-1, :] != m[1:, :], as_tuple=True)
        bd_list = []
        if hy.numel() > 0:
            bd_list.append(torch.stack([hx.float() + 0.5, hy.float()], dim=-1))
        if vy.numel() > 0:
            bd_list.append(torch.stack([vx.float(), vy.float() + 0.5], dim=-1))
        pts_bd = torch.cat(bd_list, dim=0) if bd_list else pts_fg

        # Uniform fixed-size resampling enables 100% batched multi-shape GPU tensor ops
        idx_bd = torch.linspace(0, pts_bd.shape[0] - 1, max_bd_points, device=device).long()
        idx_in = torch.linspace(0, pts_fg.shape[0] - 1, max_in_points, device=device).long()
        rel_bd = pts_bd[idx_bd] - centroid
        rel_in = pts_fg[idx_in] - centroid
        rel_all = torch.cat([rel_bd, rel_in], dim=0)
        return cls(m, sdf, centroid, rel_bd, rel_all)

def _transform_points(rel_pts: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Supports both single (P, 2) + (..., 4) and batched (B, P, 2) + (B, 4)."""
    tx, ty, th, sc = pose[..., 0:1], pose[..., 1:2], pose[..., 2:3], pose[..., 3:4]
    c, s = torch.cos(th), torch.sin(th)
    ux, uy = rel_pts[..., 0], rel_pts[..., 1]
    wx = tx + sc * (c * ux - s * uy)
    wy = ty + sc * (s * ux + c * uy)
    return torch.stack([wx, wy], dim=-1)

def _inv_transform_points(world_pts: torch.Tensor, pose: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    tx, ty, th = pose[..., 0:1], pose[..., 1:2], pose[..., 2:3]
    sc = pose[..., 3:4].clamp_min(1e-4)
    dx, dy = world_pts[..., 0] - tx, world_pts[..., 1] - ty
    c, s = torch.cos(th), torch.sin(th)
    ux = (c * dx + s * dy) / sc
    uy = (-s * dx + c * dy) / sc
    cx = centroid[..., 0:1] if centroid.ndim > 1 else centroid[0]
    cy = centroid[..., 1:2] if centroid.ndim > 1 else centroid[1]
    return torch.stack([cx + ux, cy + uy], dim=-1)

def _batched_sample_sdf(sdf_b1hw: torch.Tensor, coords_bp2: torch.Tensor) -> torch.Tensor:
    """Samples (B, 1, H, W) SDF at (B, P, 2) in a SINGLE F.grid_sample kernel launch."""
    B, _, H, W = sdf_b1hw.shape
    P = coords_bp2.shape[1]
    x_col, y_row = coords_bp2[:, :, 0], coords_bp2[:, :, 1]

    oob_x = F.relu(-x_col) + F.relu(x_col - float(W - 1))
    oob_y = F.relu(-y_row) + F.relu(y_row - float(H - 1))
    oob = torch.sqrt(oob_x * oob_x + oob_y * oob_y + 1e-8) - 1e-4

    gx = 2.0 * (x_col / max(W - 1, 1)) - 1.0
    gy = 2.0 * (y_row / max(H - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # (B, P, 1, 2)

    sampled = F.grid_sample(sdf_b1hw, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.reshape(B, P) + oob

def render_sdf_shape(shape: SDFShape, pose: torch.Tensor, out_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    H, W = out_hw if out_hw is not None else shape.mask.shape
    device = shape.mask.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    world_grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    batch_shape = pose.shape[:-1]
    B_flat = int(math.prod(batch_shape)) if len(batch_shape) > 0 else 1
    pose_2d = pose.reshape(B_flat, 4)
    exp_grid = world_grid.unsqueeze(0).expand(B_flat, H * W, 2)
    can_xy = _inv_transform_points(exp_grid, pose_2d, shape.centroid)
    sdf_exp = shape.sdf.expand(B_flat, 1, *shape.sdf.shape[-2:])
    sdf_vals = pose_2d[:, 3:4] * _batched_sample_sdf(sdf_exp, can_xy)
    return (sdf_vals < 0.0).float().reshape(*batch_shape, H, W)

def _smin(z: torch.Tensor, beta: float) -> torch.Tensor:
    w = torch.softmax(-beta * z.detach(), dim=-1)
    return (w * z).sum(dim=-1)

def _smax(z: torch.Tensor, beta: float) -> torch.Tensor:
    w = torch.softmax(beta * z.detach(), dim=-1)
    return (w * z).sum(dim=-1)

def _eval_rcc8_loss_from_samples(
    d_a_all: torch.Tensor,
    d_b_all: torch.Tensor,
    m_bd: int,
    rel: str,
    dc_margin: float,
    in_margin: float,
    beta: float,
) -> torch.Tensor:
    d_a_bd = d_a_all[..., :m_bd]
    d_b_bd = d_b_all[..., :m_bd]

    if rel == "DC":
        la = F.relu(dc_margin - _smin(d_a_all, beta)) + F.relu(dc_margin - d_a_all).mean(dim=-1)
        lb = F.relu(dc_margin - _smin(d_b_all, beta)) + F.relu(dc_margin - d_b_all).mean(dim=-1)
        return la + lb
    if rel == "NTPP":
        l_in = F.relu(_smax(d_a_bd, beta) + in_margin) + F.relu(d_a_all + in_margin).mean(dim=-1)
        l_prop = F.relu(in_margin - _smax(d_b_bd, beta))
        return l_in + l_prop
    if rel == "EC":
        min_ab = torch.minimum(d_a_bd.amin(dim=-1), d_b_bd.amin(dim=-1))
        no_pen = (
            F.relu(0.12 - d_a_all.amin(dim=-1))
            + F.relu(0.12 - d_b_all.amin(dim=-1))
            + 2.0 * (F.relu(0.12 - d_a_all).mean(dim=-1) + F.relu(0.12 - d_b_all).mean(dim=-1))
        )
        kiss = F.relu(min_ab - 0.24) + 0.25 * F.relu(_smin(d_a_bd, 2.5) - 0.35)
        return no_pen + kiss
    if rel == "TPP":
        max_a = d_a_bd.amax(dim=-1)
        no_out = F.relu(max_a + 0.08) + 2.0 * F.relu(d_a_all + 0.08).mean(dim=-1)
        kiss_in = F.relu(-0.32 - max_a) + 0.25 * F.relu(-0.35 - _smax(d_a_bd, 2.5))
        proper = F.relu(in_margin - d_b_bd.amax(dim=-1))
        return no_out + kiss_in + proper
    if rel == "PO":
        return (
            F.relu(_smin(d_a_all, beta) + 1.8)
            + F.relu(1.8 - _smax(d_a_bd, beta))
            + F.relu(1.8 - _smax(d_b_bd, beta))
        )
    if rel == "EQ":
        return (
            F.relu(_smax(d_a_bd, beta) + 0.2)
            + F.relu(_smax(d_b_bd, beta) + 0.2)
            + F.relu(d_a_all).mean(dim=-1)
            + F.relu(d_b_all).mean(dim=-1)
        )
    raise ValueError(f"Unsupported RCC-8 relation: {rel}")

class SDFRCC8Loss(nn.Module):
    """Differentiable SDF-based RCC-8 spatial loss & STL robustness evaluator."""
    def __init__(self, dc_margin: float = 2.2, in_margin: float = 2.2, beta: float = 6.0) -> None:
        super().__init__()
        self.dc_margin = dc_margin
        self.in_margin = in_margin
        self.beta = beta

    def pairwise_loss(
        self,
        shape_a: SDFShape,
        pose_a: Optional[torch.Tensor],
        shape_b: SDFShape,
        pose_b: Optional[torch.Tensor],
        relation: str,
        beta_override: Optional[float] = None,
    ) -> torch.Tensor:
        rel = relation.strip()
        beta = beta_override if beta_override is not None else self.beta
        device = shape_a.mask.device

        if pose_a is None:
            pose_a = torch.tensor([shape_a.centroid[0], shape_a.centroid[1], 0.0, 1.0], device=device)
        if pose_b is None:
            pose_b = torch.tensor([shape_b.centroid[0], shape_b.centroid[1], 0.0, 1.0], device=device)

        if rel in ("NTPPi", "TPPi"):
            inv_rel = "NTPP" if rel == "NTPPi" else "TPP"
            return self.pairwise_loss(shape_b, pose_b, shape_a, pose_a, inv_rel, beta_override=beta)

        bcast_shape = torch.broadcast_shapes(pose_a.shape[:-1], pose_b.shape[:-1])
        B_flat = int(math.prod(bcast_shape)) if len(bcast_shape) > 0 else 1
        pa = pose_a.expand(*bcast_shape, 4).reshape(B_flat, 4)
        pb = pose_b.expand(*bcast_shape, 4).reshape(B_flat, 4)

        w_all_a = _transform_points(shape_a.rel_all.unsqueeze(0).expand(B_flat, -1, 2), pa)
        w_all_b = _transform_points(shape_b.rel_all.unsqueeze(0).expand(B_flat, -1, 2), pb)

        c_a_in_b = _inv_transform_points(w_all_a, pb, shape_b.centroid)
        c_b_in_a = _inv_transform_points(w_all_b, pa, shape_a.centroid)

        d_a_all = pb[:, 3:4] * _batched_sample_sdf(shape_b.sdf.expand(B_flat, 1, *shape_b.sdf.shape[-2:]), c_a_in_b)
        d_b_all = pa[:, 3:4] * _batched_sample_sdf(shape_a.sdf.expand(B_flat, 1, *shape_a.sdf.shape[-2:]), c_b_in_a)

        out = _eval_rcc8_loss_from_samples(
            d_a_all, d_b_all, shape_a.rel_bd.shape[0], rel, self.dc_margin, self.in_margin, beta
        )
        return out.reshape(bcast_shape)

    def robustness(
        self,
        shape_a: SDFShape,
        pose_a: Optional[torch.Tensor],
        shape_b: SDFShape,
        pose_b: Optional[torch.Tensor],
        relation: str,
        margin: float = 0.25,
    ) -> torch.Tensor:
        return margin - self.pairwise_loss(shape_a, pose_a, shape_b, pose_b, relation)

class RigidToolTrajectoryPlanner:
    """SE(2) rigid tool/nozzle trajectory planner under STL + SDF-RCC8 constraints."""
    def __init__(self, dc_margin: float = 2.4, in_margin: float = 2.0) -> None:
        self.engine = SDFRCC8Loss(dc_margin=dc_margin, in_margin=in_margin, beta=6.0)

    def plan(
        self,
        tool_mask: torch.Tensor,
        obstacle_mask: torch.Tensor,
        target_mask: torch.Tensor,
        start_xy: Tuple[float, float],
        init_end_xy: Tuple[float, float],
        num_waypoints: int = 12,
        target_relation: str = "NTPPi",
        steps: int = 140,
        lr: float = 0.85,
        allow_rotation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = tool_mask.device
        H, W = obstacle_mask.shape
        tool_s = SDFShape.from_mask(tool_mask)
        obs_s = SDFShape.from_mask(obstacle_mask) if (obstacle_mask > 0.5).any() else None
        tgt_s = SDFShape.from_mask(target_mask)

        alphas = torch.linspace(0.0, 1.0, num_waypoints, device=device)
        init_x = start_xy[0] + alphas * (init_end_xy[0] - start_xy[0])
        init_y = start_xy[1] + alphas * (init_end_xy[1] - start_xy[1])

        free_xy = nn.Parameter(torch.stack([init_x[1:], init_y[1:]], dim=-1))
        free_th = nn.Parameter(torch.zeros(num_waypoints - 1, device=device))
        params = [free_xy, free_th] if allow_rotation else [free_xy]
        opt = torch.optim.Adam(params, lr=lr)

        fixed_start = torch.tensor([[start_xy[0], start_xy[1]]], device=device)
        zero_th0 = torch.zeros(1, device=device)
        ones_sc = torch.ones(num_waypoints, device=device)

        for step in range(steps):
            opt.zero_grad()
            xy = torch.cat([fixed_start, free_xy], dim=0)
            th = torch.cat([zero_th0, free_th], dim=0) if allow_rotation else torch.zeros(num_waypoints, device=device)
            poses = torch.stack([xy[:, 0], xy[:, 1], th, ones_sc], dim=-1)

            rob_tgt = self.engine.robustness(tool_s, poses, tgt_s, None, target_relation, margin=0.20)
            rob_eventually_tgt = stl_eventually(rob_tgt, tau=0.25)

            if obs_s is not None:
                rob_dc = self.engine.robustness(tool_s, poses, obs_s, None, "DC", margin=0.20)
                rob_always_safe = stl_always(rob_dc, tau=0.20)
                stl_rob = stl_and(rob_always_safe, rob_eventually_tgt, tau=0.20)
                direct_obs_pen = F.relu(-rob_dc).sum()
            else:
                stl_rob = rob_eventually_tgt
                direct_obs_pen = 0.0

            vel = xy[1:] - xy[:-1]
            smooth_loss = 0.008 * (vel * vel).sum(dim=-1).mean()
            w_pts = _transform_points(tool_s.rel_bd.unsqueeze(0).expand(num_waypoints, -1, 2), poses)
            canvas_pen = (
                F.relu(2.0 - w_pts[..., 0]).mean()
                + F.relu(w_pts[..., 0] - (W - 3.0)).mean()
                + F.relu(2.0 - w_pts[..., 1]).mean()
                + F.relu(w_pts[..., 1] - (H - 3.0)).mean()
            )

            loss = F.relu(0.35 - stl_rob) + 0.5 * direct_obs_pen + smooth_loss + 2.0 * canvas_pen
            loss.backward()
            opt.step()

            if step == steps // 3 and obs_s is not None:
                with torch.no_grad():
                    cur_xy = torch.cat([fixed_start, free_xy], dim=0)
                    cur_poses = torch.stack([cur_xy[:, 0], cur_xy[:, 1], torch.zeros_like(ones_sc), ones_sc], dim=-1)
                    l_dc = self.engine.pairwise_loss(tool_s, cur_poses[1:], obs_s, None, "DC")
                    if (l_dc > 2.0).any():
                        perp = torch.tensor([-init_end_xy[1] + start_xy[1], init_end_xy[0] - start_xy[0]], device=device)
                        perp = perp / (perp.norm() + 1e-6)
                        free_xy.add_(perp * 6.0 * (l_dc > 2.0).float().unsqueeze(-1))

        with torch.no_grad():
            xy_f = torch.cat([fixed_start, free_xy], dim=0)
            th_f = torch.cat([zero_th0, free_th], dim=0) if allow_rotation else torch.zeros(num_waypoints, device=device)
            final_poses = torch.stack([xy_f[:, 0], xy_f[:, 1], th_f, ones_sc], dim=-1)
            rendered_traj = render_sdf_shape(tool_s, final_poses, out_hw=(H, W))
        return final_poses, rendered_traj

class TopologicalTTAAdapter:
    """
    VRAM-safe Zero-Label Episodic Test-Time Adaptation for anatomical/structural containment healing
    ('NTPPi', 'NTPP', 'TPP'). Automatically chunks inputs by max_batch_size (default=4) to prevent OOM.
    """
    def __init__(
        self,
        model: nn.Module,
        target_relation: str = "NTPPi",
        kernel_size: int = 5,
        filtration_radii: Sequence[int] = (1,),
        steps: int = 8,
        lr: float = 1e-2,
        kl_anchor_weight: float = 1.0,
        max_batch_size: int = 4,
        episodic: bool = True,
    ) -> None:
        if target_relation not in ("NTPPi", "NTPP", "TPP"):
            raise ValueError(
                f"TopologicalTTAAdapter supports structural healing relations ('NTPPi', 'NTPP', 'TPP'), "
                f"got '{target_relation}'. For 'DC' false-positive pruning, use post-hoc masking or ConformalRiskCalibrator."
            )
        self.model = model
        self.target_relation = target_relation
        self.kernel_size = kernel_size
        self.filtration_radii = tuple(filtration_radii)
        self.steps = steps
        self.lr = lr
        self.kl_anchor_weight = kl_anchor_weight
        self.max_batch_size = max(1, int(max_batch_size))
        self.episodic = episodic

    def _collect_norm_params(self) -> List[nn.Parameter]:
        params = []
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                    params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)
        if not params:
            for p in list(self.model.parameters())[-2:]:
                p.requires_grad_(True)
                params.append(p)
        return params

    def _adapt_micro_batch(
        self,
        images: torch.Tensor,
        anchor_fn: Callable[[torch.Tensor], torch.Tensor],
        target_fn: Callable[[torch.Tensor], torch.Tensor],
        norm_params: List[nn.Parameter],
    ) -> torch.Tensor:
        saved_state = [p.detach().clone() for p in norm_params] if self.episodic else None

        with torch.no_grad():
            init_logits = self.model(images)
            init_probs = F.softmax(init_logits, dim=1)
            raw_anchor = torch.clamp(anchor_fn(init_probs).detach(), 0.0, 1.0)
            if raw_anchor.ndim == 3:
                raw_anchor = raw_anchor.unsqueeze(1)
            clean_anchor = granulometric_persistence(raw_anchor, self.filtration_radii, close_pinholes=True)

            if self.target_relation == "NTPPi":
                conflict_zone = morph_dilation(clean_anchor, self.kernel_size)
            elif self.target_relation == "NTPP":
                ero = morph_erosion(clean_anchor, self.kernel_size, border_value=0.0)
                conflict_zone = 1.0 - ero
            else:  # "TPP"
                ero = morph_erosion(clean_anchor, self.kernel_size, border_value=0.0)
                conflict_zone = (1.0 - clean_anchor) + 0.5 * ero

            safe_outside = (conflict_zone < 0.10).float()

        opt = torch.optim.Adam(norm_params, lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            logits = self.model(images)
            probs = F.softmax(logits, dim=1)
            p_tgt = target_fn(probs)
            if p_tgt.ndim == 3:
                p_tgt = p_tgt.unsqueeze(1)

            denom = conflict_zone.sum(dim=(-2, -1)).clamp_min(1e-5)
            if self.target_relation == "NTPPi":
                topo_loss = ((conflict_zone * (1.0 - p_tgt)).sum(dim=(-2, -1)) / denom).mean()
            else:
                topo_loss = ((conflict_zone * p_tgt).sum(dim=(-2, -1)) / denom).mean()

            log_p = F.log_softmax(logits, dim=1)
            kl_map = (init_probs * (torch.log(init_probs.clamp_min(1e-6)) - log_p)).sum(dim=1, keepdim=True)
            kl_loss = (kl_map * safe_outside).sum() / safe_outside.sum().clamp_min(1.0)

            (topo_loss + self.kl_anchor_weight * kl_loss).backward()
            opt.step()

        with torch.no_grad():
            out_logits = self.model(images).detach()
            if self.episodic and saved_state is not None:
                for p, s in zip(norm_params, saved_state):
                    p.copy_(s)
        return out_logits

    def adapt_and_predict(
        self,
        images: torch.Tensor,
        anchor_fn: Callable[[torch.Tensor], torch.Tensor],
        target_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        self.model.eval()
        norm_params = self._collect_norm_params()
        outputs = []
        for start in range(0, images.shape[0], self.max_batch_size):
            sub_imgs = images[start : start + self.max_batch_size]
            outputs.append(self._adapt_micro_batch(sub_imgs, anchor_fn, target_fn, norm_params))
        return torch.cat(outputs, dim=0)

class RCC8SceneLayoutSolver:
    """
    Fully vectorized GPU multi-object RCC-8 scene layout solver.
    Batches all C constraints into 2 grid_sample launches per step (~30-50x faster than loop dispatch).
    """
    def __init__(self, canvas_hw: Tuple[int, int] = (96, 96), dc_margin: float = 2.4, in_margin: float = 2.4) -> None:
        self.canvas_hw = canvas_hw
        self.dc_margin = dc_margin
        self.in_margin = in_margin

    def solve(
        self,
        shape_masks: Sequence[torch.Tensor] | Sequence[SDFShape],
        constraints: Sequence[Tuple[int, int, str]],
        init_poses: Optional[torch.Tensor] = None,
        fixed_indices: Sequence[int] = (0,),
        allow_rotation: bool = False,
        scale_bounds: Tuple[float, float] = (0.35, 1.15),
        steps: int = 200,
        lr: float = 0.85,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, W = self.canvas_hw
        N = len(shape_masks)
        shapes: List[SDFShape] = [
            s if isinstance(s, SDFShape) else SDFShape.from_mask(s, border_is_outside=True)
            for s in shape_masks
        ]
        device = shapes[0].mask.device
        m_bd = shapes[0].rel_bd.shape[0]
        s_min, s_max = scale_bounds

        # Stack all N shapes into contiguous GPU batch tensors ONCE before optimization
        all_sdfs = torch.cat([sh.sdf for sh in shapes], dim=0)            # (N, 1, H, W)
        centroids = torch.stack([sh.centroid for sh in shapes], dim=0)    # (N, 2)
        rel_all_stack = torch.stack([sh.rel_all for sh in shapes], dim=0) # (N, P, 2)

        # Normalize inverse relations (NTPPi -> NTPP, TPPi -> TPP) and group indices
        ia_list, ib_list, norm_rels = [], [], []
        for i, j, r in constraints:
            r_clean = r.strip()
            if r_clean == "NTPPi":
                ia_list.append(j); ib_list.append(i); norm_rels.append("NTPP")
            elif r_clean == "TPPi":
                ia_list.append(j); ib_list.append(i); norm_rels.append("TPP")
            else:
                ia_list.append(i); ib_list.append(j); norm_rels.append(r_clean)

        idx_a = torch.tensor(ia_list, dtype=torch.long, device=device)
        idx_b = torch.tensor(ib_list, dtype=torch.long, device=device)
        sdf_a_batch = all_sdfs[idx_a]   # (C, 1, H, W)
        sdf_b_batch = all_sdfs[idx_b]   # (C, 1, H, W)
        ca_batch = centroids[idx_a]     # (C, 2)
        cb_batch = centroids[idx_b]     # (C, 2)

        # Precompute boolean masks per relation type for zero-loop loss aggregation
        rel_groups = {}
        for r_type in set(norm_rels):
            rel_groups[r_type] = torch.tensor([k for k, nr in enumerate(norm_rels) if nr == r_type],
                                              dtype=torch.long, device=device)

        if init_poses is None:
            init_poses = torch.zeros((N, 4), device=device)
            init_poses[:, 0] = centroids[:, 0]
            init_poses[:, 1] = centroids[:, 1]
            init_poses[:, 3] = 0.75

        tx_p = nn.Parameter(init_poses[:, 0].clone())
        ty_p = nn.Parameter(init_poses[:, 1].clone())
        th_p = nn.Parameter(init_poses[:, 2].clone())
        init_norm_s = ((init_poses[:, 3] - s_min) / max(s_max - s_min, 1e-4)).clamp(0.05, 0.95)
        raw_s = nn.Parameter(torch.logit(init_norm_s))

        fixed_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for fi in fixed_indices:
            fixed_mask[fi] = True
        movable_idx = torch.nonzero(~fixed_mask, as_tuple=True)[0]

        opt = torch.optim.Adam([tx_p, ty_p, th_p, raw_s], lr=lr)

        for step in range(steps):
            opt.zero_grad()
            beta = 5.0 + 9.0 * (step / max(steps - 1, 1))

            tx = torch.where(fixed_mask, init_poses[:, 0], tx_p)
            ty = torch.where(fixed_mask, init_poses[:, 1], ty_p)
            th = torch.where(fixed_mask, init_poses[:, 2], th_p) if allow_rotation else init_poses[:, 2]
            sc_dyn = s_min + (s_max - s_min) * torch.sigmoid(raw_s)
            sc = torch.where(fixed_mask, init_poses[:, 3], sc_dyn)
            poses = torch.stack([tx, ty, th, sc], dim=-1)  # (N, 4)

            # 1. Transform ALL N shapes to world coordinates in one shot: (N, P, 2)
            world_all = _transform_points(rel_all_stack, poses)

            # 2. Gather C constraint pairs & sample SDFs in 2 batched grid_sample launches
            wa, wb = world_all[idx_a], world_all[idx_b]
            pa, pb = poses[idx_a], poses[idx_b]
            can_a_in_b = _inv_transform_points(wa, pb, cb_batch)
            can_b_in_a = _inv_transform_points(wb, pa, ca_batch)

            d_a_all = pb[:, 3:4] * _batched_sample_sdf(sdf_b_batch, can_a_in_b)  # (C, P)
            d_b_all = pa[:, 3:4] * _batched_sample_sdf(sdf_a_batch, can_b_in_a)  # (C, P)

            total_loss = torch.tensor(0.0, device=device)
            for r_type, g_idx in rel_groups.items():
                total_loss = total_loss + _eval_rcc8_loss_from_samples(
                    d_a_all[g_idx], d_b_all[g_idx], m_bd, r_type, self.dc_margin, self.in_margin, beta
                ).sum()

            # 3. Vectorized canvas bounds penalty across all movable shapes
            if movable_idx.numel() > 0:
                mov_bd = world_all[movable_idx, :m_bd, :]
                total_loss = total_loss + 2.0 * (
                    F.relu(3.0 - mov_bd[..., 0]).mean()
                    + F.relu(mov_bd[..., 0] - (W - 4.0)).mean()
                    + F.relu(3.0 - mov_bd[..., 1]).mean()
                    + F.relu(mov_bd[..., 1] - (H - 4.0)).mean()
                ) * movable_idx.numel()

            total_loss = total_loss + 0.05 * ((sc_dyn - 0.60) ** 2).mean()
            total_loss.backward()
            opt.step()

        with torch.no_grad():
            tx = torch.where(fixed_mask, init_poses[:, 0], tx_p)
            ty = torch.where(fixed_mask, init_poses[:, 1], ty_p)
            th = torch.where(fixed_mask, init_poses[:, 2], th_p) if allow_rotation else init_poses[:, 2]
            sc_dyn = s_min + (s_max - s_min) * torch.sigmoid(raw_s)
            sc = torch.where(fixed_mask, init_poses[:, 3], sc_dyn)
            final_poses = torch.stack([tx, ty, th, sc], dim=-1)
            rendered = torch.stack([render_sdf_shape(shapes[i], final_poses[i], out_hw=(H, W)) for i in range(N)], dim=0)
            rcc8_mat = compute_rcc8_matrix(rendered, kernel_size=3)

        return final_poses, rendered, rcc8_mat
