import torch
import torch.nn.functional as F

def box_analytical_sdf(local_xy: torch.Tensor, half_l: float, half_w: float) -> torch.Tensor:
    qx = torch.abs(local_xy[..., 0]) - half_l
    qy = torch.abs(local_xy[..., 1]) - half_w
    outside = torch.sqrt(torch.clamp(qx, min=0.0) ** 2 + torch.clamp(qy, min=0.0) ** 2 + 1e-8)
    return outside + torch.clamp(torch.maximum(qx, qy), max=0.0)

def sample_sdf_2d(sdf_2d: torch.Tensor, pts_xy: torch.Tensor) -> torch.Tensor:
    H, W = sdf_2d.shape
    orig = pts_xy.shape[:-1]
    pts = pts_xy.reshape(1, 1, -1, 2)
    grid = torch.stack([2.0 * (pts[..., 0] / max(W - 1.0, 1.0)) - 1.0,
                        2.0 * (pts[..., 1] / max(H - 1.0, 1.0)) - 1.0], dim=-1)
    return F.grid_sample(sdf_2d[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(*orig)

def sdf_dc_loss(obstacle_sdf: torch.Tensor, tool_world_pts: torch.Tensor, safety_margin: float = 1.5) -> torch.Tensor:
    viol = F.relu(safety_margin - sample_sdf_2d(obstacle_sdf, tool_world_pts))
    return (viol ** 2).mean() + 0.5 * viol.max()

def sdf_ntppi_stl_loss(poses_xy: torch.Tensor, poses_th: torch.Tensor, target_pts: torch.Tensor,
                       half_l: float, half_w: float, margin: float = 0.8, beta: float = 2.5) -> torch.Tensor:
    c, s = torch.cos(poses_th).unsqueeze(-1), torch.sin(poses_th).unsqueeze(-1)
    dx, dy = target_pts[:, 0] - poses_xy[..., 0:1], target_pts[:, 1] - poses_xy[..., 1:2]
    sd = box_analytical_sdf(torch.stack([c * dx + s * dy, -s * dx + c * dy], dim=-1), half_l, half_w)
    step_w = (1.0 / beta) * torch.logsumexp(beta * sd, dim=-1)
    ev_w = -(1.0 / beta) * torch.logsumexp(-beta * step_w, dim=-1)
    return F.relu(ev_w + margin).mean()
