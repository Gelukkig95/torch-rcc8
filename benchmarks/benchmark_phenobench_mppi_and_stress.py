import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# Geometry & SDF Utilities (기존 632장 벤치마크와 100% 동일 규격)
# ------------------------------------------------------------------------------
def wrap_angle_np(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrap_angle_torch(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def compute_sdf_2d_np(mask_np: np.ndarray) -> np.ndarray:
    m = (mask_np > 0.5).astype(np.uint8)
    if m.sum() == 0:
        return np.full(mask_np.shape, 999.0, dtype=np.float32)
    if m.sum() == m.size:
        return np.full(mask_np.shape, -999.0, dtype=np.float32)
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return (dist_out - dist_in).astype(np.float32)


def bilinear_sample_np(field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    H, W = field.shape
    x = np.clip(xs, 0.0, W - 1.0)
    y = np.clip(ys, 0.0, H - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    wx = x - x0
    wy = y - y0
    v00 = field[y0, x0]
    v01 = field[y0, x1]
    v10 = field[y1, x0]
    v11 = field[y1, x1]
    return (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)


def sample_sdf_batched(sdf_2d: torch.Tensor, pts_xy: torch.Tensor) -> torch.Tensor:
    H, W = sdf_2d.shape
    B, T, K, _ = pts_xy.shape
    gx = 2.0 * (pts_xy[..., 0] / (W - 1.0)) - 1.0
    gy = 2.0 * (pts_xy[..., 1] / (H - 1.0)) - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(1, B * T, K, 2)
    vals = F.grid_sample(
        sdf_2d[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return vals.reshape(B, T, K)


def nozzle_analytical_sdf_torch(local_xy: torch.Tensor, half_l: float, half_w: float) -> torch.Tensor:
    qx = torch.abs(local_xy[..., 0]) - half_l
    qy = torch.abs(local_xy[..., 1]) - half_w
    outside = torch.sqrt(torch.clamp(qx, min=0.0) ** 2 + torch.clamp(qy, min=0.0) ** 2 + 1e-8)
    inside = torch.clamp(torch.maximum(qx, qy), max=0.0)
    return outside + inside


def nozzle_analytical_sdf_np(weed_pts: np.ndarray, x: float, y: float, th: float, half_l: float, half_w: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    dx = weed_pts[:, 0] - x
    dy = weed_pts[:, 1] - y
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    qx = np.abs(lx) - half_l
    qy = np.abs(ly) - half_w
    outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside


def get_nozzle_local_points_np(half_l: float, half_w: float) -> np.ndarray:
    xs = np.linspace(-half_l, half_l, 9, dtype=np.float32)
    ys = np.linspace(-half_w, half_w, 5, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)


def compute_pca_from_pts(pts: np.ndarray):
    mean = pts.mean(axis=0)
    centered = pts - mean
    if len(pts) >= 3:
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        major_vec = eigvecs[:, 1]
        pca_th = float(math.atan2(major_vec[1], major_vec[0]))
    else:
        pca_th = 0.0
    return (float(mean[0]), float(mean[1])), pca_th


# ------------------------------------------------------------------------------
# Exact Scene Extractor (632개 동일 순서/인스턴스 + 마스크 반환 추가)
# ------------------------------------------------------------------------------
def extract_phenobench_scene(sem_path: Path, grid_size: int, half_l: float, half_w: float, nozzle_pts_np: np.ndarray):
    sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED)
    if sem is None:
        return None, "read_error"
    if sem.ndim == 3:
        sem = sem[..., 0]

    sem_rs = cv2.resize(sem, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
    crop_np = ((sem_rs == 1) | (sem_rs == 3)).astype(np.uint8)
    weed_all_np = ((sem_rs == 2) | (sem_rs == 4)).astype(np.uint8)

    if crop_np.sum() < 10:
        return None, "no_crop"
    if weed_all_np.sum() < 4:
        return None, "no_weed"

    crop_sdf_np = compute_sdf_2d_np(crop_np)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weed_all_np, connectivity=8)
    candidates = []
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        ys, xs = np.nonzero(labels == lbl)
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=-1)
        mean = pts.mean(axis=0)
        centered = pts - mean

        if len(pts) >= 3:
            cov = np.cov(centered.T)
            _, eigvecs = np.linalg.eigh(cov)
            major_vec = eigvecs[:, 1]
            minor_vec = eigvecs[:, 0]
            proj_maj = centered @ major_vec
            proj_min = centered @ minor_vec
            span_maj = float(proj_maj.max() - proj_maj.min())
            span_min = float(proj_min.max() - proj_min.min())
            pca_th = float(math.atan2(major_vec[1], major_vec[0]))
        else:
            span_maj, span_min, pca_th = 2.0, 2.0, 0.0

        if span_maj > (2.0 * half_l - 1.5) or span_min > (2.0 * half_w - 1.0):
            continue

        min_crop_dist = float(crop_sdf_np[ys, xs].min())
        candidates.append({
            "lbl": lbl,
            "pts": pts,
            "centroid": (float(mean[0]), float(mean[1])),
            "pca_th": pca_th,
            "min_crop_dist": min_crop_dist,
        })

    if not candidates:
        return None, "oversized_or_tiny_weed"

    separated = [c for c in candidates if c["min_crop_dist"] >= 1.5]
    target = min(separated, key=lambda c: c["min_crop_dist"]) if separated else max(candidates, key=lambda c: c["min_crop_dist"])

    weed_np = (labels == target["lbl"]).astype(np.uint8)
    gx, gy = target["centroid"]

    best_start = None
    best_score = -1e9
    margin_pad = int(math.ceil(max(half_l, half_w))) + 2
    coords = np.linspace(margin_pad, grid_size - 1 - margin_pad, 13, dtype=np.float32)
    for sy in coords:
        for sx in coords:
            wx = sx + nozzle_pts_np[:, 0]
            wy = sy + nozzle_pts_np[:, 1]
            c_min = float(bilinear_sample_np(crop_sdf_np, wx, wy).min())
            if c_min < 2.5:
                continue
            dist_to_goal = math.hypot(sx - gx, sy - gy)
            if dist_to_goal < 28.0:
                continue
            score = min(dist_to_goal, 85.0) + 0.5 * min(c_min, 10.0)
            if score > best_score:
                best_score = score
                best_start = (float(sx), float(sy))

    if best_start is None:
        return None, "no_safe_start"

    return {
        "name": sem_path.name,
        "crop_np": crop_np,
        "weed_np": weed_np,
        "crop_sdf_np": crop_sdf_np,
        "weed_pts_np": target["pts"],
        "start_xy": best_start,
        "goal_xy": (gx, gy),
        "pca_th": target["pca_th"],
        "min_crop_dist": target["min_crop_dist"],
    }, "ok"


# ------------------------------------------------------------------------------
# Zero-Padded Realistic Perception Noise (U-Net/Camera Boundary Jitter)
# ------------------------------------------------------------------------------
def shift_mask_zero_pad(mask_np: np.ndarray, dx: int, dy: int) -> np.ndarray:
    H, W = mask_np.shape
    out = np.zeros_like(mask_np)
    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = mask_np[src_y0:src_y1, src_x0:src_x1]
    return out


def corrupt_mask_realistic(mask_np: np.ndarray, rng: np.random.RandomState, min_pixels: int = 4) -> np.ndarray:
    """±2px 경계 이동(제로패딩) + 3x3 팽창/침식/유지 랜덤 교란."""
    dx = int(rng.randint(-2, 3))
    dy = int(rng.randint(-2, 3))
    shifted = shift_mask_zero_pad(mask_np, dx, dy)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mode = rng.choice(["dilate", "erode", "none"])
    if mode == "dilate":
        out = cv2.dilate(shifted, kernel, iterations=1)
    elif mode == "erode":
        out = cv2.erode(shifted, kernel, iterations=1)
    else:
        out = shifted
    if out.sum() < min_pixels:
        out = shifted if shifted.sum() >= min_pixels else mask_np.copy()
    return out


# ------------------------------------------------------------------------------
# Unified Strict Evaluator (+ Kinodynamic Angular Velocity Check)
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_trajectory(xy, th, crop_sdf, weed_pts, nozzle_pts, half_l, half_w, omega_max=None):
    H, W = crop_sdf.shape
    cos_t = torch.cos(th).unsqueeze(-1)
    sin_t = torch.sin(th).unsqueeze(-1)

    wx = xy[:, 0:1] + cos_t * nozzle_pts[:, 0].unsqueeze(0) - sin_t * nozzle_pts[:, 1].unsqueeze(0)
    wy = xy[:, 1:2] + sin_t * nozzle_pts[:, 0].unsqueeze(0) + cos_t * nozzle_pts[:, 1].unsqueeze(0)
    in_bounds = (wx >= 0.0) & (wx <= W - 1.0) & (wy >= 0.0) & (wy <= H - 1.0)
    crop_d = sample_sdf_batched(crop_sdf, torch.stack([wx, wy], dim=-1).unsqueeze(0))[0]

    safe_per_step = ((crop_d > 0.0) & in_bounds).all(dim=1)
    crop_safe_all = bool(safe_per_step.all().item())
    safe_step_ratio = float(safe_per_step.float().mean().item())

    dx = weed_pts[:, 0].unsqueeze(0) - xy[:, 0:1]
    dy = weed_pts[:, 1].unsqueeze(0) - xy[:, 1:2]
    wlx = cos_t * dx + sin_t * dy
    wly = -sin_t * dx + cos_t * dy
    w_sd = nozzle_analytical_sdf_torch(torch.stack([wlx, wly], dim=-1), half_l, half_w)

    ntppi_per_step = (w_sd < 0.0).all(dim=1) & safe_per_step
    weed_ntppi_ok = bool(ntppi_per_step.any().item())
    best_cov = float((w_sd < 0.0).float().mean(dim=1).max().item())

    dxy = xy[1:] - xy[:-1]
    dth = wrap_angle_torch(th[1:] - th[:-1])
    max_abs_dth = float(torch.abs(dth).max().item())
    kino_ok = True if omega_max is None else (max_abs_dth <= omega_max + 1e-4)

    return {
        "full_stl": int(crop_safe_all and weed_ntppi_ok and kino_ok),
        "stl_geom_only": int(crop_safe_all and weed_ntppi_ok),
        "kino_ok": int(kino_ok),
        "max_dth": max_abs_dth,
        "zero_coll": int(crop_safe_all),
        "crop_safety": safe_step_ratio,
        "weed_ntppi": int(weed_ntppi_ok),
        "weed_cov": best_cov,
        "path_len": float(torch.norm(dxy, dim=-1).sum().item()),
        "ang_tv": float(torch.abs(dth).sum().item()),
    }


# ------------------------------------------------------------------------------
# Experiment 1: Standard SE(2) MPPI Controller (Gradient-Free Sampling Baseline)
# ------------------------------------------------------------------------------
@torch.no_grad()
def plan_mppi_se2(
    crop_sdf: torch.Tensor,
    weed_pts: torch.Tensor,
    nozzle_pts: torch.Tensor,
    start_xy,
    goal_xy,
    pca_th: float,
    half_l: float,
    half_w: float,
    T: int = 16,
    K: int = 384,
    iters: int = 45,
    crop_margin: float = 1.8,
    seed: int = 42,
):
    """
    Modern SE(2) MPPI with temporally correlated + white exploration rollouts,
    optimizing the EXACT SAME crop DC + weed NTPPi cost without gradients.
    """
    device = crop_sdf.device
    H, W = crop_sdf.shape
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    p0 = torch.tensor(start_xy, device=device, dtype=torch.float32)
    p1 = torch.tensor(goal_xy, device=device, dtype=torch.float32)

    # 가장 가까운 PCA 정렬 각도 선택
    cand_a = pca_th
    cand_b = wrap_angle_np(pca_th + math.pi)
    target_th = cand_a if abs(cand_a) <= abs(cand_b) else cand_b

    # 명목 제어 시퀀스 U: (T-1, 3) = [vx, vy, omega]
    U = torch.zeros((T - 1, 3), device=device, dtype=torch.float32)
    U[:, :2] = (p1 - p0) / float(T - 1)
    U[:, 2] = target_th / float(T - 1)

    lx = nozzle_pts[:, 0].view(1, 1, -1)
    ly = nozzle_pts[:, 1].view(1, 1, -1)
    w_px = weed_pts[:, 0].view(1, 1, -1)
    w_py = weed_pts[:, 1].view(1, 1, -1)

    start_xy_k = p0.view(1, 1, 2).expand(K, 1, 2)
    start_th_k = torch.zeros((K, 1), device=device, dtype=torch.float32)

    # 3-tap 시간 평활 커널 (부드러운 우회 곡선 탐색용 Colored Noise)
    smooth_k = torch.tensor([0.25, 0.5, 0.25], device=device).view(1, 1, 3)

    best_xy = None
    best_th = None
    best_rank = (-1, -1, -1e9)

    for it in range(iters):
        prog = it / max(iters - 1, 1)
        sig_xy = 4.2 * (1.0 - 0.72 * prog)
        sig_th = 0.36 * (1.0 - 0.75 * prog)

        eps = torch.randn((K, T - 1, 3), generator=gen, device=device)
        # 절반은 시간 상관(Colored) 노이즈로 변환하여 장애물 우회 호모토피 탐색 강화
        half_k = K // 2
        c_noise = eps[:half_k].permute(0, 2, 1).reshape(half_k * 3, 1, T - 1)
        for _ in range(3):
            c_noise = F.conv1d(F.pad(c_noise, (1, 1), mode="replicate"), smooth_k)
        eps[:half_k] = c_noise.reshape(half_k, 3, T - 1).permute(0, 2, 1) * 1.9

        eps[..., :2] *= sig_xy
        eps[..., 2] *= sig_th
        # 0번 샘플은 현재 명목 시퀀스 U 그대로 유지
        eps[0].zero_()

        U_cand = U.unsqueeze(0) + eps
        U_cand[..., :2].clamp_(-16.0, 16.0)
        U_cand[..., 2].clamp_(-0.75, 0.75)

        xy_steps = start_xy_k + torch.cumsum(U_cand[..., :2], dim=1)
        xy_all = torch.cat([start_xy_k, xy_steps], dim=1)  # (K, T, 2)
        th_all = torch.cat([start_th_k, torch.cumsum(U_cand[..., 2], dim=1)], dim=1)  # (K, T)

        cos_t = torch.cos(th_all).unsqueeze(-1)
        sin_t = torch.sin(th_all).unsqueeze(-1)

        wx = xy_all[:, :, 0:1] + cos_t * lx - sin_t * ly
        wy = xy_all[:, :, 1:2] + sin_t * lx + cos_t * ly
        in_b = (wx >= 0.0) & (wx <= W - 1.0) & (wy >= 0.0) & (wy <= H - 1.0)
        crop_d = sample_sdf_batched(crop_sdf, torch.stack([wx, wy], dim=-1))

        # 1. 작물 비충돌 + 경계 비용
        viol_margin = F.relu(crop_margin - crop_d)
        hard_coll = ((crop_d <= 0.0) | (~in_b)).float()
        cost_crop = (
            (viol_margin ** 2).mean(dim=(1, 2)) * 18.0
            + viol_margin.amax(dim=(1, 2)) * 12.0
            + hard_coll.sum(dim=(1, 2)) * 25.0
        )

        # 2. 잡초 NTPPi 포섭 비용 (동일 목적함수)
        dx = w_px - xy_all[:, :, 0:1]
        dy = w_py - xy_all[:, :, 1:2]
        wlx = cos_t * dx + sin_t * dy
        wly = -sin_t * dx + cos_t * dy
        w_sd = nozzle_analytical_sdf_torch(torch.stack([wlx, wly], dim=-1), half_l, half_w)  # (K, T, M)

        step_max_w = w_sd.amax(dim=2)  # (K, T)
        best_step_w = step_max_w.amin(dim=1)  # (K,)
        cost_weed = (
            F.relu(best_step_w + 0.8) * 14.0
            + F.relu(step_max_w[:, -1] + 0.8) * 6.0
            + F.relu(w_sd[:, -1, :] + 0.3).mean(dim=1) * 10.0
        )

        # 3. 제어 매끄러움 비용
        cost_ctrl = (U_cand[..., :2] ** 2).mean(dim=(1, 2)) * 0.08 + (U_cand[..., 2] ** 2).mean(dim=1) * 3.0
        J = cost_crop + cost_weed + cost_ctrl  # (K,)

        # 엘리트 후보 추적 (MPPI 평균이 장애물 양쪽으로 갈라져 중앙 충돌하는 현상 방지)
        safe_steps = ((crop_d > 0.0) & in_b).all(dim=2)  # (K, T)
        zero_coll = safe_steps.all(dim=1)  # (K,)
        ntppi_ok = ((step_max_w < 0.0) & safe_steps).any(dim=1)  # (K,)
        full_ok = zero_coll & ntppi_ok

        if full_ok.any():
            idxs = torch.nonzero(full_ok, as_tuple=False).flatten()
            j_sub = J[idxs]
            bi = int(idxs[torch.argmin(j_sub)].item())
            rank = (2, 1, -float(J[bi].item()))
            if rank > best_rank:
                best_rank = rank
                best_xy = xy_all[bi].clone()
                best_th = th_all[bi].clone()
        elif zero_coll.any() and best_rank[0] < 2:
            idxs = torch.nonzero(zero_coll, as_tuple=False).flatten()
            j_sub = J[idxs]
            bi = int(idxs[torch.argmin(j_sub)].item())
            rank = (1, 1, -float(J[bi].item()))
            if rank > best_rank:
                best_rank = rank
                best_xy = xy_all[bi].clone()
                best_th = th_all[bi].clone()
        elif best_rank[0] < 1:
            bi = int(torch.argmin(J).item())
            rank = (0, 0, -float(J[bi].item()))
            if rank > best_rank:
                best_rank = rank
                best_xy = xy_all[bi].clone()
                best_th = th_all[bi].clone()

        # MPPI Softmax 중요도 가중 업데이트
        j_min = J.min()
        temp = torch.clamp(J.std() * 0.25, min=1.0)
        weights = torch.softmax(-(J - j_min) / temp, dim=0)
        U = (weights[:, None, None] * U_cand).sum(dim=0)

    # 최종 명목 궤적 U도 함께 비교해 더 우수한 쪽 선택
    xy_nom = torch.cat([p0.view(1, 2), p0.view(1, 2) + torch.cumsum(U[:, :2], dim=0)], dim=0)
    th_nom = torch.cat([torch.zeros(1, device=device), torch.cumsum(U[:, 2], dim=0)], dim=0)
    ev_nom = evaluate_trajectory(xy_nom, th_nom, crop_sdf, weed_pts, nozzle_pts, half_l, half_w)
    ev_best = evaluate_trajectory(best_xy, best_th, crop_sdf, weed_pts, nozzle_pts, half_l, half_w)

    if (ev_nom["full_stl"], ev_nom["zero_coll"], ev_nom["crop_safety"], ev_nom["weed_cov"]) > (
        ev_best["full_stl"], ev_best["zero_coll"], ev_best["crop_safety"], ev_best["weed_cov"]
    ):
        return xy_nom.detach(), th_nom.detach()
    return best_xy.detach(), best_th.detach()


# ------------------------------------------------------------------------------
# Experiment 2: Kinodynamic-Aware RRT* + Noise-Margin Hybrid SDF-RCC8
# ------------------------------------------------------------------------------
def is_pose_collision_free_np(x, y, th, crop_sdf_np, nozzle_pts_np, margin=0.2):
    H, W = crop_sdf_np.shape
    c, s = math.cos(th), math.sin(th)
    wx = x + c * nozzle_pts_np[:, 0] - s * nozzle_pts_np[:, 1]
    wy = y + s * nozzle_pts_np[:, 0] + c * nozzle_pts_np[:, 1]
    if wx.min() < 0.0 or wx.max() > (W - 1.0) or wy.min() < 0.0 or wy.max() > (H - 1.0):
        return False
    return bool(bilinear_sample_np(crop_sdf_np, wx, wy).min() > margin)


def is_edge_collision_free_np(q1, q2, crop_sdf_np, nozzle_pts_np, half_l, margin=0.2, start_margin=None):
    dx = q2[0] - q1[0]
    dy = q2[1] - q1[1]
    dth = wrap_angle_np(q2[2] - q1[2])
    dist = math.sqrt(dx * dx + dy * dy + (half_l * dth) ** 2)
    n_steps = max(2, int(math.ceil(dist / 3.0)))
    ts = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)[1:]
    xs = q1[0] + ts * dx
    ys = q1[1] + ts * dy
    ths = q1[2] + ts * dth
    cs = np.cos(ths)[:, None]
    ss = np.sin(ths)[:, None]
    wx = xs[:, None] + cs * nozzle_pts_np[None, :, 0] - ss * nozzle_pts_np[None, :, 1]
    wy = ys[:, None] + ss * nozzle_pts_np[None, :, 0] + cs * nozzle_pts_np[None, :, 1]
    H, W = crop_sdf_np.shape
    if wx.min() < 0.0 or wx.max() > (W - 1.0) or wy.min() < 0.0 or wy.max() > (H - 1.0):
        return False
    cds = bilinear_sample_np(crop_sdf_np, wx.reshape(-1), wy.reshape(-1)).reshape(len(ts), -1).min(axis=1)
    if start_margin is not None and start_margin < margin:
        # 출발점이 마진보다 가깝게 시작할 때만 첫 엣지에서 점진적으로 마진 확보
        req = start_margin + ts * (margin - start_margin)
        return bool((cds > req).all())
    return bool(cds.min() > margin)


def plan_rrt_star_se2_margin(
    crop_sdf_np, weed_pts_np, start_xy, goal_xy, pca_th, nozzle_pts_np, half_l, half_w,
    T=16, max_iters=500, rrt_margin=0.2, omega_max=None, seed=42
):
    rng = np.random.RandomState(seed)
    H, W = crop_sdf_np.shape
    w_th = half_l
    step_size = 11.0
    rewire_r = 20.0

    # 출발점 실제 여유 거리 확인 (노이즈 환경에서 큰 마진일 때 출발점 이탈 허용)
    sx0, sy0 = start_xy
    start_c = float(bilinear_sample_np(crop_sdf_np, sx0 + nozzle_pts_np[:, 0], sy0 + nozzle_pts_np[:, 1]).min())
    start_m = min(rrt_margin, start_c - 0.15)

    nodes = np.zeros((max_iters + 1, 3), dtype=np.float32)
    parents = np.full(max_iters + 1, -1, dtype=np.int32)
    costs = np.zeros(max_iters + 1, dtype=np.float32)
    nodes[0] = [sx0, sy0, 0.0]
    n_nodes = 1

    best_goal_idx = -1
    best_goal_cost = 1e9
    iters_after_goal = 0

    for _ in range(max_iters):
        if best_goal_idx >= 0:
            iters_after_goal += 1
            if iters_after_goal >= 100:
                break

        r = rng.rand()
        if r < 0.12:
            sx = goal_xy[0] + rng.uniform(-3.0, 3.0)
            sy = goal_xy[1] + rng.uniform(-3.0, 3.0)
            sth = wrap_angle_np(pca_th + rng.uniform(-0.25, 0.25) + (math.pi if rng.rand() < 0.5 else 0.0))
        elif r < 0.22:
            sx = goal_xy[0] + rng.uniform(-10.0, 10.0)
            sy = goal_xy[1] + rng.uniform(-10.0, 10.0)
            sth = rng.uniform(-math.pi, math.pi)
        else:
            sx = rng.uniform(2.0, W - 3.0)
            sy = rng.uniform(2.0, H - 3.0)
            sth = rng.uniform(-math.pi, math.pi)

        active = nodes[:n_nodes]
        dx = active[:, 0] - sx
        dy = active[:, 1] - sy
        dth = (active[:, 2] - sth + math.pi) % (2.0 * math.pi) - math.pi
        dists = np.sqrt(dx * dx + dy * dy + (w_th * dth) ** 2)
        near_idx = int(np.argmin(dists))
        q_near = active[near_idx]
        d_near = float(dists[near_idx])
        if d_near < 1e-5:
            continue

        scale = min(1.0, step_size / d_near)
        raw_dth = scale * wrap_angle_np(sth - float(q_near[2]))
        if omega_max is not None:
            raw_dth = max(-omega_max * 0.9, min(omega_max * 0.9, raw_dth))
        q_new = (
            float(np.clip(q_near[0] + scale * (sx - q_near[0]), 1.0, W - 2.0)),
            float(np.clip(q_near[1] + scale * (sy - q_near[1]), 1.0, H - 2.0)),
            wrap_angle_np(float(q_near[2]) + raw_dth),
        )

        if not is_pose_collision_free_np(q_new[0], q_new[1], q_new[2], crop_sdf_np, nozzle_pts_np, margin=rrt_margin):
            continue
        sm = start_m if near_idx == 0 else None
        if not is_edge_collision_free_np(q_near, q_new, crop_sdf_np, nozzle_pts_np, half_l, margin=rrt_margin, start_margin=sm):
            continue

        dx_n = active[:, 0] - q_new[0]
        dy_n = active[:, 1] - q_new[1]
        dth_n = (active[:, 2] - q_new[2] + math.pi) % (2.0 * math.pi) - math.pi
        d_all = np.sqrt(dx_n * dx_n + dy_n * dy_n + (w_th * dth_n) ** 2)
        nbrs = np.nonzero(d_all <= rewire_r)[0]

        best_p = near_idx
        best_c = float(costs[near_idx] + d_all[near_idx])
        for nb in nbrs:
            nb = int(nb)
            if omega_max is not None and abs(wrap_angle_np(q_new[2] - float(active[nb, 2]))) > omega_max * 1.4:
                continue
            cand_c = float(costs[nb] + d_all[nb])
            sm_nb = start_m if nb == 0 else None
            if cand_c < best_c and is_edge_collision_free_np(active[nb], q_new, crop_sdf_np, nozzle_pts_np, half_l, margin=rrt_margin, start_margin=sm_nb):
                best_p = nb
                best_c = cand_c

        new_idx = n_nodes
        nodes[new_idx] = q_new
        parents[new_idx] = best_p
        costs[new_idx] = best_c
        n_nodes += 1

        for nb in nbrs:
            nb = int(nb)
            if nb == best_p or nb == 0:
                continue
            if omega_max is not None and abs(wrap_angle_np(float(active[nb, 2]) - q_new[2])) > omega_max * 1.4:
                continue
            cand_c = best_c + float(d_all[nb])
            if cand_c + 1e-4 < costs[nb] and is_edge_collision_free_np(q_new, active[nb], crop_sdf_np, nozzle_pts_np, half_l, margin=rrt_margin):
                parents[nb] = new_idx
                costs[nb] = cand_c

        w_sd = nozzle_analytical_sdf_np(weed_pts_np, q_new[0], q_new[1], q_new[2], half_l, half_w)
        if w_sd.max() < 0.0 and best_c < best_goal_cost:
            best_goal_idx = new_idx
            best_goal_cost = best_c

    if best_goal_idx >= 0:
        end_idx = best_goal_idx
    else:
        active = nodes[:n_nodes]
        d_goal = np.hypot(active[:, 0] - goal_xy[0], active[:, 1] - goal_xy[1])
        end_idx = int(np.argmin(d_goal))

    chain = []
    cur = end_idx
    while cur >= 0:
        chain.append(nodes[cur])
        cur = int(parents[cur])
    chain.reverse()
    chain = np.array(chain, dtype=np.float32)

    if len(chain) == 1:
        return np.repeat(chain[:, :2], T, axis=0), np.repeat(chain[:, 2], T, axis=0)

    seg_d = np.sqrt(
        (chain[1:, 0] - chain[:-1, 0]) ** 2
        + (chain[1:, 1] - chain[:-1, 1]) ** 2
        + (w_th * ((chain[1:, 2] - chain[:-1, 2] + math.pi) % (2.0 * math.pi) - math.pi)) ** 2
    )
    cum = np.concatenate([[0.0], np.cumsum(seg_d)])
    total = float(cum[-1])
    if total < 1e-6:
        return np.repeat(chain[:1, :2], T, axis=0), np.repeat(chain[:1, 2], T, axis=0)

    targets = np.linspace(0.0, total, T, dtype=np.float32)
    xy_out = np.zeros((T, 2), dtype=np.float32)
    th_out = np.zeros((T,), dtype=np.float32)
    for i, s_val in enumerate(targets):
        j = max(0, min(int(np.searchsorted(cum, s_val, side="right") - 1), len(chain) - 2))
        denom = max(float(cum[j + 1] - cum[j]), 1e-6)
        u = (s_val - cum[j]) / denom
        xy_out[i, 0] = chain[j, 0] + u * (chain[j + 1, 0] - chain[j, 0])
        xy_out[i, 1] = chain[j, 1] + u * (chain[j + 1, 1] - chain[j, 1])
        th_out[i] = wrap_angle_np(float(chain[j, 2]) + u * wrap_angle_np(float(chain[j + 1, 2] - chain[j, 2])))
    return xy_out, th_out


def project_angles_omega_max(th: torch.Tensor, omega_max: float) -> torch.Tensor:
    """각도 궤적을 |dth| <= omega_max 로 정확히 투영."""
    dth = wrap_angle_torch(th[..., 1:] - th[..., :-1]).clamp(-omega_max, omega_max)
    z = torch.zeros_like(th[..., :1])
    return torch.cat([z, torch.cumsum(dth, dim=-1)], dim=-1)


def refine_with_sdf_rcc8_kinodynamic(
    rrt_xy_np: np.ndarray,
    rrt_th_np: np.ndarray,
    crop_sdf: torch.Tensor,
    weed_pts: torch.Tensor,
    nozzle_pts: torch.Tensor,
    start_xy,
    goal_xy,
    pca_th: float,
    half_l: float,
    half_w: float,
    crop_margin: float = 2.0,
    omega_max: float = 0.35,
    max_iters: int = 100,
    check_every: int = 10,
):
    """
    각속도 재매개변수화(dth = omega_max * tanh(u))로 물리적 회전 속도 한계를
    구조적으로 100% 보장하는 Stage 2 SDF-RCC8 정밀 보정기.
    """
    device = crop_sdf.device
    H, W = crop_sdf.shape
    T = rrt_xy_np.shape[0]

    rrt_xy = torch.from_numpy(rrt_xy_np).to(device)
    rrt_th = project_angles_omega_max(torch.from_numpy(rrt_th_np).to(device), omega_max)
    p0 = torch.tensor(start_xy, device=device, dtype=torch.float32)
    p1 = torch.tensor(goal_xy, device=device, dtype=torch.float32)

    end_th = float(rrt_th[-1].item())
    cand_a = pca_th
    cand_b = wrap_angle_np(pca_th + math.pi)
    if abs(wrap_angle_np(cand_a - end_th)) <= abs(wrap_angle_np(cand_b - end_th)):
        th_near, th_far = cand_a, cand_b
    else:
        th_near, th_far = cand_b, cand_a

    tail_w = torch.zeros(T, device=device)
    tail_w[-5:] = torch.tensor([0.15, 0.35, 0.55, 0.80, 1.0], device=device)
    xy_b0 = rrt_xy * (1.0 - tail_w.unsqueeze(-1)) + p1.unsqueeze(0) * tail_w.unsqueeze(-1)
    th_b0 = project_angles_omega_max(rrt_th + tail_w * wrap_angle_np(th_near - end_th), omega_max)
    th_b1 = project_angles_omega_max(rrt_th + tail_w * wrap_angle_np(th_far - end_th), omega_max)

    t_lin = torch.linspace(0.0, 1.0, T, device=device).unsqueeze(-1)
    base_xy = p0 * (1.0 - t_lin) + p1 * t_lin
    d_vec = p1 - p0
    dist = torch.norm(d_vec).clamp(min=1e-6)
    normal = torch.stack([-d_vec[1] / dist, d_vec[0] / dist])
    arch = torch.sin(math.pi * t_lin)

    xy_b2 = base_xy - 24.0 * arch * normal.unsqueeze(0)
    xy_b3 = base_xy + 24.0 * arch * normal.unsqueeze(0)
    th_arc = project_angles_omega_max(torch.linspace(0.0, wrap_angle_np(th_near), T, device=device), omega_max)

    init_xy = torch.stack([xy_b0, xy_b0.clone(), xy_b2, xy_b3], dim=0)  # (4, T, 2)
    init_th = torch.stack([th_b0, th_b1, th_arc, th_arc.clone()], dim=0)  # (4, T)
    init_xy[..., 0].clamp_(1.0, W - 2.0)
    init_xy[..., 1].clamp_(1.0, H - 2.0)

    # 각속도 파라미터 초기화: u = atanh(clip(dth / omega_max, -0.92, 0.92))
    init_dth = wrap_angle_torch(init_th[:, 1:] - init_th[:, :-1])
    init_u = torch.atanh((init_dth / omega_max).clamp(-0.92, 0.92))

    B = 4
    xy_var = init_xy[:, 1:, :].clone().detach().requires_grad_(True)
    u_var = init_u.clone().detach().requires_grad_(True)

    lr_xy_init, lr_u_init = 1.3, 0.25
    opt = torch.optim.Adam([
        {"params": [xy_var], "lr": lr_xy_init},
        {"params": [u_var], "lr": lr_u_init},
    ])

    start_xy_t = p0.view(1, 1, 2).expand(B, 1, 2)
    start_th_t = torch.zeros(B, 1, device=device)
    lx = nozzle_pts[:, 0].view(1, 1, -1)
    ly = nozzle_pts[:, 1].view(1, 1, -1)
    w_px = weed_pts[:, 0].view(1, 1, -1)
    w_py = weed_pts[:, 1].view(1, 1, -1)

    # 마진 조기종료 기준 (노이즈 환경에서는 입력 SDF 상에서 최소 안전 마진을 확보해야 GT에서 안전)
    req_margin_check = max(0.0, crop_margin * 0.75)

    best_feasible_xy = None
    best_feasible_th = None
    best_feasible_ang = 1e9

    for it in range(1, max_iters + 1):
        progress = (it - 1) / max(max_iters - 1, 1)
        lr_scale = 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress))
        opt.param_groups[0]["lr"] = lr_xy_init * lr_scale
        opt.param_groups[1]["lr"] = lr_u_init * lr_scale

        eff_margin = crop_margin + 1.5 * (1.0 - progress)
        dc_weight = 7.0 + 7.0 * progress

        opt.zero_grad(set_to_none=True)
        xy = torch.cat([start_xy_t, xy_var], dim=1)
        dth = omega_max * torch.tanh(u_var)  # 엄격히 |dth| < omega_max 보장!
        th = torch.cat([start_th_t, torch.cumsum(dth, dim=1)], dim=1)

        cos_t = torch.cos(th).unsqueeze(-1)
        sin_t = torch.sin(th).unsqueeze(-1)

        wx = xy[:, :, 0:1] + cos_t * lx - sin_t * ly
        wy = xy[:, :, 1:2] + sin_t * lx + cos_t * ly
        crop_d = sample_sdf_batched(crop_sdf, torch.stack([wx, wy], dim=-1))

        # 출발점 t=0은 고정되므로 t>=1 중심으로 강한 마진 적용
        viol_dc = F.relu(eff_margin - crop_d[:, 1:, :])
        loss_dc = (viol_dc ** 2).mean(dim=(1, 2)) * dc_weight + viol_dc.amax(dim=(1, 2)) * (dc_weight * 0.75)

        loss_bound = (
            F.relu(-wx).mean(dim=(1, 2))
            + F.relu(wx - (W - 1.0)).mean(dim=(1, 2))
            + F.relu(-wy).mean(dim=(1, 2))
            + F.relu(wy - (H - 1.0)).mean(dim=(1, 2))
        ) * 8.0

        dx = w_px - xy[:, :, 0:1]
        dy = w_py - xy[:, :, 1:2]
        wlx = cos_t * dx + sin_t * dy
        wly = -sin_t * dx + cos_t * dy
        w_sd = nozzle_analytical_sdf_torch(torch.stack([wlx, wly], dim=-1), half_l, half_w)

        step_worst = (1.0 / 2.5) * torch.logsumexp(2.5 * w_sd, dim=2)
        ev_ntppi = -(1.0 / 2.0) * torch.logsumexp(-2.0 * step_worst, dim=1)
        loss_target = (
            F.relu(ev_ntppi + 0.8) * 2.2
            + F.relu(step_worst[:, -1] + 0.8) * 1.2
            + F.relu(w_sd[:, -1, :] + 0.4).mean(dim=1) * 1.5
        )

        loss_smooth = (
            ((xy[:, 1:] - xy[:, :-1]) ** 2).mean(dim=(1, 2)) * 0.04
            + (dth ** 2).mean(dim=1) * 0.55
        )

        (loss_dc + loss_bound + loss_target + loss_smooth).sum().backward()
        opt.step()

        if it % check_every == 0 or it == max_iters:
            with torch.no_grad():
                xy_c = torch.cat([start_xy_t, xy_var], dim=1)
                dth_c = omega_max * torch.tanh(u_var)
                th_c = torch.cat([start_th_t, torch.cumsum(dth_c, dim=1)], dim=1)
                c_t = torch.cos(th_c).unsqueeze(-1)
                s_t = torch.sin(th_c).unsqueeze(-1)
                wx_c = xy_c[:, :, 0:1] + c_t * lx - s_t * ly
                wy_c = xy_c[:, :, 1:2] + s_t * lx + c_t * ly
                in_b = (wx_c >= 0.0) & (wx_c <= W - 1.0) & (wy_c >= 0.0) & (wy_c <= H - 1.0)
                cd_c = sample_sdf_batched(crop_sdf, torch.stack([wx_c, wy_c], dim=-1))
                # t>=1에서 요구 안전 마진 확보 여부 확인 (입력 노이즈 SDF 기준, GT 누수 0%)
                safe_t0 = ((cd_c[:, :1, :] > 0.0) & in_b[:, :1, :]).all(dim=2)
                safe_rest = ((cd_c[:, 1:, :] > req_margin_check) & in_b[:, 1:, :]).all(dim=2)
                safe_s = torch.cat([safe_t0, safe_rest], dim=1)
                zero_coll = safe_s.all(dim=1)

                if zero_coll.any():
                    dx_c = w_px - xy_c[:, :, 0:1]
                    dy_c = w_py - xy_c[:, :, 1:2]
                    wlx_c = c_t * dx_c + s_t * dy_c
                    wly_c = -s_t * dx_c + c_t * dy_c
                    wsd_c = nozzle_analytical_sdf_torch(torch.stack([wlx_c, wly_c], dim=-1), half_l, half_w)
                    ntppi_s = (wsd_c < 0.0).all(dim=2) & safe_s
                    full_ok = zero_coll & ntppi_s.any(dim=1)
                    if full_ok.any():
                        for b in torch.nonzero(full_ok, as_tuple=False).flatten().tolist():
                            ang = float(torch.abs(dth_c[b]).sum().item())
                            if ang < best_feasible_ang:
                                best_feasible_ang = ang
                                best_feasible_xy = xy_c[b].detach().clone()
                                best_feasible_th = th_c[b].detach().clone()
                        break

    if best_feasible_xy is not None:
        return best_feasible_xy, best_feasible_th

    # Fail-Safe Fallback (입력 SDF 기준 가장 안전하며 omega_max 100% 준수하는 경로 선택)
    with torch.no_grad():
        xy_c = torch.cat([start_xy_t, xy_var], dim=1)
        dth_c = omega_max * torch.tanh(u_var)
        th_c = torch.cat([start_th_t, torch.cumsum(dth_c, dim=1)], dim=1)
        cands = [(rrt_xy, rrt_th)] + [(xy_c[b].detach(), th_c[b].detach()) for b in range(B)]
        best_pair = (rrt_xy, rrt_th)
        best_key = (-1, -1.0, -1e9)
        for cand_xy, cand_th in cands:
            c_t = torch.cos(cand_th).unsqueeze(-1)
            s_t = torch.sin(cand_th).unsqueeze(-1)
            wx_f = cand_xy[:, 0:1] + c_t * nozzle_pts[:, 0].unsqueeze(0) - s_t * nozzle_pts[:, 1].unsqueeze(0)
            wy_f = cand_xy[:, 1:2] + s_t * nozzle_pts[:, 0].unsqueeze(0) + c_t * nozzle_pts[:, 1].unsqueeze(0)
            cd_f = sample_sdf_batched(crop_sdf, torch.stack([wx_f, wy_f], dim=-1).unsqueeze(0))[0]
            min_cd = float(cd_f[1:].amin().item())
            ev = evaluate_trajectory(cand_xy, cand_th, crop_sdf, weed_pts, nozzle_pts, half_l, half_w, omega_max=omega_max)
            key = (int(ev["zero_coll"] and min_cd >= req_margin_check), ev["zero_coll"], ev["crop_safety"], min(min_cd, crop_margin), ev["weed_cov"])
            if key > best_key:
                best_key = key
                best_pair = (cand_xy, cand_th)
    return best_pair[0], best_pair[1]


# ------------------------------------------------------------------------------
# Stats & Printing
# ------------------------------------------------------------------------------
def mcnemar_stats(a_flags, b_flags):
    only_a = sum(1 for a, b in zip(a_flags, b_flags) if a == 1 and b == 0)
    only_b = sum(1 for a, b in zip(a_flags, b_flags) if a == 0 and b == 1)
    denom = only_a + only_b
    chi2 = ((abs(only_a - only_b) - 1.0) ** 2) / denom if denom > 0 else 0.0
    p_val = math.erfc(math.sqrt(chi2 / 2.0))
    return only_a, only_b, chi2, p_val


def summarize_method(rows_by_scene, scene_names, method_key):
    sub = [rows_by_scene[s][method_key] for s in scene_names if method_key in rows_by_scene[s]]
    n = len(sub)
    if n == 0:
        return None
    return {
        "n": n,
        "full_stl_cnt": sum(int(r["full_stl"]) for r in sub),
        "full_stl_pct": 100.0 * np.mean([float(r["full_stl"]) for r in sub]),
        "zero_coll_pct": 100.0 * np.mean([float(r["zero_coll"]) for r in sub]),
        "crop_safety_pct": 100.0 * np.mean([float(r["crop_safety"]) for r in sub]),
        "weed_cov_pct": 100.0 * np.mean([float(r["weed_cov"]) for r in sub]),
        "ang_tv": float(np.mean([float(r["ang_tv"]) for r in sub])),
        "latency_ms": float(np.mean([float(r["latency_ms"]) for r in sub])),
    }


def print_main_table(title, rows_by_scene, scene_names, methods):
    print(f"\n{title} (N={len(scene_names)})")
    print("-" * 122)
    print(f"{'Method':<38} | {'Full STL':>14} | {'Zero-Coll':>9} | {'CropSafe':>8} | {'WeedCov':>7} | {'AngTV':>6} | {'Latency':>9}")
    print("-" * 122)
    out = {}
    for k, label in methods:
        s = summarize_method(rows_by_scene, scene_names, k)
        if s is None:
            continue
        out[k] = s
        print(
            f"{label:<38} | {s['full_stl_cnt']:4d}/{s['n']:<4d} ({s['full_stl_pct']:5.1f}%) "
            f"| {s['zero_coll_pct']:8.1f}% | {s['crop_safety_pct']:7.1f}% "
            f"| {s['weed_cov_pct']:6.1f}% | {s['ang_tv']:6.2f} | {s['latency_ms']:6.1f} ms"
        )
    print("-" * 122)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--half_l", type=float, default=14.5)
    parser.add_argument("--half_w", type=float, default=7.0)
    parser.add_argument("--omega_max", type=float, default=0.35, help="Max angular velocity rad/step (~20 deg/step)")
    parser.add_argument("--mppi_k", type=int, default=384)
    parser.add_argument("--mppi_iters", type=int, default=45)
    parser.add_argument("--max_scenes", type=int, default=0)
    args = parser.parse_args()

    print("=== [Part 0] Hardware & Dataset Initialization ===")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print(f"* Python: {sys.executable} | PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} ({gpu_name})")

    sem_dir = Path(args.val_dir) / "semantics"
    sem_files = sorted(sem_dir.glob("*.png"))
    nozzle_pts_np = get_nozzle_local_points_np(args.half_l, args.half_w)

    scenes = []
    for p in sem_files:
        sc, status = extract_phenobench_scene(p, args.grid_size, args.half_l, args.half_w, nozzle_pts_np)
        if status == "ok":
            scenes.append(sc)
            if args.max_scenes > 0 and len(scenes) >= args.max_scenes:
                break
    all_names = [sc["name"] for sc in scenes]
    print(f"* Loaded {len(scenes)} Real PhenoBench Val Scenes.")

    # 기존 CSV(5종 + static_feasible 메타) 로드
    rows_by_scene = defaultdict(dict)
    feas_map = {}
    for fname in ["phenobench_full_benchmark_per_scene.csv", "phenobench_hybrid_per_scene.csv"]:
        fp = Path(fname)
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows_by_scene[row["scene"]][row["method"]] = row
                    if "static_feasible" in row:
                        feas_map[row["scene"]] = int(row["static_feasible"])
            print(f"* Loaded cached results from {fname}")

    feas_names = [n for n in all_names if feas_map.get(n, 1) == 1]
    tight_names = [sc["name"] for sc in scenes if sc["min_crop_dist"] < 12.0]

    # ==========================================================================
    # EXPERIMENT 1: Standard SE(2) MPPI (Auto-select fastest device CPU vs CUDA)
    # ==========================================================================
    print(f"\n=== [Exp 1] Standard SE(2) MPPI Baseline (K={args.mppi_k}, iters={args.mppi_iters}) on N={len(scenes)} ===")
    s0 = scenes[0]

    def bench_mppi_dev(dev_str):
        dev = torch.device(dev_str)
        c_t = torch.from_numpy(s0["crop_sdf_np"]).to(dev)
        w_t = torch.from_numpy(s0["weed_pts_np"]).to(dev)
        n_t = torch.from_numpy(nozzle_pts_np).to(dev)
        # warmup
        plan_mppi_se2(c_t, w_t, n_t, s0["start_xy"], s0["goal_xy"], s0["pca_th"], args.half_l, args.half_w, K=args.mppi_k, iters=10)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t_s = time.perf_counter()
            plan_mppi_se2(c_t, w_t, n_t, s0["start_xy"], s0["goal_xy"], s0["pca_th"], args.half_l, args.half_w, K=args.mppi_k, iters=args.mppi_iters)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t_s) * 1000.0)
        return float(np.mean(ts))

    mppi_cpu_ms = bench_mppi_dev("cpu")
    mppi_cuda_ms = bench_mppi_dev("cuda") if torch.cuda.is_available() else 1e9
    mppi_dev = torch.device("cuda" if mppi_cuda_ms < mppi_cpu_ms else "cpu")
    print(f"* MPPI Micro-Benchmark -> CPU: {mppi_cpu_ms:.1f} ms | CUDA: {mppi_cuda_ms:.1f} ms -> Selected: {mppi_dev}")

    nozzle_mppi_t = torch.from_numpy(nozzle_pts_np).to(mppi_dev)
    mppi_records = []
    t_mppi_all = time.perf_counter()
    for idx, sc in enumerate(scenes):
        c_t = torch.from_numpy(sc["crop_sdf_np"]).to(mppi_dev)
        w_t = torch.from_numpy(sc["weed_pts_np"]).to(mppi_dev)
        if mppi_dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        xy_m, th_m = plan_mppi_se2(
            c_t, w_t, nozzle_mppi_t, sc["start_xy"], sc["goal_xy"], sc["pca_th"],
            args.half_l, args.half_w, T=16, K=args.mppi_k, iters=args.mppi_iters, seed=2000 + idx
        )
        if mppi_dev.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        ev = evaluate_trajectory(xy_m, th_m, c_t, w_t, nozzle_mppi_t, args.half_l, args.half_w)
        rec = {
            "scene": sc["name"],
            "min_crop_dist": round(sc["min_crop_dist"], 3),
            "method": "mppi_se2",
            "latency_ms": round(dt_ms, 2),
            **ev,
        }
        mppi_records.append(rec)
        rows_by_scene[sc["name"]]["mppi_se2"] = rec
        if (idx + 1) % 100 == 0 or (idx + 1) == len(scenes):
            el = time.perf_counter() - t_mppi_all
            print(f"  [MPPI] Progress: {idx+1:4d}/{len(scenes)} | Elapsed: {el:5.1f}s", flush=True)

    methods_unified = [
        ("straight", "0. Initial Straight Path"),
        ("mppi_se2", f"1. Standard SE(2) MPPI ({mppi_dev.type.upper()})"),
        ("rrt_star", "2. Standard SE(2) RRT*"),
        ("plain_sdf_stl", "3. Plain SDF + STL (B=5)"),
        ("sdf_rcc8_stl", "4. SDF-RCC8 (NTPPi) + STL (B=5)"),
        ("hybrid_rrt_sdf_rcc8", "5. Hybrid RRT* + SDF-RCC8 (CPU)"),
    ]

    t1 = print_main_table("[Updated Table 1] ALL Real PhenoBench Val Scenes (with MPPI)", rows_by_scene, all_names, methods_unified)
    t2 = print_main_table("[Updated Table 2] STATICALLY FEASIBLE Subset (with MPPI)", rows_by_scene, feas_names, methods_unified)
    t3 = print_main_table("[Updated Table 3] TIGHT-CROP Subset < 12.0 px (with MPPI)", rows_by_scene, tight_names, methods_unified)

    print("\n=== [McNemar Significance vs MPPI (N=632)] ===")
    mppi_flags = [int(rows_by_scene[n]["mppi_se2"]["full_stl"]) for n in all_names]
    mcnemar_mppi = {}
    for k, lbl in [("hybrid_rrt_sdf_rcc8", "Hybrid RRT*+SDF-RCC8"), ("sdf_rcc8_stl", "Standalone SDF-RCC8")]:
        if all(k in rows_by_scene[n] for n in all_names):
            f_a = [int(rows_by_scene[n][k]["full_stl"]) for n in all_names]
            oa, ob, chi2, pv = mcnemar_stats(f_a, mppi_flags)
            mcnemar_mppi[k] = {"method_only": oa, "mppi_only": ob, "chi2": round(chi2, 3), "p_value": pv}
            print(f"  * {lbl:<22} vs MPPI | Method-Only: {oa:3d} vs MPPI-Only: {ob:3d} | chi2={chi2:6.2f}, p={pv:.3e}")

    # ==========================================================================
    # EXPERIMENT 2: 632-Scene Real-World Perception Noise + Kinodynamic omega_max
    # ==========================================================================
    print(f"\n=== [Exp 2] Real-World Perception Noise + Kinodynamic Limit (omega_max={args.omega_max:.2f} rad/step) on N={len(scenes)} ===")
    # 씬별 고정 시드 노이즈 마스크 1회 생성 (마진 0.0 / 2.5 / 4.5 공정 비교)
    noisy_cache = []
    for idx, sc in enumerate(scenes):
        rng = np.random.RandomState(2026 + idx)
        n_crop = corrupt_mask_realistic(sc["crop_np"], rng, min_pixels=10)
        n_weed = corrupt_mask_realistic(sc["weed_np"], rng, min_pixels=4)
        n_crop_sdf = compute_sdf_2d_np(n_crop)
        wy, wx = np.nonzero(n_weed > 0.5)
        n_weed_pts = np.stack([wx.astype(np.float32), wy.astype(np.float32)], axis=-1)
        n_goal, n_pca = compute_pca_from_pts(n_weed_pts)
        noisy_cache.append({
            "crop_sdf_np": n_crop_sdf,
            "weed_pts_np": n_weed_pts,
            "goal_xy": n_goal,
            "pca_th": n_pca,
        })

    cpu_dev = torch.device("cpu")
    nozzle_cpu_t = torch.from_numpy(nozzle_pts_np).to(cpu_dev)

    stress_configs = [
        ("clean_kino", "A. Clean GT + omega_max (Margin=1.5px)", False, 1.5, 0.2),
        ("noisy_m0.0", "B. Noisy Mask + omega_max (Margin=0.0px)", True, 0.0, 0.2),
        ("noisy_m2.5", "C. Noisy Mask + omega_max (Margin=2.5px)", True, 2.5, 2.0),
        ("noisy_m4.5", "D. Noisy Mask + omega_max (Margin=4.5px)", True, 4.5, 3.8),
    ]

    stress_records = []
    stress_summary_all = {}
    stress_summary_feas = {}

    for cfg_key, cfg_label, is_noisy, opt_margin, rrt_margin in stress_configs:
        t_cfg = time.perf_counter()
        cfg_rows = []
        for idx, sc in enumerate(scenes):
            # 플래너 입력 (노이즈 모드에서는 GT 절대 참조 금지)
            if is_noisy:
                nc = noisy_cache[idx]
                in_crop_sdf_np = nc["crop_sdf_np"]
                in_weed_pts_np = nc["weed_pts_np"]
                in_goal_xy = nc["goal_xy"]
                in_pca_th = nc["pca_th"]
            else:
                in_crop_sdf_np = sc["crop_sdf_np"]
                in_weed_pts_np = sc["weed_pts_np"]
                in_goal_xy = sc["goal_xy"]
                in_pca_th = sc["pca_th"]

            in_crop_t = torch.from_numpy(in_crop_sdf_np).to(cpu_dev)
            in_weed_t = torch.from_numpy(in_weed_pts_np).to(cpu_dev)
            gt_crop_t = torch.from_numpy(sc["crop_sdf_np"]).to(cpu_dev)
            gt_weed_t = torch.from_numpy(sc["weed_pts_np"]).to(cpu_dev)

            t0 = time.perf_counter()
            xy_rrt_np, th_rrt_np = plan_rrt_star_se2_margin(
                in_crop_sdf_np, in_weed_pts_np, sc["start_xy"], in_goal_xy, in_pca_th,
                nozzle_pts_np, args.half_l, args.half_w,
                T=16, max_iters=500, rrt_margin=rrt_margin, omega_max=args.omega_max, seed=3000 + idx
            )
            xy_rrt_t = torch.from_numpy(xy_rrt_np).to(cpu_dev)
            th_rrt_t = torch.from_numpy(th_rrt_np).to(cpu_dev)

            # Stage 1 Fast-Exit 여부도 오직 입력(노이즈) SDF + omega_max 기준으로 판정
            ev_in = evaluate_trajectory(
                xy_rrt_t, th_rrt_t, in_crop_t, in_weed_t, nozzle_cpu_t, args.half_l, args.half_w, omega_max=args.omega_max
            )
            if ev_in["full_stl"] == 1:
                final_xy, final_th = xy_rrt_t, th_rrt_t
            else:
                final_xy, final_th = refine_with_sdf_rcc8_kinodynamic(
                    xy_rrt_np, th_rrt_np, in_crop_t, in_weed_t, nozzle_cpu_t,
                    sc["start_xy"], in_goal_xy, in_pca_th, args.half_l, args.half_w,
                    crop_margin=opt_margin, omega_max=args.omega_max, max_iters=100, check_every=10
                )
            dt_ms = (time.perf_counter() - t0) * 1000.0

            # 최종 채점은 반드시 TRUE Ground Truth + 엄격 omega_max 기준!
            ev_gt = evaluate_trajectory(
                final_xy, final_th, gt_crop_t, gt_weed_t, nozzle_cpu_t, args.half_l, args.half_w, omega_max=args.omega_max
            )
            rec = {
                "scene": sc["name"],
                "config": cfg_key,
                "static_feasible": feas_map.get(sc["name"], 1),
                "latency_ms": round(dt_ms, 2),
                **ev_gt,
            }
            cfg_rows.append(rec)
            stress_records.append(rec)

        el = time.perf_counter() - t_cfg
        feas_rows = [r for r in cfg_rows if r["static_feasible"] == 1]

        def pack_stress(rlist):
            return {
                "n": len(rlist),
                "full_stl_cnt": sum(r["full_stl"] for r in rlist),
                "full_stl_pct": 100.0 * np.mean([r["full_stl"] for r in rlist]),
                "zero_coll_pct": 100.0 * np.mean([r["zero_coll"] for r in rlist]),
                "crop_safety_pct": 100.0 * np.mean([r["crop_safety"] for r in rlist]),
                "weed_cov_pct": 100.0 * np.mean([r["weed_cov"] for r in rlist]),
                "kino_ok_pct": 100.0 * np.mean([r["kino_ok"] for r in rlist]),
                "max_dth_mean": float(np.mean([r["max_dth"] for r in rlist])),
                "latency_ms": float(np.mean([r["latency_ms"] for r in rlist])),
            }

        sa = pack_stress(cfg_rows)
        sf = pack_stress(feas_rows)
        stress_summary_all[cfg_key] = sa
        stress_summary_feas[cfg_key] = sf
        print(f"  Done {cfg_label:<42} ({el:5.1f}s) | ALL FullSTL={sa['full_stl_pct']:5.1f}% | ZeroColl={sa['zero_coll_pct']:5.1f}% | CropSafe={sa['crop_safety_pct']:5.1f}% | KinoOK={sa['kino_ok_pct']:5.1f}%", flush=True)

    # Table 4 출력 (전체 632장 + 물리 가능 536장)
    for title, sdict, n_disp in [
        (f"[Table 4A] Perception Noise + Kinodynamic (omega_max={args.omega_max}) on ALL Scenes", stress_summary_all, len(all_names)),
        (f"[Table 4B] Perception Noise + Kinodynamic (omega_max={args.omega_max}) on FEASIBLE Scenes", stress_summary_feas, len(feas_names)),
    ]:
        print(f"\n{title} (N={n_disp})")
        print("-" * 130)
        print(f"{'Condition (Plan -> Eval on True GT)':<42} | {'True Full STL':>14} | {'Zero-Coll':>9} | {'CropSafe':>8} | {'WeedCov':>7} | {'KinoOK':>6} | {'Latency':>9}")
        print("-" * 130)
        for cfg_key, cfg_label, _, _, _ in stress_configs:
            s = sdict[cfg_key]
            print(
                f"{cfg_label:<42} | {s['full_stl_cnt']:4d}/{s['n']:<4d} ({s['full_stl_pct']:5.1f}%) "
                f"| {s['zero_coll_pct']:8.1f}% | {s['crop_safety_pct']:7.1f}% "
                f"| {s['weed_cov_pct']:6.1f}% | {s['kino_ok_pct']:5.1f}% | {s['latency_ms']:6.1f} ms"
            )
        print("-" * 130)

    # 파일 저장
    with open("phenobench_mppi_per_scene.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mppi_records[0].keys()))
        w.writeheader()
        w.writerows(mppi_records)

    with open("phenobench_stress632_per_scene.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stress_records[0].keys()))
        w.writeheader()
        w.writerows(stress_records)

    with open("phenobench_final_ral_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "mppi_device": str(mppi_dev),
            "mppi_micro_ms": {"cpu": round(mppi_cpu_ms, 2), "cuda": round(mppi_cuda_ms, 2)},
            "omega_max_rad_per_step": args.omega_max,
            "table1_all_with_mppi": t1,
            "table2_feasible_with_mppi": t2,
            "table3_tight_with_mppi": t3,
            "mcnemar_vs_mppi": mcnemar_mppi,
            "table4a_stress_all": stress_summary_all,
            "table4b_stress_feasible": stress_summary_feas,
        }, f, indent=2)

    print("[Saved] phenobench_mppi_per_scene.csv, phenobench_stress632_per_scene.csv, phenobench_final_ral_summary.json")


if __name__ == "__main__":
    main()
