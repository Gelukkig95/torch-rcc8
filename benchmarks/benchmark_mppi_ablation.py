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


def get_nozzle_local_points_np(half_l: float, half_w: float) -> np.ndarray:
    xs = np.linspace(-half_l, half_l, 9, dtype=np.float32)
    ys = np.linspace(-half_w, half_w, 5, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)


# ------------------------------------------------------------------------------
# Exact Scene Extractor (632장 동일 인스턴스 + weed_sdf_np 포함)
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
    weed_sdf_np = compute_sdf_2d_np(weed_np)
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
        "crop_sdf_np": crop_sdf_np,
        "weed_sdf_np": weed_sdf_np,
        "weed_pts_np": target["pts"],
        "start_xy": best_start,
        "goal_xy": (gx, gy),
        "pca_th": target["pca_th"],
        "min_crop_dist": target["min_crop_dist"],
    }, "ok"


# ------------------------------------------------------------------------------
# Unified Strict External Evaluator (모든 알고리즘 동일 채점)
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_trajectory(xy, th, crop_sdf, weed_pts, nozzle_pts, half_l, half_w):
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
    return {
        "full_stl": int(crop_safe_all and weed_ntppi_ok),
        "zero_coll": int(crop_safe_all),
        "crop_safety": safe_step_ratio,
        "weed_ntppi": int(weed_ntppi_ok),
        "weed_cov": best_cov,
        "path_len": float(torch.norm(dxy, dim=-1).sum().item()),
        "ang_tv": float(torch.abs(dth).sum().item()),
    }


@torch.no_grad()
def evaluate_plain_sdf_key(xy, th, crop_sdf, weed_sdf, nozzle_pts):
    """Plain SDF 전용 내부 후보 비교 키 (NTPPi/weed_pts 절대 미사용)."""
    H, W = crop_sdf.shape
    T = xy.shape[0]
    cos_t = torch.cos(th).unsqueeze(-1)
    sin_t = torch.sin(th).unsqueeze(-1)
    wx = xy[:, 0:1] + cos_t * nozzle_pts[:, 0].unsqueeze(0) - sin_t * nozzle_pts[:, 1].unsqueeze(0)
    wy = xy[:, 1:2] + sin_t * nozzle_pts[:, 0].unsqueeze(0) + cos_t * nozzle_pts[:, 1].unsqueeze(0)
    in_b = (wx >= 0.0) & (wx <= W - 1.0) & (wy >= 0.0) & (wy <= H - 1.0)
    crop_d = sample_sdf_batched(crop_sdf, torch.stack([wx, wy], dim=-1).unsqueeze(0))[0]
    safe_s = ((crop_d > 0.0) & in_b).all(dim=1)
    z_coll = int(safe_s.all().item())
    c_safe = float(safe_s.float().mean().item())
    wd = sample_sdf_batched(weed_sdf, xy.view(1, T, 1, 2))[0, :, 0]
    p_ok = int(((wd < 0.0) & safe_s).any().item())
    return (int(z_coll == 1 and p_ok == 1), z_coll, c_safe, -float(wd.min().item()))


# ------------------------------------------------------------------------------
# Sync-Free Paired SE(2) MPPI Solver (Plain SDF vs SDF-RCC8, Zero-Init vs PCA-Init)
# ------------------------------------------------------------------------------
@torch.no_grad()
def plan_mppi_se2_ablation(
    crop_sdf: torch.Tensor,
    weed_sdf: torch.Tensor,
    weed_pts: torch.Tensor,
    nozzle_pts: torch.Tensor,
    start_xy,
    goal_xy,
    pca_th: float,
    half_l: float,
    half_w: float,
    mode: str = "sdf_rcc8",
    use_pca_init: bool = True,
    T: int = 16,
    K: int = 384,
    iters: int = 45,
    crop_margin: float = 1.8,
    seed: int = 42,
):
    device = crop_sdf.device
    H, W = crop_sdf.shape
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    p0 = torch.tensor(start_xy, device=device, dtype=torch.float32)
    p1 = torch.tensor(goal_xy, device=device, dtype=torch.float32)

    U = torch.zeros((T - 1, 3), device=device, dtype=torch.float32)
    U[:, :2] = (p1 - p0) / float(T - 1)
    if use_pca_init:
        cand_a = pca_th
        cand_b = wrap_angle_np(pca_th + math.pi)
        target_th = cand_a if abs(cand_a) <= abs(cand_b) else cand_b
        U[:, 2] = target_th / float(T - 1)

    lx = nozzle_pts[:, 0].view(1, 1, -1)
    ly = nozzle_pts[:, 1].view(1, 1, -1)
    w_px = weed_pts[:, 0].view(1, 1, -1)
    w_py = weed_pts[:, 1].view(1, 1, -1)

    start_xy_k = p0.view(1, 1, 2).expand(K, 1, 2)
    start_th_k = torch.zeros((K, 1), device=device, dtype=torch.float32)
    smooth_k = torch.tensor([0.25, 0.5, 0.25], device=device).view(1, 1, 3)

    # 루프 내 CPU-GPU 동기화(0회)를 위한 순수 GPU 엘리트 상태 버퍼
    best_tier = torch.tensor(-1, device=device, dtype=torch.int64)
    best_J = torch.tensor(float("inf"), device=device, dtype=torch.float32)
    best_xy = torch.zeros((T, 2), device=device, dtype=torch.float32)
    best_th = torch.zeros((T,), device=device, dtype=torch.float32)
    inf_val = torch.tensor(float("inf"), device=device, dtype=torch.float32)

    for it in range(iters):
        prog = it / max(iters - 1, 1)
        sig_xy = 4.2 * (1.0 - 0.72 * prog)
        sig_th = 0.36 * (1.0 - 0.75 * prog)

        eps = torch.randn((K, T - 1, 3), generator=gen, device=device)
        half_k = K // 2
        c_noise = eps[:half_k].permute(0, 2, 1).reshape(half_k * 3, 1, T - 1)
        for _ in range(3):
            c_noise = F.conv1d(F.pad(c_noise, (1, 1), mode="replicate"), smooth_k)
        eps[:half_k] = c_noise.reshape(half_k, 3, T - 1).permute(0, 2, 1) * 1.9

        eps[..., :2] *= sig_xy
        eps[..., 2] *= sig_th
        eps[0].zero_()

        U_cand = U.unsqueeze(0) + eps
        U_cand[..., :2].clamp_(-16.0, 16.0)
        U_cand[..., 2].clamp_(-0.75, 0.75)

        xy_steps = start_xy_k + torch.cumsum(U_cand[..., :2], dim=1)
        xy_all = torch.cat([start_xy_k, xy_steps], dim=1)
        th_all = torch.cat([start_th_k, torch.cumsum(U_cand[..., 2], dim=1)], dim=1)

        cos_t = torch.cos(th_all).unsqueeze(-1)
        sin_t = torch.sin(th_all).unsqueeze(-1)

        wx = xy_all[:, :, 0:1] + cos_t * lx - sin_t * ly
        wy = xy_all[:, :, 1:2] + sin_t * lx + cos_t * ly
        in_b = (wx >= 0.0) & (wx <= W - 1.0) & (wy >= 0.0) & (wy <= H - 1.0)
        crop_d = sample_sdf_batched(crop_sdf, torch.stack([wx, wy], dim=-1))

        # 1. 공통 작물 비충돌(DC) 비용
        viol_margin = F.relu(crop_margin - crop_d)
        hard_coll = ((crop_d <= 0.0) | (~in_b)).float()
        cost_crop = (
            (viol_margin ** 2).mean(dim=(1, 2)) * 18.0
            + viol_margin.amax(dim=(1, 2)) * 12.0
            + hard_coll.sum(dim=(1, 2)) * 25.0
        )

        safe_steps = ((crop_d > 0.0) & in_b).all(dim=2)
        zero_coll = safe_steps.all(dim=1)

        # 2. 타깃 목적함수 분기: Plain SDF vs SDF-RCC8 (NTPPi)
        if mode == "plain_sdf":
            center_d = sample_sdf_batched(weed_sdf, xy_all.unsqueeze(2)).squeeze(-1)  # (K, T)
            best_center_d = center_d.amin(dim=1)
            cost_weed = (
                F.relu(best_center_d + 2.0) * 14.0
                + F.relu(center_d[:, -1] + 2.0) * 6.0
            )
            target_ok = ((center_d < 0.0) & safe_steps).any(dim=1)
        else:
            dx = w_px - xy_all[:, :, 0:1]
            dy = w_py - xy_all[:, :, 1:2]
            wlx = cos_t * dx + sin_t * dy
            wly = -sin_t * dx + cos_t * dy
            w_sd = nozzle_analytical_sdf_torch(torch.stack([wlx, wly], dim=-1), half_l, half_w)
            step_max_w = w_sd.amax(dim=2)
            best_step_w = step_max_w.amin(dim=1)
            cost_weed = (
                F.relu(best_step_w + 0.8) * 14.0
                + F.relu(step_max_w[:, -1] + 0.8) * 6.0
                + F.relu(w_sd[:, -1, :] + 0.3).mean(dim=1) * 10.0
            )
            target_ok = ((step_max_w < 0.0) & safe_steps).any(dim=1)

        # 3. 공통 제어 매끄러움 비용
        cost_ctrl = (U_cand[..., :2] ** 2).mean(dim=(1, 2)) * 0.08 + (U_cand[..., 2] ** 2).mean(dim=1) * 3.0
        J = cost_crop + cost_weed + cost_ctrl

        # Sync-Free GPU 엘리트 갱신 (기존 tuple 비교와 수학적으로 100% 동치)
        full_ok = zero_coll & target_ok
        tier_k = zero_coll.to(torch.int64) + full_ok.to(torch.int64)  # 0, 1, 2
        max_tier = tier_k.max()
        J_masked = torch.where(tier_k == max_tier, J, inf_val)
        bi = torch.argmin(J_masked)
        cand_J = J[bi]
        better = (max_tier > best_tier) | ((max_tier == best_tier) & (cand_J < best_J))
        best_tier = torch.where(better, max_tier, best_tier)
        best_J = torch.where(better, cand_J, best_J)
        best_xy = torch.where(better.view(1, 1), xy_all[bi], best_xy)
        best_th = torch.where(better.view(1), th_all[bi], best_th)

        j_min = J.min()
        temp = torch.clamp(J.std() * 0.25, min=1.0)
        weights = torch.softmax(-(J - j_min) / temp, dim=0)
        U = (weights[:, None, None] * U_cand).sum(dim=0)

    xy_nom = torch.cat([p0.view(1, 2), p0.view(1, 2) + torch.cumsum(U[:, :2], dim=0)], dim=0)
    th_nom = torch.cat([torch.zeros(1, device=device), torch.cumsum(U[:, 2], dim=0)], dim=0)

    if mode == "plain_sdf":
        # Plain SDF는 최종 선택에서도 절대 NTPPi를 참조하지 않음 (누수 0%)
        k_nom = evaluate_plain_sdf_key(xy_nom, th_nom, crop_sdf, weed_sdf, nozzle_pts)
        k_best = evaluate_plain_sdf_key(best_xy, best_th, crop_sdf, weed_sdf, nozzle_pts)
        if k_nom > k_best:
            return xy_nom.detach(), th_nom.detach()
        return best_xy.detach(), best_th.detach()
    else:
        ev_nom = evaluate_trajectory(xy_nom, th_nom, crop_sdf, weed_pts, nozzle_pts, half_l, half_w)
        ev_best = evaluate_trajectory(best_xy, best_th, crop_sdf, weed_pts, nozzle_pts, half_l, half_w)
        if (ev_nom["full_stl"], ev_nom["zero_coll"], ev_nom["crop_safety"], ev_nom["weed_cov"]) > (
            ev_best["full_stl"], ev_best["zero_coll"], ev_best["crop_safety"], ev_best["weed_cov"]
        ):
            return xy_nom.detach(), th_nom.detach()
        return best_xy.detach(), best_th.detach()


# ------------------------------------------------------------------------------
# Reporting & McNemar Helpers
# ------------------------------------------------------------------------------
def mcnemar_stats(a_flags, b_flags):
    both_ok = sum(1 for a, b in zip(a_flags, b_flags) if a == 1 and b == 1)
    only_a = sum(1 for a, b in zip(a_flags, b_flags) if a == 1 and b == 0)
    only_b = sum(1 for a, b in zip(a_flags, b_flags) if a == 0 and b == 1)
    both_fail = sum(1 for a, b in zip(a_flags, b_flags) if a == 0 and b == 0)
    denom = only_a + only_b
    chi2 = ((abs(only_a - only_b) - 1.0) ** 2) / denom if denom > 0 else 0.0
    p_val = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "both_ok": both_ok,
        "only_a": only_a,
        "only_b": only_b,
        "both_fail": both_fail,
        "chi2": round(chi2, 3),
        "p_value": p_val,
    }


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


def print_table(title, rows_by_scene, scene_names, methods):
    print(f"\n{title} (N={len(scene_names)})")
    print("-" * 124)
    print(f"{'Method':<40} | {'Full STL':>14} | {'Zero-Coll':>9} | {'CropSafe':>8} | {'WeedCov':>7} | {'AngTV':>6} | {'Latency':>9}")
    print("-" * 124)
    out = {}
    for k, label in methods:
        s = summarize_method(rows_by_scene, scene_names, k)
        if s is None:
            continue
        out[k] = s
        print(
            f"{label:<40} | {s['full_stl_cnt']:4d}/{s['n']:<4d} ({s['full_stl_pct']:5.1f}%) "
            f"| {s['zero_coll_pct']:8.1f}% | {s['crop_safety_pct']:7.1f}% "
            f"| {s['weed_cov_pct']:6.1f}% | {s['ang_tv']:6.2f} | {s['latency_ms']:6.1f} ms"
        )
    print("-" * 124)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--half_l", type=float, default=14.5)
    parser.add_argument("--half_w", type=float, default=7.0)
    parser.add_argument("--mppi_k", type=int, default=384)
    parser.add_argument("--mppi_iters", type=int, default=45)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"=== [Solver-Agnostic 2x2 MPPI Ablation: Plain SDF vs SDF-RCC8] ===")
    print(f"* Device: {device} ({gpu_name}) | PyTorch: {torch.__version__}")

    sem_dir = Path(args.val_dir) / "semantics"
    sem_files = sorted(sem_dir.glob("*.png"))
    nozzle_pts_np = get_nozzle_local_points_np(args.half_l, args.half_w)
    nozzle_t = torch.from_numpy(nozzle_pts_np).to(device)

    scenes = []
    for p in sem_files:
        sc, status = extract_phenobench_scene(p, args.grid_size, args.half_l, args.half_w, nozzle_pts_np)
        if status == "ok":
            scenes.append(sc)
    all_names = [sc["name"] for sc in scenes]
    tight_names = [sc["name"] for sc in scenes if sc["min_crop_dist"] < 12.0]
    print(f"* Loaded {len(scenes)} Real PhenoBench Val Scenes.")

    # 기존 캐시 로드 (Adam Plain SDF, Adam SDF-RCC8, RRT*, Hybrid, 이전 MPPI 검증용)
    rows_by_scene = defaultdict(dict)
    feas_map = {}
    for fname in [
        "phenobench_full_benchmark_per_scene.csv",
        "phenobench_hybrid_per_scene.csv",
        "phenobench_mppi_per_scene.csv",
    ]:
        fp = Path(fname)
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows_by_scene[row["scene"]][row["method"]] = row
                    if "static_feasible" in row:
                        feas_map[row["scene"]] = int(row["static_feasible"])
            print(f"* Loaded cached baseline: {fname}")

    feas_names = [n for n in all_names if feas_map.get(n, 1) == 1]

    # CUDA 커널 웜업 + 동기화 제거 속도 점검
    s0 = scenes[0]
    c0 = torch.from_numpy(s0["crop_sdf_np"]).to(device)
    ws0 = torch.from_numpy(s0["weed_sdf_np"]).to(device)
    wp0 = torch.from_numpy(s0["weed_pts_np"]).to(device)
    for _ in range(3):
        plan_mppi_se2_ablation(c0, ws0, wp0, nozzle_t, s0["start_xy"], s0["goal_xy"], s0["pca_th"], args.half_l, args.half_w, mode="sdf_rcc8", use_pca_init=True)
        plan_mppi_se2_ablation(c0, ws0, wp0, nozzle_t, s0["start_xy"], s0["goal_xy"], s0["pca_th"], args.half_l, args.half_w, mode="plain_sdf", use_pca_init=False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 2x2 요인 실험 구성
    ablation_configs = [
        ("mppi_plain_zero", "Plain SDF", False, "MPPI + Plain SDF (Zero-Init, th0=0)"),
        ("mppi_plain_pca",  "Plain SDF", True,  "MPPI + Plain SDF (with PCA-Init)"),
        ("mppi_rcc8_zero",  "SDF-RCC8",  False, "MPPI + SDF-RCC8 (Zero-Init, th0=0)"),
        ("mppi_rcc8_pca",   "SDF-RCC8",  True,  "MPPI + SDF-RCC8 (with PCA-Init, SyncFree)"),
    ]

    new_records = []
    t_total = time.perf_counter()

    for cfg_key, mode_lbl, use_pca, disp_name in ablation_configs:
        mode_arg = "plain_sdf" if mode_lbl == "Plain SDF" else "sdf_rcc8"
        t_cfg = time.perf_counter()
        ok_cnt = 0
        for idx, sc in enumerate(scenes):
            c_t = torch.from_numpy(sc["crop_sdf_np"]).to(device)
            ws_t = torch.from_numpy(sc["weed_sdf_np"]).to(device)
            wp_t = torch.from_numpy(sc["weed_pts_np"]).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            xy_out, th_out = plan_mppi_se2_ablation(
                c_t, ws_t, wp_t, nozzle_t,
                sc["start_xy"], sc["goal_xy"], sc["pca_th"],
                args.half_l, args.half_w,
                mode=mode_arg, use_pca_init=use_pca,
                T=16, K=args.mppi_k, iters=args.mppi_iters, seed=2000 + idx,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0

            ev = evaluate_trajectory(xy_out, th_out, c_t, wp_t, nozzle_t, args.half_l, args.half_w)
            ok_cnt += ev["full_stl"]
            rec = {
                "scene": sc["name"],
                "min_crop_dist": round(sc["min_crop_dist"], 3),
                "static_feasible": feas_map.get(sc["name"], 1),
                "method": cfg_key,
                "latency_ms": round(dt_ms, 2),
                **ev,
            }
            new_records.append(rec)
            rows_by_scene[sc["name"]][cfg_key] = rec

        el = time.perf_counter() - t_cfg
        mean_lat = np.mean([rows_by_scene[n][cfg_key]["latency_ms"] for n in all_names])
        print(f"  [Done] {disp_name:<42} | Full STL: {ok_cnt:3d}/{len(scenes)} ({100.0*ok_cnt/len(scenes):5.1f}%) | Mean Latency: {mean_lat:5.1f} ms | Wall: {el:5.1f}s", flush=True)

    print(f"* Total Wall-Clock Time for 4x632 (=2,528) Rollout Optimizations: {time.perf_counter() - t_total:.1f}s")

    # 재현성 검증: 이전 mppi_se2 (452/632) vs Sync-Free mppi_rcc8_pca 일치율 확인
    if all("mppi_se2" in rows_by_scene[n] for n in all_names):
        exact_match = sum(
            1 for n in all_names
            if int(rows_by_scene[n]["mppi_se2"]["full_stl"]) == int(rows_by_scene[n]["mppi_rcc8_pca"]["full_stl"])
        )
        old_ms = np.mean([float(rows_by_scene[n]["mppi_se2"]["latency_ms"]) for n in all_names])
        new_ms = np.mean([float(rows_by_scene[n]["mppi_rcc8_pca"]["latency_ms"]) for n in all_names])
        print(f"\n[Reproducibility & Sync-Free Speedup Audit]")
        print(f"  * Cached mppi_se2 vs Sync-Free mppi_rcc8_pca Scene-by-Scene Agreement: {exact_match}/{len(all_names)} ({100.0*exact_match/len(all_names):.2f}%)")
        print(f"  * CUDA Latency Reduction (Removing per-iter CPU sync): {old_ms:.1f} ms -> {new_ms:.1f} ms ({old_ms/max(new_ms, 1e-6):.2f}x faster)")

    # 1. 2x2 MPPI Factorial Ablation Table
    mppi_methods = [(k, lbl) for k, _, _, lbl in ablation_configs]
    t_mppi_all = print_table("[Table A] 2x2 MPPI Factorial Ablation — ALL Scenes", rows_by_scene, all_names, mppi_methods)
    t_mppi_feas = print_table("[Table B] 2x2 MPPI Factorial Ablation — STATICALLY FEASIBLE Only", rows_by_scene, feas_names, mppi_methods)
    t_mppi_tight = print_table("[Table C] 2x2 MPPI Factorial Ablation — TIGHT-CROP (< 12.0 px)", rows_by_scene, tight_names, mppi_methods)

    # 2. Solver-Agnostic Master Table (Adam vs MPPI vs RRT* vs Hybrid)
    master_methods = [
        ("straight",            "0. Initial Straight Path"),
        ("rrt_star",            "1. Standard SE(2) RRT* (CPU)"),
        ("plain_sdf_stl",       "2. [Adam B=5] Plain SDF + STL"),
        ("sdf_rcc8_stl",        "3. [Adam B=5] SDF-RCC8 (NTPPi) + STL"),
        ("mppi_plain_zero",     "4. [MPPI K=384] Plain SDF (Zero-Init)"),
        ("mppi_rcc8_zero",      "5. [MPPI K=384] SDF-RCC8 (Zero-Init)"),
        ("mppi_plain_pca",      "6. [MPPI K=384] Plain SDF (PCA-Init)"),
        ("mppi_rcc8_pca",       "7. [MPPI K=384] SDF-RCC8 (PCA-Init)"),
        ("hybrid_rrt_sdf_rcc8", "8. [Hybrid CPU] RRT* + SDF-RCC8"),
    ]
    t_master_all = print_table("[Master Table 1] Solver-Agnostic Topology Benchmark — ALL Scenes", rows_by_scene, all_names, master_methods)
    t_master_feas = print_table("[Master Table 2] Solver-Agnostic Topology Benchmark — FEASIBLE Scenes", rows_by_scene, feas_names, master_methods)

    # 3. McNemar Statistical Significance Tests
    print("\n=== [McNemar Paired Significance Tests (N=632)] ===")
    pairs = [
        ("mppi_rcc8_zero", "mppi_plain_zero", "Pure Topology Effect (Zero-Init: SDF-RCC8 vs Plain SDF)"),
        ("mppi_rcc8_pca",  "mppi_plain_pca",  "With PCA Prior       (PCA-Init : SDF-RCC8 vs Plain SDF)"),
        ("mppi_rcc8_pca",  "mppi_plain_zero", "Full MPPI SDF-RCC8   vs Standard Plain SDF MPPI"),
        ("mppi_rcc8_zero", "mppi_plain_pca",  "SDF-RCC8 (No PCA!)   vs Plain SDF (Gifted PCA!)"),
    ]
    mcnemar_summary = {}
    for ka, kb, title in pairs:
        fa = [int(rows_by_scene[n][ka]["full_stl"]) for n in all_names]
        fb = [int(rows_by_scene[n][kb]["full_stl"]) for n in all_names]
        st = mcnemar_stats(fa, fb)
        mcnemar_summary[f"{ka}_vs_{kb}"] = st
        print(
            f"  * {title:<56} | Wins {st['only_a']:3d} vs {st['only_b']:3d} "
            f"| chi2={st['chi2']:6.2f}, p={st['p_value']:.3e}"
        )

    # 저장
    with open("phenobench_mppi_ablation_per_scene.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(new_records[0].keys()))
        w.writeheader()
        w.writerows(new_records)

    with open("phenobench_mppi_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "device": str(device),
            "gpu_name": gpu_name,
            "mppi_2x2_all": t_mppi_all,
            "mppi_2x2_feasible": t_mppi_feas,
            "mppi_2x2_tight": t_mppi_tight,
            "master_all": t_master_all,
            "master_feasible": t_master_feas,
            "mcnemar_tests": mcnemar_summary,
        }, f, indent=2)

    print("\n[Saved] phenobench_mppi_ablation_per_scene.csv & phenobench_mppi_ablation_summary.json")


if __name__ == "__main__":
    main()
