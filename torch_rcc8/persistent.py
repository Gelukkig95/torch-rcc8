r"""
torch_rcc8/persistent.py
------------------------------
Pure PyTorch 2D/3D Multi-Scale Granulometric Filtration & Persistent RCC-8 Operators.
Uses exact Galois-adjunction domain-restricted erosion and -inf-padded dilation so both
isotropic and 4-directional anisotropic openings satisfy strict idempotence:
gamma(gamma(f)) == gamma(f) to machine precision (0.00e+00).
"""

from __future__ import annotations
from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .diagnostics import compute_rcc8_matrix, morph_dilation, morph_erosion
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from torch_rcc8.diagnostics import compute_rcc8_matrix, morph_dilation, morph_erosion

def _adjunction_isotropic_open(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Exact Galois-adjunction isotropic opening: unpadded erosion + (-inf)-padded dilation."""
    if radius <= 0:
        return x
    orig_shape = x.shape
    t = x if x.is_floating_point() else x.float()
    while t.ndim < 4:
        t = t.unsqueeze(0)

    spatial = t.shape[2:]
    r = min(radius, *(max(0, (s - 1) // 2) for s in spatial))
    if r <= 0:
        return x
    k = 2 * r + 1

    if t.ndim == 5:
        ero = -F.max_pool3d(-t, kernel_size=k, stride=1, padding=0)
        ero_pad = F.pad(ero, (2 * r,) * 6, mode="constant", value=float("-inf"))
        out = F.max_pool3d(ero_pad, kernel_size=k, stride=1, padding=0)
    else:
        ero = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=0)
        ero_pad = F.pad(ero, (2 * r,) * 4, mode="constant", value=float("-inf"))
        out = F.max_pool2d(ero_pad, kernel_size=k, stride=1, padding=0)
    return out.reshape(orig_shape)

def _adjunction_directional_open(x: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Matheron's algebraic opening: supremum of exact Galois-adjunction 1D openings
    across {0, 90, 45, 135} degrees (2D) or {X, Y, Z} axes (3D).
    """
    if radius <= 0:
        return x
    orig_shape = x.shape
    t = x if x.is_floating_point() else x.float()
    while t.ndim < 4:
        t = t.unsqueeze(0)

    spatial = t.shape[2:]
    r = min(radius, *(max(0, (s - 1) // 2) for s in spatial))
    if r <= 0:
        return x
    k = 2 * r + 1

    if t.ndim == 5:
        B, C, D, H, W = t.shape
        tf = t.reshape(B * C, 1, D, H, W)
        ez = -F.max_pool3d(-tf, (k, 1, 1), 1, 0)
        oz = F.max_pool3d(F.pad(ez, (0, 0, 0, 0, 2 * r, 2 * r), "constant", float("-inf")), (k, 1, 1), 1, 0)
        ey = -F.max_pool3d(-tf, (1, k, 1), 1, 0)
        oy = F.max_pool3d(F.pad(ey, (0, 0, 2 * r, 2 * r, 0, 0), "constant", float("-inf")), (1, k, 1), 1, 0)
        ex = -F.max_pool3d(-tf, (1, 1, k), 1, 0)
        ox = F.max_pool3d(F.pad(ex, (2 * r, 2 * r, 0, 0, 0, 0), "constant", float("-inf")), (1, 1, k), 1, 0)
        return torch.maximum(torch.maximum(oz, oy), ox).reshape(orig_shape)

    B, C, H, W = t.shape
    tf = t.reshape(B * C, 1, H, W)
    Hp, Wp = H - 2 * r, W - 2 * r

    # 0 deg (Horizontal)
    eh = -F.max_pool2d(-tf, (1, k), 1, 0)
    oh = F.max_pool2d(F.pad(eh, (2 * r, 2 * r, 0, 0), "constant", float("-inf")), (1, k), 1, 0)

    # 90 deg (Vertical)
    ev = -F.max_pool2d(-tf, (k, 1), 1, 0)
    ov = F.max_pool2d(F.pad(ev, (0, 0, 2 * r, 2 * r), "constant", float("-inf")), (k, 1), 1, 0)

    # 45 deg (NW-SE diagonal)
    ed1 = tf[:, :, 0:Hp, 0:Wp]
    for d in range(1, k):
        ed1 = torch.minimum(ed1, tf[:, :, d : d + Hp, d : d + Wp])
    ed1_pad = F.pad(ed1, (2 * r, 2 * r, 2 * r, 2 * r), "constant", float("-inf"))
    od1 = ed1_pad[:, :, 0:H, 0:W]
    for s in range(1, k):
        od1 = torch.maximum(od1, ed1_pad[:, :, s : s + H, s : s + W])

    # 135 deg (NE-SW diagonal)
    ed2 = tf[:, :, 0:Hp, 2 * r : 2 * r + Wp]
    for d in range(1, k):
        col = 2 * r - d
        ed2 = torch.minimum(ed2, tf[:, :, d : d + Hp, col : col + Wp])
    ed2_pad = F.pad(ed2, (2 * r, 2 * r, 2 * r, 2 * r), "constant", float("-inf"))
    od2 = ed2_pad[:, :, 0:H, 2 * r : 2 * r + W]
    for s in range(1, k):
        col = 2 * r - s
        od2 = torch.maximum(od2, ed2_pad[:, :, s : s + H, col : col + W])

    out = torch.maximum(torch.maximum(oh, ov), torch.maximum(od1, od2))
    return out.reshape(orig_shape)

def morph_open_clean(x: torch.Tensor, radius: int = 1, mode: str = "isotropic") -> torch.Tensor:
    if radius <= 0:
        return x
    if mode == "anisotropic":
        return _adjunction_directional_open(x, radius)
    if mode == "isotropic":
        return _adjunction_isotropic_open(x, radius)
    raise ValueError(f"Invalid mode '{mode}'. Expected 'isotropic' or 'anisotropic'.")

def morph_close_clean(x: torch.Tensor, radius: int = 1, mode: str = "isotropic") -> torch.Tensor:
    if radius <= 0:
        return x
    # Exact morphological duality: phi(x) = -gamma(-x)
    return -morph_open_clean(-x, radius=radius, mode=mode)

def granulometric_persistence(
    prob: torch.Tensor,
    filtration_radii: Sequence[int] = (1, 2, 4),
    close_pinholes: bool = False,
    mode: str = "isotropic",
) -> torch.Tensor:
    radii = sorted([int(r) for r in filtration_radii if r >= 0])
    if not radii:
        return prob

    acc = torch.zeros_like(prob, dtype=torch.float32)
    t_prob = prob.float()
    for r in radii:
        filtered = morph_open_clean(t_prob, radius=r, mode=mode)
        if close_pinholes and r > 0:
            filtered = morph_close_clean(filtered, radius=r, mode=mode)
        acc = acc + filtered
    return acc / float(len(radii))

@torch.no_grad()
def compute_persistent_rcc8_matrix(
    masks: torch.Tensor,
    filtration_radii: Sequence[int] = (0, 1, 2, 3),
    kernel_size: int = 3,
    close_pinholes: bool = True,
    raw_scale_weight: float = 0.10,
    mode: str = "isotropic",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if masks.ndim not in (3, 4, 5):
        raise ValueError(f"Expected 3D, 4D, or 5D mask tensor, got {masks.shape}")

    is_unbatched = masks.ndim == 3
    if is_unbatched:
        masks = masks.unsqueeze(0)

    B, N = masks.shape[:2]
    spatial_dims = tuple(range(2, masks.ndim))
    device = masks.device

    if N == 0:
        empty_dom = torch.zeros((B, 0, 0), dtype=torch.long, device=device)
        empty_dist = torch.zeros((B, 0, 0, 8), dtype=torch.float32, device=device)
        empty_surv = torch.zeros((B, 0), dtype=torch.float32, device=device)
        if is_unbatched:
            return empty_dom.squeeze(0), empty_dist.squeeze(0), empty_surv.squeeze(0)
        return empty_dom, empty_dist, empty_surv

    radii = sorted(set(int(r) for r in filtration_radii if r >= 0))
    if not radii:
        radii = [0]

    if len(radii) == 1:
        scale_weights = torch.ones(1, dtype=torch.float32, device=device)
    else:
        pos_count = sum(1 for r in radii if r > 0)
        w_list = []
        for r in radii:
            if r == 0 and pos_count > 0:
                w_list.append(raw_scale_weight)
            else:
                rem = (1.0 - raw_scale_weight) if (0 in radii and pos_count > 0) else 1.0
                w_list.append(rem / max(pos_count, 1))
        scale_weights = torch.tensor(w_list, dtype=torch.float32, device=device)

    m_bin = (masks > 0.5).float()
    orig_mass = m_bin.sum(dim=spatial_dims).clamp_min(1e-6)

    rcc8_accum = torch.zeros((B, N, N, 8), dtype=torch.float32, device=device)
    obj_survival = torch.zeros((B, N), dtype=torch.float32, device=device)

    for w_s, r in zip(scale_weights, radii):
        flat_in = m_bin.reshape(B * N, 1, *masks.shape[2:])
        filtered = morph_open_clean(flat_in, radius=r, mode=mode)
        if close_pinholes and r > 0:
            filtered = morph_close_clean(filtered, radius=r, mode=mode)
        filtered = (filtered > 0.5).float().reshape(masks.shape)

        surv_mass = filtered.sum(dim=spatial_dims)
        surv_ratio = (surv_mass / orig_mass).clamp(0.0, 1.0)
        obj_survival = obj_survival + w_s * surv_ratio

        pair_survival = surv_ratio.unsqueeze(2) * surv_ratio.unsqueeze(1)
        both_originally_empty = m_bin.sum(dim=spatial_dims) < 0.5
        if both_originally_empty.any():
            pair_survival = torch.where(
                both_originally_empty.unsqueeze(2) | both_originally_empty.unsqueeze(1),
                torch.ones_like(pair_survival),
                pair_survival,
            )

        rcc_r = compute_rcc8_matrix(filtered, kernel_size=kernel_size)
        one_hot = F.one_hot(rcc_r.long(), num_classes=8).float()
        rcc8_accum = rcc8_accum + (w_s * pair_survival).unsqueeze(-1) * one_hot

    total_vote = rcc8_accum.sum(dim=-1, keepdim=True)
    dc_fallback = torch.zeros_like(rcc8_accum)
    dc_fallback[..., 0] = 1.0
    rcc8_probs = torch.where(total_vote > 1e-6, rcc8_accum / total_vote.clamp_min(1e-6), dc_fallback)
    dominant_rcc8 = rcc8_probs.argmax(dim=-1)

    if is_unbatched:
        return dominant_rcc8.squeeze(0), rcc8_probs.squeeze(0), obj_survival.squeeze(0)
    return dominant_rcc8, rcc8_probs, obj_survival

class PersistentAnchoredRCC8Loss(nn.Module):
    def __init__(
        self,
        filtration_radii: Sequence[int] = (1, 2),
        kernel_size: int = 3,
        border_value: float = 0.0,
        target_relation: str = "DC",
        weight_pred_persistence: bool = True,
        mode: str = "isotropic",
        reduction: str = "mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if target_relation not in ("PO", "DC", "EC", "NTPP", "TPP"):
            raise ValueError(f"Unsupported target_relation: {target_relation}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.filtration_radii = tuple(sorted(int(r) for r in filtration_radii if r >= 0))
        self.kernel_size = kernel_size
        self.border_value = border_value
        self.target_relation = target_relation
        self.weight_pred_persistence = weight_pred_persistence
        self.mode = mode
        self.reduction = reduction
        self.eps = eps

    @torch.no_grad()
    def _build_clean_topo_zone(self, anchor_zone: torch.Tensor) -> torch.Tensor:
        a_clean = granulometric_persistence(
            torch.clamp(anchor_zone.detach(), 0.0, 1.0),
            filtration_radii=self.filtration_radii,
            close_pinholes=True,
            mode=self.mode,
        )
        if self.target_relation == "PO":
            ero_a = morph_erosion(a_clean, self.kernel_size, border_value=self.border_value)
            return 0.5 * (a_clean + ero_a)
        if self.target_relation == "DC":
            return morph_dilation(a_clean, self.kernel_size)
        if self.target_relation == "EC":
            dil_a = morph_dilation(a_clean, self.kernel_size)
            return 1.0 - F.relu(dil_a - a_clean)
        if self.target_relation == "NTPP":
            ero_a = morph_erosion(a_clean, self.kernel_size, border_value=self.border_value)
            return 1.0 - ero_a
        ero_a = morph_erosion(a_clean, self.kernel_size, border_value=self.border_value)
        return (1.0 - a_clean) + 0.5 * ero_a

    def forward(self, anchor_zone: torch.Tensor, pred_prob: torch.Tensor) -> torch.Tensor:
        if anchor_zone.ndim == 3:
            anchor_zone = anchor_zone.unsqueeze(1)
        if pred_prob.ndim == 3:
            pred_prob = pred_prob.unsqueeze(1)
        if anchor_zone.shape[2:] != pred_prob.shape[2:]:
            raise ValueError(
                f"Spatial shape mismatch: anchor_zone {tuple(anchor_zone.shape[2:])} "
                f"vs pred_prob {tuple(pred_prob.shape[2:])}."
            )

        p_clamped = torch.clamp(pred_prob, 0.0, 1.0)
        topo_zone = self._build_clean_topo_zone(anchor_zone)

        if self.weight_pred_persistence:
            with torch.no_grad():
                pred_pers = granulometric_persistence(
                    p_clamped.detach(),
                    filtration_radii=self.filtration_radii,
                    close_pinholes=False,
                    mode=self.mode,
                )
                eff_zone = topo_zone * (0.5 + 0.5 * pred_pers)
        else:
            eff_zone = topo_zone

        sum_dims = (-3, -2, -1) if eff_zone.ndim == 5 else (-2, -1)
        denom = topo_zone.sum(dim=sum_dims).clamp_min(self.eps)
        loss_per_sample = (eff_zone * p_clamped).sum(dim=sum_dims) / denom

        if self.reduction == "mean":
            return loss_per_sample.mean()
        if self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample

