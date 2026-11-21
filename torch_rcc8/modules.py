r"""
torch_rcc8/modules.py
---------------------------
Vectorized GPU Matrix-Multiplication RCC-8 External Connection (EC) Adjacency & Higra Token Extractor.
"""

import time
import numpy as np
try:
    pass
except ImportError:
    hg = None
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_rcc8_ec_matrix(token_masks: list | torch.Tensor, dilation_iter: int = 2) -> np.ndarray | torch.Tensor:
    r"""
    Vectorized PyTorch Tensor RCC8-EC Adjacency Matrix following Bloch (2005) Eq. (35).
    Uses float64 inner products with < 0.5 tolerance to prevent TF32/FP32 precision issues.
    """
    is_tensor = isinstance(token_masks, torch.Tensor)
    if len(token_masks) == 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=token_masks.device) if is_tensor else np.zeros((0, 0), dtype=np.float32)

    if is_tensor:
        masks_t = (token_masks > 0.5).float()
    else:
        masks_t = torch.from_numpy(np.stack([np.asarray(m, dtype=np.float32) > 0.5 for m in token_masks], axis=0)).float()

    k_size = 2 * dilation_iter + 1
    dilated = F.max_pool2d(masks_t.unsqueeze(1), kernel_size=k_size, stride=1, padding=dilation_iter).squeeze(1)

    flat_dil = dilated.flatten(1).to(torch.float64)
    flat_orig = masks_t.flatten(1).to(torch.float64)
    disjoint = (torch.matmul(flat_orig, flat_orig.T) < 0.5)
    touch = (torch.matmul(flat_dil, flat_orig.T) > 0.5) | (torch.matmul(flat_orig, flat_dil.T) > 0.5)
    ec_mat = (disjoint & touch).float()
    ec_mat.fill_diagonal_(0.0)
    return ec_mat if is_tensor else ec_mat.cpu().numpy()

class TopoRCC8Transformer(nn.Module):
    """Object-Centric Topological Token & RCC8 Relational Self-Attention Module."""
    def __init__(self, d_model: int = 16):
        super().__init__()
        self.learnable_tau = nn.Parameter(torch.full((5,), 0.30))
        self.encoder = nn.Linear(5, d_model)
        self.q_lin = nn.Linear(d_model, d_model, bias=False)
        self.k_lin = nn.Linear(d_model, d_model, bias=False)
        self.v_lin = nn.Linear(d_model, d_model, bias=False)
        self.rcc8_ec_gain = nn.Parameter(torch.tensor(3.2))
        self.rcc8_dc_mask = nn.Parameter(torch.tensor(-3.5))
        self.head = nn.Linear(d_model, 1)

    def forward(self, topo_feats: torch.Tensor, rcc8_ec_mat: torch.Tensor):
        if topo_feats.shape[0] == 0:
            return torch.zeros((0,), dtype=topo_feats.dtype, device=topo_feats.device), rcc8_ec_mat
        predicates = torch.sigmoid(10.0 * (topo_feats - self.learnable_tau))
        h = F.silu(self.encoder(predicates))
        Q, K, V = self.q_lin(h), self.k_lin(h), self.v_lin(h)
        scores = torch.matmul(Q, K.T) / np.sqrt(h.shape[-1])

        eye = torch.eye(len(topo_feats), device=topo_feats.device)
        rel_bias = eye * 2.0 + rcc8_ec_mat * self.rcc8_ec_gain + (1.0 - eye - rcc8_ec_mat) * self.rcc8_dc_mask
        attn = F.softmax(scores + rel_bias, dim=-1)

        h_ctx = h + torch.matmul(attn, V)
        omega = torch.sigmoid(self.head(h_ctx)).squeeze(-1)
        return omega, attn
