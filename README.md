# torch-rcc8: Differentiable Region Connection Calculus (RCC-8) in PyTorch

`torch-rcc8` is a pure PyTorch library that brings qualitative spatial reasoning (Region Connection Calculus, RCC-8) into deep learning `autograd`, supporting both **discrete raster masks** and **continuous Signed Distance Fields (SDF)**.

## 1. Motivation

1. **Qualitative Spatial Reasoning Deficit:** Cohn & Blackwell (2024) and Blackwell & Cohn (2026) showed that foundation models struggle with raster RCC-8 diagrams, boundary contacts (`EC`, `TPP`, `TPPi`), and asymmetric inverse containment (`NTPPi`).
2. **Subgradient Fragmentation in Raster Fuzzy RCC-8:** Directly backpropagating classical fuzzy morphological equations (Bloch, 2005) through `max_pool2d` concentrates gradients onto isolated local maxima, causing mask fragmentation ("cheese-holing") and denominator instability.
3. **Zero-Gradient Dead Zones in Free Space:** Mask-overlap losses vanish when two regions are disconnected (`DC`). Lifting `DC` and `NTPPi` onto continuous SDFs provides non-vanishing spatial gradients for rigid-body `SE(2)` optimization and trajectory planning.

## 2. Key Contributions

* **Vectorized 3D/4D Raster RCC-8 Matrix (`compute_rcc8_matrix`):** Computes all 8 pairwise RCC-8 relations (`DC`, `EC`, `PO`, `TPP`, `NTPP`, `EQ`, `TPPi`, `NTPPi`) for `(N, H, W)` and `(B, N, H, W)` tensors via `F.max_pool2d` and `torch.matmul`.
* **Differentiable & Gradient-Stabilized Fuzzy RCC-8 (`DifferentiableFuzzyRCC8Loss`, `AnchoredFuzzyRCC8Loss`):** Implements Bloch (2005) fuzzy RCC-8 relations in PyTorch and resolves CNN `max_pool2d` subgradient fragmentation via detached morphological anchors.
* **Continuous SDF-RCC8 & STL (`sdf_dc_loss`, `sdf_ntppi_stl_loss`):** Differentiable separation (`DC`) and smooth temporal containment (`NTPPi`) objectives on continuous SDFs, compatible with both gradient-based (`Adam`) and sampling-based (`MPPI`) solvers.

## 3. Empirical Validation

### 512x512 Segmentation Ablation (Raster Fuzzy RCC-8)

| Method (`512x512` U-Net, Identical Schedule) | Target IoU | Precision | Recall | False-Positive Blobs (`>=6px`) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Baseline U-Net** (`CE + Dice`) | 0.6518 | 0.7992 | 0.7794 | 2,549 |
| **2. Naive Fuzzy RCC-8** (Direct `max_pool2d` grad) | 0.4283 | 0.7616 | 0.4945 | 7,992 (+213.5%) |
| **3. AnchoredFuzzyRCC8Loss** (**Proposed**) | **0.6566** | **0.8102** | 0.7759 | **2,309 (-9.4%)** |

### PhenoBench `SE(2)` Planning Benchmark (N=632 Real Scenes / N=536 Feasible)

| Method | Full STL (All) | Full STL (Feasible) | Zero-Coll (All) | Latency |
| :--- | :---: | :---: | :---: | :---: |
| Standard `SE(2)` RRT* (CPU) | 54.0% | 63.6% | 92.4% | 56.6 ms |
| `[Adam B=5]` Plain SDF + STL | 47.3% | 55.8% | 52.8% | 1456.0 ms |
| `[Adam B=5]` **SDF-RCC8 (NTPPi) + STL** | **60.4%** | **71.3%** | 63.4% | 1666.8 ms |
| `[MPPI K=384]` Plain SDF (Zero-Init) | 62.7% | 73.9% | 96.0% | 192.0 ms |
| `[MPPI K=384]` **SDF-RCC8 (Zero-Init)** | **72.2%** | **85.1%** | 95.9% | 222.2 ms |
| `[Hybrid CPU]` **RRT* + SDF-RCC8** | 66.8% | 78.7% | **97.2%** | 199.2 ms |

## 4. Quickstart

```bash
pip install -e .
```

```python
import torch
from torch_rcc8 import (
    compute_rcc8_matrix,
    DifferentiableFuzzyRCC8Loss,
    AnchoredFuzzyRCC8Loss,
    sdf_dc_loss,
    sdf_ntppi_stl_loss,
)

# 1. Batched Pairwise RCC-8 Matrix on Raster Masks (B, N, H, W)
masks = torch.randint(0, 2, (2, 5, 64, 64), dtype=torch.float32)
rcc_matrix = compute_rcc8_matrix(masks, kernel_size=3)

# 2. Gradient-Stabilized Repulsion Loss for Segmentation
criterion = AnchoredFuzzyRCC8Loss(kernel_size=3, reduction="mean")
anchor_zone = torch.rand(4, 1, 128, 128)
pred_prob = torch.rand(4, 1, 128, 128, requires_grad=True)
criterion(anchor_zone, pred_prob).backward()

# 3. Continuous SDF-RCC8 + STL Containment Objective
poses_xy = torch.tensor([[10.0, 10.0], [20.0, 20.0]], requires_grad=True)
poses_th = torch.tensor([0.0, 0.2], requires_grad=True)
target_pts = torch.tensor([[19.0, 20.0], [21.0, 20.0]])
sdf_ntppi_stl_loss(poses_xy, poses_th, target_pts, half_l=10.0, half_w=5.0).backward()
```

## 5. Verification

```bash
pytest test
python verify_torch_rcc8.py
```

## 6. References

* Randell, D. A., Cui, Z., & Cohn, A. G. (1992). A spatial logic based on regions and connection. *KR*, 165-176.
* Bloch, I. (2005). Fuzzy spatial relationships for image processing and interpretation: a review. *Image and Vision Computing*, 23(2), 89-110.
* Cohn, A. G., & Blackwell, R. E. (2024). Can Large Language Models reason about the Region Connection Calculus? *arXiv:2411.19589*.
* Blackwell, R. E., & Cohn, A. G. (2026). RCC-8 as a benchmark for diagrammatic reasoning in multimodal foundation models. *LIPIcs, COSIT 2026*, 4:1-4:20.
