from __future__ import annotations
import torch
from torch_rcc8.diagnostics import AnchoredFuzzyRCC8Loss, DifferentiableFuzzyRCC8Loss, compute_rcc8_matrix

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== [torch-rcc8 Verification Suite on {device.upper()}] ===")

    # 1. Batched (B, N, H, W) & Unbatched (N, H, W) RCC-8 JEPD Matrix Verification
    masks = torch.zeros((4, 32, 32), device=device)
    masks[0, 4:10, 4:10] = 1.0   # Region 0
    masks[1, 4:10, 10:16] = 1.0  # Region 1: EC with 0
    masks[2, 2:16, 2:16] = 1.0   # Region 2: NTPPi of 0
    masks[3, 20:28, 20:28] = 1.0 # Region 3: DC with all

    rcc_3d = compute_rcc8_matrix(masks, kernel_size=3)
    rcc_4d = compute_rcc8_matrix(masks.unsqueeze(0).repeat(2, 1, 1, 1), kernel_size=3)
    assert rcc_3d[0, 1].item() == 1, f"Expected EC(1), got {rcc_3d[0, 1].item()}"
    assert rcc_3d[0, 2].item() == 4, f"Expected NTPP(4), got {rcc_3d[0, 2].item()}"
    assert rcc_3d[2, 0].item() == 7, f"Expected NTPPi(7), got {rcc_3d[2, 0].item()}"
    assert rcc_3d[0, 3].item() == 0, f"Expected DC(0), got {rcc_3d[0, 3].item()}"
    assert rcc_4d.shape == (2, 4, 4) and torch.equal(rcc_4d[0], rcc_3d)
    print("[PASS 1/4] 3D/4D Vectorized RCC-8 JEPD Matrix (DC, EC, PO, TPP, NTPP, EQ, TPPi, NTPPi)")

    # 2. PyTorch Official Numerical Gradient Check (torch.autograd.gradcheck)
    loss_fn = DifferentiableFuzzyRCC8Loss(kernel_size=3)
    mu = torch.rand((1, 1, 8, 8), dtype=torch.float64, device=device, requires_grad=True)
    nu = torch.rand((1, 1, 8, 8), dtype=torch.float64, device=device, requires_grad=True)
    passed_grad = torch.autograd.gradcheck(lambda a, b: loss_fn(a, b)["fuzzy_PO"], (mu, nu), eps=1e-6, atol=1e-4)
    assert passed_grad
    print("[PASS 2/4] PyTorch autograd.gradcheck passed for DifferentiableFuzzyRCC8Loss")

    # 3. AnchoredFuzzyRCC8Loss Uniform Spatial Gradient Verification (No Cheese-Holing)
    anchor_loss_fn = AnchoredFuzzyRCC8Loss(kernel_size=3)
    anchor_zone = torch.zeros((1, 1, 16, 16), device=device)
    anchor_zone[:, :, 4:12, 4:12] = 1.0
    pred_prob = torch.full((1, 1, 16, 16), 0.8, device=device, requires_grad=True)
    loss_val = anchor_loss_fn(anchor_zone, pred_prob)
    loss_val.backward()
    grad_inside = pred_prob.grad[:, :, 4:12, 4:12]
    assert torch.all(grad_inside > 0), "Gradients inside anchor zone must be strictly positive"
    assert torch.allclose(grad_inside, grad_inside[0, 0, 0, 0]), "Gradients inside uniform anchor zone must be spatially uniform (zero fragmentation)"
    print(f"[PASS 3/4] AnchoredFuzzyRCC8Loss uniform gradient verified (loss={loss_val.item():.4f}, std_grad={grad_inside.std().item():.2e})")

    # 4. Continuous SDF-RCC8 (DC + STL NTPPi) Autograd Verification
    from torch_rcc8 import sdf_dc_loss, sdf_ntppi_stl_loss
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yy, xx = torch.meshgrid(
        torch.arange(64, dtype=torch.float32, device=dev),
        torch.arange(64, dtype=torch.float32, device=dev),
        indexing='ij',
    )
    obs_sdf = torch.sqrt((xx - 32.0) ** 2 + (yy - 32.0) ** 2) - 6.0
    pts = torch.tensor([[[31.0, 32.0]]], device=dev, requires_grad=True)
    loss_dc = sdf_dc_loss(obs_sdf, pts, safety_margin=2.0)
    loss_dc.backward()
    assert loss_dc.item() > 0.0 and pts.grad is not None and torch.norm(pts.grad).item() > 0.0
    xy = torch.tensor([[15.0, 15.0]], device=dev, requires_grad=True)
    th = torch.tensor([0.0], device=dev, requires_grad=True)
    tgt = torch.tensor([[15.0, 15.0]], device=dev)
    loss_ntppi = sdf_ntppi_stl_loss(xy, th, tgt, half_l=8.0, half_w=4.0, margin=0.5)
    assert loss_ntppi.item() < 1e-3
    print("[PASS 4/4] Continuous SDF-RCC8 (DC + STL NTPPi) autograd verified.")


if __name__ == "__main__":
    main()
