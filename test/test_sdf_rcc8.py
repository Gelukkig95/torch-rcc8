import torch
from torch_rcc8 import sdf_dc_loss, sdf_ntppi_stl_loss

def test_continuous_sdf_rcc8():
    yy, xx = torch.meshgrid(torch.arange(64, dtype=torch.float32), torch.arange(64, dtype=torch.float32), indexing="ij")
    obs_sdf = torch.sqrt((xx - 32.0) ** 2 + (yy - 32.0) ** 2) - 8.0
    pts = torch.tensor([[[32.0, 32.0], [34.0, 32.0]]], requires_grad=True)
    l1 = sdf_dc_loss(obs_sdf, pts, safety_margin=2.0)
    l1.backward()
    assert l1.item() > 0.0 and pts.grad is not None and torch.norm(pts.grad).item() > 0.0
    xy = torch.tensor([[10.0, 10.0], [20.0, 20.0]], requires_grad=True)
    th = torch.tensor([0.0, 0.0], requires_grad=True)
    l2 = sdf_ntppi_stl_loss(xy, th, torch.tensor([[19.0, 20.0], [21.0, 20.0]]), half_l=10.0, half_w=5.0, margin=0.5)
    assert l2.item() < 1e-3
