import pathlib
import numpy as np
import torch

from torch_rcc8 import kuratowski_opening, classify_rcc8_grid_jepd, compute_rcc8_matrix, DifferentiableFuzzyRCC8Loss, TopoRCC8Transformer, ConformalRiskCalibrator

def test_tier1_kuratowski_and_rcc8_jepd_axioms():
    rng = np.random.default_rng(42)
    valid_relations = {"DC", "EC", "PO", "EQ", "TPP", "NTPP", "TPPi", "NTPPi"}
    for _ in range(30):
        A = rng.random((32, 32)) > 0.65
        B = rng.random((32, 32)) > 0.65
        op_A = kuratowski_opening(A)
        assert np.array_equal(kuratowski_opening(op_A), op_A)
        assert classify_rcc8_grid_jepd(A, B) in valid_relations

    masks = torch.zeros((3, 16, 16))
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 2:6, 6:10] = 1.0
    masks[2, 10:14, 10:14] = 1.0
    mat = compute_rcc8_matrix(masks)
    assert int(mat[0, 1].item()) == 1
    assert int(mat[0, 2].item()) == 0

def test_tier1_bloch_fuzzy_rcc8_all_8_relations_gradcheck():
    """Verifies analytical gradients across all 8 fuzzy RCC-8 relations and reduction modes."""
    torch.manual_seed(42)
    loss_fn = DifferentiableFuzzyRCC8Loss(kernel_size=3, border_value=0.0, reduction="mean")
    mu = torch.sigmoid(torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True))
    nu = torch.sigmoid(torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True))

    def func(a, b):
        out = loss_fn(a, b)
        return sum(out.values())

    assert torch.autograd.gradcheck(func, (mu, nu), eps=1e-6, atol=1e-4)

    loss_none = DifferentiableFuzzyRCC8Loss(kernel_size=3, reduction="none")(mu, nu)
    assert set(loss_none.keys()) == {
        "fuzzy_DC", "fuzzy_EC", "fuzzy_PO", "fuzzy_TPP",
        "fuzzy_NTPP", "fuzzy_TPPi", "fuzzy_NTPPi", "fuzzy_EQ"
    }
    assert loss_none["fuzzy_EC"].shape == (1, 1)

