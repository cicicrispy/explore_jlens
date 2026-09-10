import torch

from jlens_spec import cka as cka_mod
from jlens_spec.lens import Lens


def test_cka_diagonal_is_one_and_matrix_symmetric(standin_model, random_lens):
    C, layers = cka_mod.cka_matrix(standin_model, random_lens, n_tokens=64, seed=0)
    assert torch.allclose(torch.diagonal(C), torch.ones(len(layers)), atol=1e-5)
    assert torch.allclose(C, C.T, atol=1e-5)


def test_cka_invariant_to_shared_orthogonal_transform(standin_model, random_lens):
    d = random_lens.d_model
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    J2 = {l: J @ Q for l, J in random_lens.J.items()}
    lens2 = Lens(
        J=J2, d_model=d, meta={}, layers=random_lens.layers, dtype=torch.float32, file_size_bytes=0
    )
    C1, _ = cka_mod.cka_matrix(standin_model, random_lens, n_tokens=64, seed=0)
    C2, _ = cka_mod.cka_matrix(standin_model, lens2, n_tokens=64, seed=0)
    assert torch.allclose(C1, C2, atol=1e-3)


def test_cka_off_diagonal_roughly_uniform_on_random_lens(standin_model, random_lens):
    """Spec: on a random lens the off-diagonal is roughly uniform, so block structure on the real
    lens can't be a plotting artifact. The LEVEL is not meaningful: a random square Gaussian J
    roughly preserves the geometry of W_U rows (J J^T ~ d I), so linear CKA between any two random
    layers is near 1. What must hold is that no pair stands out -- i.e. a small spread."""
    C, layers = cka_mod.cka_matrix(standin_model, random_lens, n_tokens=64, seed=0)
    n = len(layers)
    off = C[~torch.eye(n, dtype=torch.bool)]
    print(f"random-lens off-diagonal CKA: mean={float(off.mean()):.4f} std={float(off.std()):.4f}")
    assert float(off.std()) < 0.05
    assert float(off.max() - off.min()) < 0.2
