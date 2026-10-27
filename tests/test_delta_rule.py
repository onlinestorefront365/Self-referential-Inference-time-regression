"""SRWM and DeltaNet against naive per-stream, per-head reference implementations."""

from collections.abc import Iterator

import pytest
import torch
from torch import Tensor

from srwm.layers import DeltaNetLayer, SRWMLayer


@pytest.fixture(autouse=True)
def _float64() -> Iterator[None]:
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _srwm_reference(layer: SRWMLayer, x: Tensor, W: Tensor) -> tuple[Tensor, Tensor]:
    d = layer.d_head
    ys, W_new = [], W.clone()
    for b in range(x.shape[0]):
        heads = []
        for h in range(layer.n_heads):
            Wbh = W[b, h]
            out = Wbh @ x[b, h * d : (h + 1) * d]
            y, k, q, beta = out[:d], out[d : 2 * d], out[2 * d : 3 * d], out[3 * d :]
            phi_k, phi_q = torch.softmax(k, 0), torch.softmax(q, 0)
            diff = Wbh @ phi_q - Wbh @ phi_k
            lr = torch.cat([torch.sigmoid(beta[i]).repeat(n) for i, n in enumerate([d, d, d, 4])])
            W_new[b, h] = Wbh + torch.outer(lr * diff, phi_k)
            heads.append(y)
        ys.append(torch.cat(heads))
    return layer.out(torch.stack(ys)), W_new


def _deltanet_reference(layer: DeltaNetLayer, x: Tensor, W: Tensor) -> tuple[Tensor, Tensor]:
    d = layer.d_head
    proj = layer.slow(x)
    ys, W_new = [], W.clone()
    for b in range(x.shape[0]):
        heads = []
        for h in range(layer.n_heads):
            p = proj[b, h * (3 * d + 1) : (h + 1) * (3 * d + 1)]
            q, k, v, beta = p[:d], p[d : 2 * d], p[2 * d : 3 * d], p[3 * d]
            phi_q, phi_k = torch.softmax(q, 0), torch.softmax(k, 0)
            Wbh = W[b, h] + torch.sigmoid(beta) * torch.outer(v - W[b, h] @ phi_k, phi_k)
            W_new[b, h] = Wbh
            heads.append(Wbh @ phi_q)
        ys.append(torch.cat(heads))
    return layer.out(torch.stack(ys)), W_new


@pytest.mark.parametrize("n_heads", [1, 2])
def test_srwm_matches_reference(n_heads: int) -> None:
    torch.manual_seed(0)
    layer = SRWMLayer(d_model=6, n_heads=n_heads)
    s = layer.init_state(3)
    for _ in range(4):
        x = torch.randn(3, 6)
        y_ref, W_ref = _srwm_reference(layer, x, s.W)
        y, s = layer.step(x, s)
        torch.testing.assert_close(y, y_ref)
        torch.testing.assert_close(s.W, W_ref)


@pytest.mark.parametrize("n_heads", [1, 2])
def test_deltanet_matches_reference(n_heads: int) -> None:
    torch.manual_seed(0)
    layer = DeltaNetLayer(d_model=6, n_heads=n_heads)
    with torch.no_grad():
        layer.W0.normal_()
    s = layer.init_state(3)
    for _ in range(4):
        x = torch.randn(3, 6)
        y_ref, W_ref = _deltanet_reference(layer, x, s.W)
        y, s = layer.step(x, s)
        torch.testing.assert_close(y, y_ref)
        torch.testing.assert_close(s.W, W_ref)


def test_srwm_modifies_itself() -> None:
    torch.manual_seed(0)
    layer = SRWMLayer(d_model=4, n_heads=2)
    s0 = layer.init_state(2)
    _, s = layer.scan(torch.randn(2, 5, 4), s0)
    assert not torch.allclose(s.W, s0.W)


def test_srwm_frozen_keeps_initial_matrix() -> None:
    torch.manual_seed(0)
    layer = SRWMLayer(d_model=4, n_heads=2, self_modify=False)
    s0 = layer.init_state(2)
    _, s = layer.scan(torch.randn(2, 5, 4), s0)
    assert torch.equal(s.W, s0.W)


def test_delta_rule_shrinks_retrieval_error() -> None:
    """One update scales the error W φ(k) − v by exactly (1 − σ(β) ‖φ(k)‖²).

    Softmax keys have ‖φ(k)‖ < 1, so a single step never stores v exactly; the
    model has to learn how sharp to make its keys and how large β should be.
    """
    torch.manual_seed(0)
    layer = DeltaNetLayer(d_model=4, n_heads=1)
    d = layer.d_head
    x = torch.randn(1, 4)
    with torch.no_grad():
        layer.W0.normal_()
    s0 = layer.init_state(1)
    _, s1 = layer.step(x, s0)
    q, k, v, beta = layer.slow(x)[0].split([d, d, d, 1])
    phi_k = torch.softmax(k, 0)
    err_before = s0.W[0, 0] @ phi_k - v
    err_after = s1.W[0, 0] @ phi_k - v
    factor = 1 - torch.sigmoid(beta) * phi_k.square().sum()
    torch.testing.assert_close(err_after, factor * err_before)


@pytest.mark.parametrize("cls", [SRWMLayer, DeltaNetLayer])
def test_heads_must_divide_d_model(cls: type) -> None:
    with pytest.raises(ValueError, match="divisible"):
        cls(d_model=5, n_heads=2)
