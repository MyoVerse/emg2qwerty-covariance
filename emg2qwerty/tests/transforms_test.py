import numpy as np
import pytest
import torch

from emg2qwerty.transforms import (
    bind_session,
    CausalSessionRecentering,
    CausalZNormalizeTime,
    Compose,
    ForEach,
    log_cholesky_mean,
    SessionRecentering,
    SlidingCovariance,
    SPDAugment,
    sym_inv_sqrt,
    ZNormalizeTime,
)


def random_emg(T: int = 1000, bands: int = 2, C: int = 16, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(T, bands, C, generator=generator)


def batch_log_cholesky_mean(cov: torch.Tensor) -> torch.Tensor:
    """Log-Cholesky Fréchet mean over the frames of ``cov`` (T, ..., C, C),
    as computed by scripts/compute_frechet_means.py."""
    L = torch.linalg.cholesky(cov.double())
    diag = L.diagonal(dim1=-2, dim2=-1)
    lower = L - torch.diag_embed(diag)
    return log_cholesky_mean(lower.sum(0), diag.log().sum(0), len(cov))


def assert_spd(x: torch.Tensor) -> None:
    assert torch.allclose(x, x.transpose(-1, -2), atol=1e-6)
    assert torch.linalg.eigvalsh(x).min() > 0


def test_sliding_covariance():
    emg = random_emg()
    cov = SlidingCovariance(window_length=64, hop_length=16, shrinkage=0.0, eps=0.0)
    out = cov(emg)
    assert out.shape == (59, 2, 16, 16)  # (1000 - 64) // 16 + 1 frames

    # Frame 3 of band 1 covers samples [48, 112)
    window = emg[48:112, 1]
    assert torch.allclose(out[3, 1], window.T @ window / 64, atol=1e-5)

    assert_spd(SlidingCovariance()(emg))


def test_sliding_covariance_is_causal():
    emg = random_emg()
    cov = SlidingCovariance()
    full = cov(emg)
    altered = cov(torch.cat([emg[:500], random_emg(500, seed=1)]))
    num_frames = (500 - 64) // 16 + 1  # Frames entirely within the first 500 samples
    assert torch.equal(full[:num_frames], altered[:num_frames])


def test_causal_znormalize_time():
    x = random_emg(T=50)
    eps = 1e-5
    out = CausalZNormalizeTime(eps=eps)(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    for t in range(len(x)):
        past = x[: t + 1].double()
        expected = (x[t].double() - past.mean(0)) / (past.std(0, unbiased=False) + eps)
        assert torch.allclose(out[t].double(), expected, atol=1e-4)


def test_causal_znormalize_time_matches_offline_at_end():
    x = random_emg(T=2000)
    offline = ZNormalizeTime(eps=0.0)(x)
    causal = CausalZNormalizeTime(eps=0.0)(x)
    # Offline uses the unbiased std, the running estimate the biased one
    scale = np.sqrt(len(x) / (len(x) - 1))
    assert torch.allclose(causal[-1], offline[-1] * scale, atol=1e-3)


def test_causal_znormalize_time_warmup():
    x = random_emg(T=50)
    warmup = 10
    out = CausalZNormalizeTime(warmup=warmup, eps=0.0)(x)
    head = x[:warmup].double()
    expected = (head - head.mean(0)) / head.std(0, unbiased=False)
    assert torch.allclose(out[:warmup].double(), expected, atol=1e-4)
    assert torch.allclose(
        out[warmup:], CausalZNormalizeTime(eps=0.0)(x)[warmup:], atol=1e-4
    )


def test_causal_session_recentering():
    cov = SlidingCovariance()(random_emg(T=800))
    out = CausalSessionRecentering()(cov)
    assert out.shape == cov.shape and out.dtype == cov.dtype
    assert_spd(out)

    # The first frame is its own mean and so maps to the identity
    assert torch.allclose(out[0], torch.eye(16).expand(2, 16, 16), atol=1e-4)

    # The last frame is recentered at the full-session mean
    inv_sqrt = sym_inv_sqrt(batch_log_cholesky_mean(cov))
    expected = inv_sqrt @ cov[-1].double() @ inv_sqrt
    assert torch.allclose(out[-1].double(), expected, atol=1e-4)

    # Causal: a prefix of the input yields a prefix of the output
    assert torch.allclose(CausalSessionRecentering()(cov[:20]), out[:20], atol=1e-5)


def test_causal_session_recentering_warmup():
    cov = SlidingCovariance()(random_emg(T=800))
    warmup = 5
    out = CausalSessionRecentering(warmup=warmup)(cov)
    inv_sqrt = sym_inv_sqrt(batch_log_cholesky_mean(cov[:warmup]))
    expected = inv_sqrt @ cov[:warmup].double() @ inv_sqrt
    assert torch.allclose(out[:warmup].double(), expected, atol=1e-4)
    assert torch.allclose(
        out[warmup:], CausalSessionRecentering()(cov)[warmup:], atol=1e-5
    )


def test_session_recentering(tmp_path):
    cov = SlidingCovariance()(random_emg(T=800))
    means_path = tmp_path.joinpath("frechet_means.pt")
    inv_sqrt = sym_inv_sqrt(batch_log_cholesky_mean(cov)).float()
    torch.save({"inv_sqrt": {"session_a": inv_sqrt}}, means_path)

    unbound = SessionRecentering(means_path=str(means_path))
    with pytest.raises(AssertionError):
        unbound(cov)

    out = bind_session(unbound, "session_a")(cov)
    assert_spd(out)
    # Agrees with the causal running mean at the end of the session
    assert torch.allclose(out[-1], CausalSessionRecentering()(cov)[-1], atol=1e-3)

    with pytest.raises(AssertionError):
        bind_session(unbound, "session_b")(cov)


def test_bind_session():
    transform = Compose(
        [ZNormalizeTime(), ForEach(SessionRecentering(means_path="means.pt"))]
    )
    bound = bind_session(transform, "session_a")
    assert bound.transforms[1].transform.session == "session_a"
    assert transform.transforms[1].transform.session is None  # Original untouched
    assert bound.transforms[0] is transform.transforms[0]  # Shared


def test_spd_augment():
    cov = CausalSessionRecentering()(SlidingCovariance()(random_emg(T=800)))
    assert torch.equal(SPDAugment()(cov), cov)

    np.random.seed(0)
    augment = SPDAugment(
        n_time_masks=3, time_mask_param=10, n_electrode_masks=2, electrode_mask_param=3
    )
    out = augment(cov)
    assert out.shape == cov.shape
    assert_spd(out)
    # Every entry is either unchanged or that of the identity
    eye = torch.eye(16).expand_as(cov)
    assert torch.all((out == cov) | (out == eye))
    assert not torch.equal(out, cov)
