# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)


@dataclass
class ZNormalizeTime:
    """Z-normalizes each channel along the time dimension (dim 0) using the
    statistics of the entire input. Since transforms are applied per dataset
    item, each training/validation window (including its padding) is
    normalized independently. The input must be of shape (T, ...).

    Not causal: every sample is normalized with statistics that include
    future samples. Use ``CausalZNormalizeTime`` at test time.

    Args:
        eps (float): Added to the standard deviation for numerical stability.
            (default: 1e-5)
    """

    eps: float = 1e-5

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True)
        return (tensor - mean) / (std + self.eps)


@dataclass
class CausalZNormalizeTime:
    """Causal counterpart of ``ZNormalizeTime``: each sample is z-normalized
    per channel with the running (expanding-window) mean and standard
    deviation of all samples up to and including itself, so no future samples
    are used. The input must be of shape (T, ...).

    At the end of a session the running statistics coincide with those of
    ``ZNormalizeTime`` (up to the biased/unbiased variance factor), so the two
    transforms agree asymptotically; they differ early in the session and
    under amplitude drift.

    Args:
        warmup (int): Samples at the start use the statistics of the first
            ``warmup`` samples (a lookahead of ``warmup`` samples only there);
            0 is strictly causal. (default: 0)
        eps (float): Added to the standard deviation. (default: 1e-5)
    """

    warmup: int = 0
    eps: float = 1e-5

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.to(torch.float64)
        T = x.shape[0]
        t = torch.arange(1, T + 1, dtype=torch.float64).view(T, *([1] * (x.ndim - 1)))
        mean = x.cumsum(0) / t
        var = ((x * x).cumsum(0) / t - mean * mean).clamp(min=0.0)
        std = var.sqrt()
        w = min(self.warmup, T)
        if w > 1:
            mean[:w] = mean[w - 1]
            std[:w] = std[w - 1]
        return ((x - mean) / (std + self.eps)).to(tensor.dtype)


@dataclass
class SlidingCovariance:
    """Computes per-band spatial covariance matrices of the EMG signal over
    sliding windows. Windows are aligned with ``LogSpectrogram`` frames
    (same ``window_length``/``hop_length`` as ``n_fft``/``hop_length``), so the
    output has the same number of timesteps as the spectrogram would.
    The input must be of shape (T, ..., C) and the output is of shape
    (T', ..., C, C). Each output frame only depends on the ``window_length``
    samples ending at it, so the transform is causal.

    Args:
        window_length (int): Number of raw EMG samples per window. (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            windows. (default: 16)
        shrinkage (float): Shrinks each covariance towards a scaled identity,
            ``(1 - shrinkage) * cov + shrinkage * tr(cov) / C * I``, to keep
            the matrices well-conditioned. (default: 0.1)
        eps (float): Added to the diagonal to guarantee positive definiteness
            for windows with (near-)zero signal. (default: 1e-6)
        center (bool): Whether to subtract each window's mean before computing
            the covariance. Unnecessary for z-normalized input. (default: False)
    """

    window_length: int = 64
    hop_length: int = 16
    shrinkage: float = 0.1
    eps: float = 1e-6
    center: bool = False

    def __post_init__(self) -> None:
        assert 0.0 <= self.shrinkage <= 1.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        x = x.unfold(-1, self.window_length, self.hop_length)  # (..., C, T', win)
        if self.center:
            x = x - x.mean(dim=-1, keepdim=True)
        x = x.movedim(-2, 0)  # (T', ..., C, win)
        cov = x @ x.transpose(-1, -2) / self.window_length  # (T', ..., C, C)

        C = cov.shape[-1]
        eye = torch.eye(C, dtype=cov.dtype, device=cov.device)
        trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        cov = (1 - self.shrinkage) * cov + self.shrinkage * trace / C * eye
        return cov + self.eps * eye


def log_cholesky_mean(
    lower_sum: torch.Tensor, log_diag_sum: torch.Tensor, num: torch.Tensor | int
) -> torch.Tensor:
    """Log-Cholesky Fréchet mean (Lin, 2019, Corollary 12) of SPD matrices
    from the sums of the strictly lower triangular parts ``lower_sum``
    (..., C, C) and of the log diagonals ``log_diag_sum`` (..., C) of their
    Cholesky factors over ``num`` matrices:

        L_bar = lower_sum / num + diag(exp(log_diag_sum / num)),
        M = L_bar L_bar^T.
    """
    L_bar = lower_sum / num + torch.diag_embed((log_diag_sum / num).exp())
    return L_bar @ L_bar.transpose(-1, -2)


def sym_inv_sqrt(M: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Symmetric inverse square root ``M^{-1/2}`` of SPD matrices of shape
    (..., C, C), with eigenvalues floored at ``eps``."""
    eigvals, eigvecs = torch.linalg.eigh(M)
    inv_sqrt_eigvals = eigvals.clamp(min=eps).rsqrt()
    return eigvecs @ torch.diag_embed(inv_sqrt_eigvals) @ eigvecs.transpose(-1, -2)


@lru_cache(maxsize=None)
def _load_session_means(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


@dataclass
class SessionRecentering:
    """Recenters covariance matrices of shape (T, bands, C, C) at their
    session's Log-Cholesky Fréchet mean M (Lin, 2019) via the congruence
    ``M^{-1/2} P M^{-1/2}``, with a separate M per band.

    ``means_path`` is the file written by ``scripts/compute_frechet_means.py``.
    The session is not known when the transform is configured, so
    ``WindowedEMGDataset`` binds it via ``for_session()`` (see ``bind_session``).

    Not causal: M is computed from the whole session. This is intended for
    training (and for validation during training, to select checkpoints);
    use ``CausalSessionRecentering`` at test time, which needs no means file.

    Args:
        means_path (str): Path to the per-session Fréchet mean file.
        session (str): Session name. Set by ``for_session()``. (default: None)
    """

    means_path: str
    session: str | None = None

    def for_session(self, session: str) -> "SessionRecentering":
        return dataclasses.replace(self, session=session)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self.session is not None, "Session not bound; see for_session()"
        means = _load_session_means(str(Path(self.means_path).expanduser()))
        assert self.session in means["inv_sqrt"], f"No Fréchet mean for {self.session}"
        inv_sqrt = means["inv_sqrt"][self.session].to(tensor.dtype)  # (bands, C, C)
        x = inv_sqrt @ tensor @ inv_sqrt
        return (x + x.transpose(-1, -2)) / 2  # Remove float32 round-off asymmetry


@dataclass
class CausalSessionRecentering:
    """Causal counterpart of ``SessionRecentering``: recenters each covariance
    frame at the running Log-Cholesky Fréchet mean (Lin, 2019) of all frames
    up to and including itself, ``M_t^{-1/2} P_t M_t^{-1/2}``, with

        L_bar_t = mean_{i<=t} floor(L_i) + exp(mean_{i<=t} log D(L_i)),
        M_t = L_bar_t L_bar_t^T,

    i.e., exactly the estimator of ``scripts/compute_frechet_means.py``
    restricted to the past. Needs no precomputed means file. The input must
    be of shape (T, bands, C, C).

    Args:
        warmup (int): Frames at the start use the mean of the first ``warmup``
            frames (a lookahead of ``warmup`` frames only there); 0 is strictly
            causal, in which case the first frame maps to the identity.
            (default: 0)
        eps (float): Eigenvalue floor for the inverse square root. (default: 1e-8)
    """

    warmup: int = 0
    eps: float = 1e-8

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.to(torch.float64)
        T = x.shape[0]
        L = torch.linalg.cholesky(x)  # (T, bands, C, C)
        diag = L.diagonal(dim1=-2, dim2=-1)  # (T, bands, C)
        lower = L - torch.diag_embed(diag)
        num = torch.arange(1, T + 1, dtype=torch.float64).view(T, 1, 1, 1)
        lower_sum = lower.cumsum(0)
        log_diag_sum = diag.log().cumsum(0)
        w = min(self.warmup, T)
        if w > 1:
            lower_sum[:w] = lower_sum[w - 1]
            log_diag_sum[:w] = log_diag_sum[w - 1]
            num[:w] = w
        M = log_cholesky_mean(lower_sum, log_diag_sum, num)
        inv_sqrt = sym_inv_sqrt(M, eps=self.eps)
        y = inv_sqrt @ x @ inv_sqrt
        y = (y + y.transpose(-1, -2)) / 2
        return y.to(tensor.dtype)


@dataclass
class SPDAugment:
    """SpecAugment counterpart for recentered covariance matrices of shape
    (T, bands, C, C). Masked entries are set to the identity, i.e., the
    session mean after ``SessionRecentering``, so the matrices stay SPD:

    - Time masks replace consecutive frames with the identity.
    - Electrode masks replace the rows and columns of a contiguous (circular)
      block of electrodes with those of the identity, akin to frequency masks
      and to those electrodes losing contact.

    Masks are drawn independently per band.

    Args:
        n_time_masks (int): Maximum number of time masks per band, uniformly
            sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask, uniformly
            sampled from 0. (default: 0)
        n_electrode_masks (int): Maximum number of electrode masks per band,
            uniformly sampled from 0. (default: 0)
        electrode_mask_param (int): Maximum number of electrodes per mask,
            uniformly sampled from 0. (default: 0)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    n_electrode_masks: int = 0
    electrode_mask_param: int = 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        T, bands, C, _ = tensor.shape
        x = tensor.clone()
        eye = torch.eye(C, dtype=x.dtype, device=x.device)

        for band in range(bands):
            for _ in range(np.random.randint(self.n_time_masks + 1)):
                length = np.random.randint(self.time_mask_param + 1)
                if length == 0 or length > T:
                    continue
                start = np.random.randint(T - length + 1)
                x[start : start + length, band] = eye

            for _ in range(np.random.randint(self.n_electrode_masks + 1)):
                width = np.random.randint(self.electrode_mask_param + 1)
                if width == 0:
                    continue
                idx = (np.random.randint(C) + np.arange(width)) % C
                x[:, band, idx, :] = eye[idx, :]
                x[:, band, :, idx] = eye[:, idx]

        return x


def bind_session(transform: Transform[Any, Any], session: str) -> Transform[Any, Any]:
    """Returns ``transform`` with any session-specific transforms (those
    defining ``for_session()``, possibly nested in ``Compose``/``ForEach``)
    bound to ``session``. Other transforms are shared as is."""
    if isinstance(transform, Compose):
        return Compose([bind_session(t, session) for t in transform.transforms])
    if isinstance(transform, ForEach):
        return dataclasses.replace(
            transform, transform=bind_session(transform.transform, session)
        )
    if callable(getattr(transform, "for_session", None)):
        return transform.for_session(session)
    return transform
