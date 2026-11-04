"""
Script to compute, for every train/val session of the given user configs, the
Log-Cholesky Fréchet mean M of the session's sliding-window EMG covariance
matrices (separately for the left and right bands), as per

    Lin, Z. Riemannian geometry of symmetric positive definite matrices via
    Cholesky decomposition. SIAM J. Matrix Anal. Appl. 40(4), 2019.

The mean has a closed form (Corollary 12): with L_i the Cholesky factors,

    L_bar = mean_i floor(L_i) + exp(mean_i log D(L_i)),    M = L_bar L_bar^T

so it is accumulated in a single streaming pass over the data. The output
file stores M and M^{-1/2} per session, the latter consumed by
`emg2qwerty.transforms.SessionRecentering` to recenter the training and
validation windows (config/transforms/covariance.yaml). Test sessions are
recentered at a causal running mean (`CausalSessionRecentering`) and need
no precomputed mean.

The covariances are computed exactly as the model sees them during training:
each dataset window (``--window-length`` plus ``--padding``, as fed by
`WindowedEMGDataset` without jitter) is z-normalized on its own
(`emg2qwerty.transforms.ZNormalizeTime`) before its covariances are taken.
The covariance options must match the transform config.

Example (all sessions needed to train the generic and the 8 personalized
models):

    python scripts/compute_frechet_means.py \\
        --user-config config/user/generic.yaml \\
        --user-config "config/user/user[0-7].yaml"
"""

import glob
import logging
from multiprocessing import Pool
from pathlib import Path

import click
import torch
import yaml

from emg2qwerty.data import EMGSessionData
from emg2qwerty.transforms import (
    log_cholesky_mean,
    SlidingCovariance,
    sym_inv_sqrt,
    ZNormalizeTime,
)


log = logging.getLogger(__name__)

NUM_BANDS = 2
NUM_CHANNELS = 16


def window_segments(
    session_length: int, window_length: int, padding: tuple[int, int]
) -> list[tuple[int, int]]:
    """(start, end) sample ranges that `WindowedEMGDataset` (without jitter)
    feeds as individual items, i.e., the ranges ``ZNormalizeTime`` sees."""
    left, right = padding
    num_windows = max(session_length - window_length, 0) // window_length + 1
    return [
        (max(i * window_length - left, 0), i * window_length + window_length + right)
        for i in range(num_windows)
    ]


def log_cholesky_sums(
    emg: torch.Tensor, covariance: SlidingCovariance, chunk_frames: int = 16384
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Returns the sums over all covariance windows of ``emg`` (T, bands, C)
    of the strictly lower triangular parts (bands, C, C) and of the log
    diagonals (bands, C) of the Cholesky factors, along with the number of
    windows."""
    win, hop = covariance.window_length, covariance.hop_length
    num_frames = (len(emg) - win) // hop + 1

    lower_sum = torch.zeros(NUM_BANDS, NUM_CHANNELS, NUM_CHANNELS, dtype=torch.float64)
    log_diag_sum = torch.zeros(NUM_BANDS, NUM_CHANNELS, dtype=torch.float64)
    for start in range(0, num_frames, chunk_frames):
        n = min(chunk_frames, num_frames - start)
        chunk = emg[start * hop : (start + n - 1) * hop + win]
        L = torch.linalg.cholesky(covariance(chunk))  # (n, bands, C, C)
        lower_sum += L.tril(diagonal=-1).sum(0)
        log_diag_sum += L.diagonal(dim1=-2, dim2=-1).log().sum(0)

    return lower_sum, log_diag_sum, num_frames


def session_log_cholesky_sums(
    hdf5_path: Path,
    covariance: SlidingCovariance,
    window_length: int,
    padding: tuple[int, int],
) -> tuple[str, tuple[torch.Tensor, torch.Tensor, int]]:
    """Accumulates `log_cholesky_sums()` over the dataset windows of a session,
    each z-normalized on its own. Frames in overlapping (padded) windows are
    counted once per window, as the model sees them."""
    with EMGSessionData(hdf5_path) as session:
        timeseries = session[:]
    emg = torch.stack(
        [torch.as_tensor(timeseries[f]) for f in ("emg_left", "emg_right")], dim=1
    ).double()  # (T, bands, C)

    znorm = ZNormalizeTime()
    lower_sum = torch.zeros(NUM_BANDS, NUM_CHANNELS, NUM_CHANNELS, dtype=torch.float64)
    log_diag_sum = torch.zeros(NUM_BANDS, NUM_CHANNELS, dtype=torch.float64)
    num_frames = 0
    for start, end in window_segments(len(emg), window_length, padding):
        lower, log_diag, n = log_cholesky_sums(znorm(emg[start:end]), covariance)
        lower_sum += lower
        log_diag_sum += log_diag
        num_frames += n
    return hdf5_path.stem, (lower_sum, log_diag_sum, num_frames)


def _worker(args):
    return session_log_cholesky_sums(*args)


@click.command()
@click.option(
    "--dataset-root",
    type=str,
    default=Path(__file__).parents[1].joinpath("data"),
    help="Dataset root directory",
)
@click.option(
    "--output",
    type=str,
    default=Path(__file__).parents[1].joinpath("models", "frechet_means.pt"),
    help="Output file with per-session Fréchet means",
)
@click.option(
    "--user-config",
    type=str,
    multiple=True,
    required=True,
    help="User config yaml(s) whose sessions to process (repeatable, globs allowed)",
)
@click.option(
    "--splits",
    type=str,
    multiple=True,
    default=("train", "val"),
    help="Splits of the user configs to process (repeatable)",
)
@click.option("--window-length", type=int, default=8000, help="Dataset window length")
@click.option("--padding", type=(int, int), default=(1800, 200), help="Window padding")
@click.option("--cov-window-length", type=int, default=64)
@click.option("--cov-hop-length", type=int, default=16)
@click.option("--shrinkage", type=float, default=0.1)
@click.option("--center/--no-center", default=False)
@click.option("--num-workers", type=int, default=8)
def main(
    dataset_root: str,
    output: str,
    user_config: tuple[str, ...],
    splits: tuple[str, ...],
    window_length: int,
    padding: tuple[int, int],
    cov_window_length: int,
    cov_hop_length: int,
    shrinkage: float,
    center: bool,
    num_workers: int,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    sessions: set[str] = set()
    for pattern in user_config:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                dataset = yaml.safe_load(f)["dataset"]
            for split in splits:
                sessions.update(s["session"] for s in dataset[split])

    covariance = SlidingCovariance(
        window_length=cov_window_length,
        hop_length=cov_hop_length,
        shrinkage=shrinkage,
        center=center,
    )
    jobs = [
        (
            Path(dataset_root).joinpath(f"{session}.hdf5"),
            covariance,
            window_length,
            padding,
        )
        for session in sorted(sessions)
    ]
    log.info(f"Computing Fréchet means over {len(jobs)} sessions")

    means, inv_sqrts, counts = {}, {}, {}
    # One intra-op thread per worker: the work is many small batched ops, and
    # num_workers processes each spawning a thread per core oversubscribes the CPU.
    with Pool(num_workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for i, (session, (lower_sum, log_diag_sum, n)) in enumerate(
            pool.imap_unordered(_worker, jobs)
        ):
            M = log_cholesky_mean(lower_sum, log_diag_sum, n)  # (bands, C, C)
            means[session] = M.float()
            inv_sqrts[session] = sym_inv_sqrt(M).float()
            counts[session] = n
            log.info(f"[{i + 1}/{len(jobs)}] {session}: {n} windows")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": means,
            "inv_sqrt": inv_sqrts,
            "num_windows": counts,
            "covariance": {
                "window_length": cov_window_length,
                "hop_length": cov_hop_length,
                "shrinkage": shrinkage,
                "center": center,
            },
            "znorm": {"window_length": window_length, "padding": padding},
        },
        output,
    )
    log.info(f"Saved Fréchet means for {len(means)} sessions to {output}")


if __name__ == "__main__":
    main()
