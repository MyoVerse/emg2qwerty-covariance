"""
Script to tabulate the test CER of the prediction dumps written by
`scripts/dump_predictions.py` into ``results/test_cer.md``, alongside the
upstream log-spectrogram baselines of the emg2qwerty paper
(`scripts/experimental_results.py`).

Dumps are expected at ``<predictions_dir>/<model>_<decoder>/user*.json``,
with ``<model>`` one of ``generic`` (zero-shot), ``randominit`` or
``finetuned``, and ``<decoder>`` one of ``greedy`` or ``beam``.

With ``--split val``, the ``val/CER`` of ``python -m emg2qwerty.train
train=False +metrics_path=...`` (windowed, non-causal validation, the metric
that selects checkpoints, as upstream) is tabulated from the metrics files at
``<metrics_dir>/<model>_<decoder>/user*.json`` into ``results/val_cer.md``,
alongside the upstream Val CER of Table 2. Decoders without any file are left
out.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import click
import numpy as np

from experimental_results import EXPERIMENTAL_RESULTS


USERS = [f"user{i}" for i in range(8)]
MODELS = {
    "generic": ("Generic, zero-shot", "Generic (no personalization)"),
    "randominit": ("Personalized, random init", "Personalized (random-init)"),
    "finetuned": ("Personalized, fine-tuned from generic", "Personalized (finetuned)"),
}
DECODERS = {"greedy": "No LM", "beam": "LM"}


def baseline(benchmark: str, lm: str, metric: str = "Test CER") -> list[float]:
    for result in EXPERIMENTAL_RESULTS:
        if (result["Model benchmark"], result["LM"], result["Metric"]) == (
            benchmark,
            lm,
            metric,
        ):
            return result["CER"]
    raise KeyError((benchmark, lm, metric))


def load_cer(results_dir: Path, split: str = "test") -> list[float | None]:
    """Test CER of `scripts/dump_predictions.py` dumps, or val/CER of the
    metrics files of `emg2qwerty.train` (see ``--split``)."""
    cers = []
    for user in USERS:
        path = results_dir.joinpath(f"{user}.json")
        if not path.exists():
            cers.append(None)
        elif split == "test":
            cers.append(json.loads(path.read_text())["test_metrics"]["CER"])
        else:
            cers.append(json.loads(path.read_text())["val_metrics"][0]["val/CER"])
    return cers


def row(label: str, cers: Sequence[float | None]) -> str:
    done = [c for c in cers if c is not None]
    cells = ["—" if c is None else f"{c:.2f}" for c in cers]
    mean = f"{np.mean(done):.2f}" if len(done) == len(cers) else "—"
    return f"| {label} | " + " | ".join(cells) + f" | {mean} |"


@click.command()
@click.option(
    "--predictions-dir",
    type=str,
    default=Path(__file__).parents[1].joinpath("results", "predictions"),
    help="Directory containing <model>_<decoder>/user*.json prediction dumps",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Markdown file to write (default: results/<split>_cer.md)",
)
@click.option(
    "--metrics-dir",
    type=str,
    default=Path(__file__).parents[1].joinpath("results", "metrics"),
    help="Directory containing <model>_<decoder>/user*.json metrics files of "
    "emg2qwerty.train (for --split val)",
)
@click.option(
    "--split",
    type=click.Choice(["test", "val"]),
    default="test",
    help="Tabulate the test CER of the prediction dumps, or the val/CER of the "
    "metrics files",
)
def main(predictions_dir: str, output: str | None, metrics_dir: str, split: str):
    if output is None:
        output = Path(__file__).parents[1].joinpath("results", f"{split}_cer.md")
    if split == "test":
        results_dir = Path(predictions_dir)
        metric = "Test CER"
        lines = [
            "# Test CER (%)",
            "",
            "Covariance models are evaluated with strictly causal test-time "
            "normalization (config/transforms/covariance.yaml); the spectrogram "
            "baselines are the upstream emg2qwerty results (Table 2 of the paper).",
            "",
        ]
    else:
        results_dir = Path(metrics_dir)
        metric = "Val CER"
        lines = [
            "# Validation CER (%)",
            "",
            "`val/CER` of `python -m emg2qwerty.train train=False` on the released "
            "checkpoints: the validation sessions in 4 s windows under the "
            "non-causal val transforms of config/transforms/covariance.yaml "
            "(per-window z-normalization, recentering at the precomputed session "
            "Fréchet mean of models/frechet_means.pt), i.e., the metric that "
            "selected the checkpoints, as upstream. The spectrogram baselines "
            "are the upstream Val CER (Table 2 of the paper). Test sessions "
            "are evaluated causally instead, see results/test_cer.md.",
            "",
        ]
    for decoder, lm in DECODERS.items():
        dirs = [results_dir.joinpath(f"{model}_{decoder}") for model in MODELS]
        if split == "val" and not any(d.exists() for d in dirs):
            continue
        lines += [
            f"## {lm} ({decoder})",
            "",
            "| | " + " | ".join(USERS) + " | mean |",
            "|---" * (len(USERS) + 2) + "|",
        ]
        for model, (label, benchmark) in MODELS.items():
            cers = load_cer(results_dir.joinpath(f"{model}_{decoder}"), split)
            lines.append(row(f"{label}, covariance", cers))
            lines.append(
                row(f"{label}, spectrogram (upstream)", baseline(benchmark, lm, metric))
            )
        lines.append("")

    Path(output).write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
