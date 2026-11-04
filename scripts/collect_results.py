"""
Script to tabulate the test CER of the prediction dumps written by
`scripts/dump_predictions.py` into ``results/test_cer.md``, alongside the
upstream log-spectrogram baselines of the emg2qwerty paper
(`scripts/experimental_results.py`).

Dumps are expected at ``<predictions_dir>/<model>_<decoder>/user*.json``,
with ``<model>`` one of ``generic`` (zero-shot), ``randominit`` or
``finetuned``, and ``<decoder>`` one of ``greedy`` or ``beam``.
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


def baseline(benchmark: str, lm: str) -> list[float]:
    for result in EXPERIMENTAL_RESULTS:
        if (result["Model benchmark"], result["LM"], result["Metric"]) == (
            benchmark,
            lm,
            "Test CER",
        ):
            return result["CER"]
    raise KeyError((benchmark, lm))


def load_cer(predictions_dir: Path) -> list[float | None]:
    cers = []
    for user in USERS:
        path = predictions_dir.joinpath(f"{user}.json")
        if path.exists():
            cers.append(json.loads(path.read_text())["test_metrics"]["CER"])
        else:
            cers.append(None)
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
    default=Path(__file__).parents[1].joinpath("results", "test_cer.md"),
    help="Markdown file to write",
)
def main(predictions_dir: str, output: str):
    lines = [
        "# Test CER (%)",
        "",
        "Covariance models are evaluated with strictly causal test-time "
        "normalization (config/transforms/covariance.yaml); the spectrogram "
        "baselines are the upstream emg2qwerty results (Table 2 of the paper).",
        "",
    ]
    for decoder, lm in DECODERS.items():
        lines += [
            f"## {lm} ({decoder})",
            "",
            "| | " + " | ".join(USERS) + " | mean |",
            "|---" * (len(USERS) + 2) + "|",
        ]
        for model, (label, benchmark) in MODELS.items():
            cers = load_cer(Path(predictions_dir).joinpath(f"{model}_{decoder}"))
            lines.append(row(f"{label}, covariance", cers))
            lines.append(row(f"{label}, spectrogram (upstream)", baseline(benchmark, lm)))
        lines.append("")

    Path(output).write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
