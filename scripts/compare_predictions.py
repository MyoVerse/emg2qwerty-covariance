"""
Script to compare two models character by character on the test sessions of
all users, from the predictions dumped by `scripts/dump_predictions.py`
(``<dir>/user*.json``, one directory per model).

Each prediction is aligned to its target with Levenshtein edit operations (as
in `emg2qwerty.metrics.CharacterErrorRates`), and every edit is attributed to
a target character: substitutions and deletions to the affected character,
insertions to the character they precede. Per-character error counts thus sum
to the edit distance, and their mean is the CER.

Reports, pooled over all users (A = --model-a, B = --model-b):

- CER of each model and the absolute/relative CER reduction of A over B.
- Character-level Wilcoxon signed-rank test on the per-character error
  counts of B vs A. Since the paired differences are almost all 0 or ±1, this
  is close to a sign (McNemar) test, and it treats characters as independent,
  which errors are not (they come in bursts), so its p-value is optimistic.
- Utterance-level Wilcoxon signed-rank test on the per-utterance CERs, a paired
  permutation test (sign-flips of the per-utterance error difference) and a
  bootstrap CI of the CER reduction, resampling utterances within each user.
  Utterances (prompts) are the target text split at the enter (⏎) keystroke
  that ends each prompt. These account for errors being correlated within an
  utterance.

Tests are one-sided (A has lower CER than B) by default, see --alternative.
"""

import json
from pathlib import Path

import click
import Levenshtein
import numpy as np
from scipy import stats


ENTER = "⏎"  # ⏎


def char_errors(prediction: str, target: str) -> np.ndarray:
    """Number of edits attributed to each target character."""
    errors = np.zeros(len(target), dtype=np.int64)
    for op, _, dpos in Levenshtein.editops(prediction, target):
        # "delete" removes an extra predicted character before target[dpos]
        errors[min(dpos, len(target) - 1)] += 1
    return errors


def sentence_ids(target: str) -> np.ndarray:
    """Sentence index of each target character (the ⏎ ends its sentence)."""
    is_enter = np.array([c == ENTER for c in target])
    return np.concatenate([[0], np.cumsum(is_enter)[:-1]])


def load(predictions_dir: str) -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text())
        for path in sorted(Path(predictions_dir).glob("user*.json"))
    }


@click.command()
@click.option("--model-a", type=str, required=True, help="Predictions dir of model A")
@click.option("--model-b", type=str, required=True, help="Predictions dir of model B")
@click.option("--num-resamples", type=int, default=100_000)
@click.option("--seed", type=int, default=0)
@click.option(
    "--alternative",
    type=click.Choice(["greater", "two-sided"]),
    default="greater",
    help="greater: one-sided, A has lower CER than B",
)
def main(
    model_a: str, model_b: str, num_resamples: int, seed: int, alternative: str
):
    preds_a, preds_b = load(model_a), load(model_b)
    users = sorted(preds_a.keys() & preds_b.keys())
    assert len(users) > 0, "No users with predictions from both models"
    if missing := sorted(preds_a.keys() ^ preds_b.keys()):
        print(f"Skipping users without predictions from both models: {missing}")

    # Per-character errors of both models, and per-sentence sums
    chars_a, chars_b = [], []  # per-character error counts
    sent_a, sent_b, sent_len, sent_user = [], [], [], []  # per-sentence
    print(f"\n{'user':<8}{'chars':>8}{'sents':>7}{'CER A':>8}{'CER B':>8}")
    for u, user in enumerate(users):
        sessions_b = {s["session"]: s for s in preds_b[user]["sessions"]}
        user_a, user_b, user_len, user_sents = 0, 0, 0, 0
        for session_a in preds_a[user]["sessions"]:
            session_b = sessions_b[session_a["session"]]
            target = session_a["target"]
            assert target == session_b["target"], session_a["session"]

            err_a = char_errors(session_a["prediction"], target)
            err_b = char_errors(session_b["prediction"], target)
            chars_a.append(err_a)
            chars_b.append(err_b)

            ids = sentence_ids(target)
            sent_a.append(np.bincount(ids, weights=err_a))
            sent_b.append(np.bincount(ids, weights=err_b))
            sent_len.append(np.bincount(ids))
            sent_user.append(np.full(ids[-1] + 1, u))

            user_a += err_a.sum()
            user_b += err_b.sum()
            user_len += len(target)
            user_sents += ids[-1] + 1
        print(
            f"{user:<8}{user_len:>8}{user_sents:>7}"
            f"{100 * user_a / user_len:>8.2f}{100 * user_b / user_len:>8.2f}"
        )

    chars_a, chars_b = np.concatenate(chars_a), np.concatenate(chars_b)
    sent_a, sent_b = np.concatenate(sent_a), np.concatenate(sent_b)
    sent_len, sent_user = np.concatenate(sent_len), np.concatenate(sent_user)
    num_chars = len(chars_a)

    cer_a = 100 * chars_a.sum() / num_chars
    cer_b = 100 * chars_b.sum() / num_chars
    print(f"\nPooled over {len(users)} users, {num_chars} target characters:")
    print(f"  CER A = {cer_a:.2f}, CER B = {cer_b:.2f}")
    print(
        f"  A reduces CER by {cer_b - cer_a:.2f} points "
        f"({100 * (cer_b - cer_a) / cer_b:.1f}% relative)"
    )

    print(f"\nTests ({alternative}):")

    # Character-level Wilcoxon signed-rank test on per-character error counts
    num_better = int((chars_b > chars_a).sum())
    num_worse = int((chars_b < chars_a).sum())
    p_char = stats.wilcoxon(chars_b, chars_a, alternative=alternative).pvalue
    print(
        f"  Character-level Wilcoxon: fewer errors under A on {num_better} chars, "
        f"more on {num_worse}, p = {p_char:.3g} (assumes independent chars)"
    )

    # Utterance-level Wilcoxon signed-rank test on per-sentence CERs
    p_sent = stats.wilcoxon(
        sent_b / sent_len, sent_a / sent_len, alternative=alternative
    ).pvalue
    print(f"  Utterance-level Wilcoxon ({len(sent_len)} utterances): p = {p_sent:.3g}")

    # Utterance-level paired permutation test: under the null, the two models'
    # errors on each sentence are exchangeable, i.e., their difference flips sign.
    rng = np.random.default_rng(seed)
    diff = sent_b - sent_a
    observed = diff.sum()
    num_extreme = 0
    for start in range(0, num_resamples, 10_000):
        n = min(10_000, num_resamples - start)
        signs = rng.choice([-1, 1], size=(n, len(diff)))
        permuted = signs @ diff
        if alternative == "two-sided":
            num_extreme += (np.abs(permuted) >= abs(observed)).sum()
        else:
            num_extreme += (permuted >= observed).sum()
    p_perm = (num_extreme + 1) / (num_resamples + 1)
    print(f"  Utterance-level paired permutation: p = {p_perm:.3g}")

    # Bootstrap CI of the CER reduction, resampling sentences within each user
    boot_a = np.zeros(num_resamples // 10)
    boot_b = np.zeros(num_resamples // 10)
    boot_len = np.zeros(num_resamples // 10)
    for u in range(len(users)):
        idx = np.flatnonzero(sent_user == u)
        resampled = idx[rng.integers(0, len(idx), size=(len(boot_a), len(idx)))]
        boot_a += sent_a[resampled].sum(1)
        boot_b += sent_b[resampled].sum(1)
        boot_len += sent_len[resampled].sum(1)
    abs_reduction = 100 * (boot_b - boot_a) / boot_len
    rel_reduction = 100 * (boot_b - boot_a) / boot_b
    print(
        f"\nBootstrap 95% CI ({len(boot_a)} resamples, utterances within users): "
        f"absolute reduction [{np.percentile(abs_reduction, 2.5):.2f}, "
        f"{np.percentile(abs_reduction, 97.5):.2f}] points, relative "
        f"[{np.percentile(rel_reduction, 2.5):.1f}, "
        f"{np.percentile(rel_reduction, 97.5):.1f}]%"
    )


if __name__ == "__main__":
    main()
