"""
Script to dump the decoded test-set predictions of a trained model, one entry
per test session, so that models can be compared character by character (see
`scripts/compare_predictions.py`). Takes the same Hydra config and overrides
as `emg2qwerty.train` (``checkpoint`` is required), plus the output file:

    python scripts/dump_predictions.py user=user0 \
        checkpoint=models/personalized-finetuned/user0.ckpt \
        +predictions_path=results/predictions/finetuned_greedy/user0.json

The test CER of the dumped predictions is printed and matches ``test/CER`` of
`emg2qwerty.train` for the same checkpoint and decoder.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import get_class, instantiate, to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.transforms import Transform


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    assert config.checkpoint is not None, "checkpoint=<path> is required"
    predictions_path = Path(to_absolute_path(config.predictions_path))

    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    checkpoint = to_absolute_path(config.checkpoint)
    log.info(f"Loading module from checkpoint {checkpoint}")
    module = get_class(config.module._target_).load_from_checkpoint(
        checkpoint,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device).eval()

    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        # `setup()` builds all splits (cheaply, without loading data), so the
        # train/val sessions are passed too even though only test is used
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )
    datamodule.setup("test")

    # Test sessions are fed whole with batch size 1 and no shuffling, so
    # batches line up with the per-session datasets.
    sessions = []
    metrics = CharacterErrorRates()
    with torch.no_grad():
        for dataset, batch in zip(
            datamodule.test_dataset.datasets, datamodule.test_dataloader()
        ):
            inputs = batch["inputs"].to(device)
            emissions = module(inputs)

            # Same as `TDSConvCTCModule._step()`
            T_diff = inputs.shape[0] - emissions.shape[0]
            emission_lengths = batch["input_lengths"] - T_diff
            prediction = module.decoder.decode_batch(
                emissions=emissions.cpu().numpy(),
                emission_lengths=emission_lengths.numpy(),
            )[0]
            target = LabelData.from_labels(
                batch["targets"][: batch["target_lengths"][0], 0].numpy()
            )

            session_metrics = CharacterErrorRates()
            session_metrics.update(prediction=prediction, target=target)
            metrics.update(prediction=prediction, target=target)
            log.info(
                f"{dataset.hdf5_path.stem}: "
                f"CER {session_metrics.compute()['CER']:.2f} ({len(target)} chars)"
            )
            sessions.append(
                {
                    "session": dataset.hdf5_path.stem,
                    "prediction": prediction.text,
                    "target": target.text,
                }
            )

    log.info(f"{config.user} test metrics: {metrics.compute()}")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w") as f:
        json.dump(
            {
                "user": config.user,
                "checkpoint": checkpoint,
                "decoder": OmegaConf.to_container(config.decoder, resolve=True),
                "test_metrics": metrics.compute(),
                "sessions": sessions,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    log.info(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()
