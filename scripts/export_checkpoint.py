"""
Script to export a training checkpoint for distribution: strips the
optimizer, scheduler and loop states (two thirds of the file), keeping only
the weights and the hyper-parameters needed by
`TDSConvCTCModule.load_from_checkpoint()`. The exported checkpoint is reloaded
and checked against the original before the script returns.

    python scripts/export_checkpoint.py \\
        logs/<run>/checkpoints/epoch=142-step=128128.ckpt models/generic.ckpt
"""

import click
import torch

from emg2qwerty.lightning import TDSConvCTCModule


KEEP = ["pytorch-lightning_version", "epoch", "global_step", "hyper_parameters"]


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False))
@click.argument("target", type=click.Path(dir_okay=False))
def main(source: str, target: str):
    checkpoint = torch.load(source, map_location="cpu")
    exported = {key: checkpoint[key] for key in KEEP if key in checkpoint}
    exported["state_dict"] = checkpoint["state_dict"]
    torch.save(exported, target)

    module = TDSConvCTCModule.load_from_checkpoint(target, map_location="cpu")
    for name, param in module.state_dict().items():
        assert torch.equal(param, checkpoint["state_dict"][name]), name
    print(f"Exported {source} -> {target} ({len(module.state_dict())} tensors)")


if __name__ == "__main__":
    main()
