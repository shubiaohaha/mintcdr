from __future__ import annotations

import argparse

from mintcdr.config import load_config
from mintcdr.baseline_trainer import BaselineTrainer
from mintcdr.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    trainer = BaselineTrainer(cfg) if "baseline" in cfg else Trainer(cfg)
    if hasattr(trainer, "load"):
        trainer.load(args.checkpoint)
    else:
        import torch

        checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model"])
    print(trainer.evaluate(args.split))


if __name__ == "__main__":
    main()

