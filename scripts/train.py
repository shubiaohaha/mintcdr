from __future__ import annotations

import argparse

from mintcdr.config import load_config
from mintcdr.baseline_trainer import BaselineTrainer
from mintcdr.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    trainer = BaselineTrainer(cfg) if "baseline" in cfg else Trainer(cfg)
    best = trainer.fit()
    test = trainer.evaluate("test")
    print({"best_valid": best, "test": test})


if __name__ == "__main__":
    main()

