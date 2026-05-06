# MINTCDR

Code for MINTCDR, a cross-domain recommendation model with LLM-based semantic features and source-domain utility calibration.

The current version keeps the training pipeline, evaluation code, preprocessing scripts, and two simple MF baselines. Semantic vectors are loaded from cached files, so experiments do not call an external LLM during training.

## Files

```text
src/mintcdr/      model, data loader, trainer, metrics
configs/         model and baseline configs
scripts/         preprocessing, training, evaluation
prompts/         prompt templates used before semantic-vector caching
docs/            data and experiment notes
```

## Install

```bash
pip install -e .
```

## Run a small check

```bash
python scripts/generate_sample_data.py --output data/sample
python scripts/train.py --config configs/sample.yaml
python scripts/evaluate.py --config configs/sample.yaml --checkpoint runs/sample/best.pt
```

## Baselines

```bash
python scripts/train.py --config configs/baseline_tgt.yaml
python scripts/train.py --config configs/baseline_cdr_mf.yaml
```

## Data

Expected files:

- `interactions.csv`
- `user_semantics.json`
- `item_semantics.json`

See [docs/data_format.md](docs/data_format.md) for columns and JSON keys. The Amazon splits used in the paper follow the protocol in [docs/experiment_protocol.md](docs/experiment_protocol.md).
