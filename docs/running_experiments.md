# Running Experiments

## Main Model

```bash
python scripts/train.py --config configs/sample.yaml
```

## Baselines

```bash
python scripts/train.py --config configs/baseline_tgt.yaml
python scripts/train.py --config configs/baseline_cdr_mf.yaml
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/sample.yaml --checkpoint runs/sample/best.pt
```
