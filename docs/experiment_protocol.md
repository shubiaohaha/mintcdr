# Experiment Protocol

Tasks used in the Amazon experiments:

- Clothing -> Sports
- Sports -> Clothing
- Clothing -> Video
- Video -> Clothing

Target-domain metrics:

- `Recall@10`
- `Recall@20`
- `NDCG@10`
- `NDCG@20`

Default settings:

| item | value |
| --- | --- |
| embedding dim | `64` |
| optimizer | Adam |
| learning rate | `0.001` |
| source loss weight | `0.2` |
| contrastive weight | `1e-4` |
| weight decay | `1e-6` |
| temperature | `0.2` |
| DDAN weight | `0.5` |
| alpha temperature | `1.0` |

The sample config uses a smaller batch size for quick checks. For the full Amazon runs I use `batch_size=2048`.
