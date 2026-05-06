# Data Format

MINTCDR reads three files from `data.root`.

## `interactions.csv`

```text
domain,user_id,item_id,label,timestamp,split
source,u1,s12,1,100,train
target,u1,t03,1,101,test
```

Columns:

- `domain`: `source` or `target`
- `user_id`: raw user id
- `item_id`: raw item id inside that domain
- `label`: implicit feedback label
- `timestamp`: used for sequence order
- `split`: `train`, `valid`, or `test`

Target interactions are split by user as `8:1:1`. Source interactions use `8:2`.

## `user_semantics.json`

```json
{"u1": [0.01, 0.02, 0.03]}
```

## `item_semantics.json`

```json
{
  "source:s12": [0.01, 0.02, 0.03],
  "target:t03": [0.04, 0.05, 0.06]
}
```

All semantic vectors should have the same dimension.
