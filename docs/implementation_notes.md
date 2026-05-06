# Implementation Notes

The code follows the MINTCDR pipeline used in the method chapter.

## Code Map

| component | location |
| --- | --- |
| user sequence encoder | `SequenceEncoder` |
| dual ID branches | `source_user_id`, `target_user_id`, `source_item_id`, `target_item_id` |
| DDAN losses | `MINTCDR.ddan_losses` |
| semantic-collaborative alignment | `MINTCDR.cross_modal_losses` |
| DICN | `DomainInfluenceCalibrationNetwork` |
| source utility labels | `Trainer.estimate_source_utilities` |
| utility distillation | `UtilityDistiller` |

## Notes

`z_u` and `z_i` are read as cached vectors. I generate them before training and store them in JSON files, so the recommender stage does not repeatedly call the text encoder.

The source-sample utility estimator is written as a separate trainer method. For large runs it uses a batched approximation; if an exact leave-one-out variant is needed, this is the place to replace it.
