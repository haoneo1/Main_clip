# Frozen I-JEPA Baseline Run Record

## Run Summary

- Experiment name: `ijepa_h14_224_frozen_frozen_ijepa`
- Script: `experiments/train_frozen_ijepa_baseline.py`
- Backbone arch: `ijepa_vit_h14_224_in1k`
- Backbone ckpt: `/home/mig/Documents/SBIR_Data/IN1K-vit.h.14-300e.pth.tar`
- Batch size: `16`
- Freeze backbone: `True`
- Adapter hidden dim: `512`
- Projection dim: `512`
- Learning rate (`frozen_lr`): `1e-4`
- SigReg: `use_sigreg=True`, `lambda_sigreg=0.1`

## Artifacts

- Checkpoint dir:
  - `/home/mig/Documents/SBIR_Data/saved_models/ijepa_h14_224_frozen_frozen_ijepa`
- TensorBoard dir:
  - `/home/mig/Documents/SBIR/Main_clip/tb_logs/ijepa_h14_224_frozen_frozen_ijepa/version_0`
- Current best checkpoint:
  - `/home/mig/Documents/SBIR_Data/saved_models/ijepa_h14_224_frozen_frozen_ijepa/best-epoch=05-mAP=0.2471.ckpt`
- Last checkpoint:
  - `/home/mig/Documents/SBIR_Data/saved_models/ijepa_h14_224_frozen_frozen_ijepa/last.ckpt`

## Epoch Metrics (from TensorBoard scalars)

> Note: Lightning epoch indexing here starts from `0`.

| Epoch | mAP | val_loss | train_loss_epoch | train_triplet |
|---:|---:|---:|---:|---:|
| 0 | 0.1826 | 0.1015 | 0.1966 | 0.0997 |
| 1 | 0.2052 | 0.0988 | 0.1625 | 0.0661 |
| 2 | 0.2107 | 0.0994 | 0.1506 | 0.0544 |
| 3 | 0.2067 | 0.0978 | 0.1459 | 0.0498 |
| 4 | 0.2340 | 0.0903 | 0.1421 | 0.0460 |
| 5 | 0.2471 | 0.0878 | 0.1392 | 0.0432 |
| 6 | 0.2357 | 0.0894 | 0.1366 | 0.0406 |
| 7 | 0.2357 | 0.0940 | 0.1354 | 0.0395 |

## Current Observation

- The run already passed several epochs and reached a best mAP of `0.2471` at epoch `5`.
- Training process is/was active while these metrics were collected.
- Trend suggests early gains with mild overfitting signs after epoch 5 (mAP drop + val_loss rebound).
