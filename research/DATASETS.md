# Dataset Registry Plan

This file defines the intended dataset registry. Do not hard-code these paths in training or benchmark code.

## UFaceNet Analysis Tasks

| Task | Train | Test | Metric |
|---|---|---|---|
| FP | CelebAMask-HQ | CelebAMask-HQ | F1-score |
| LD | 300W | 300W, 300VW | NME |
| HPE | 300W-LP | BIWI | MAE |
| Attr | CelebA | CelebA, LFWA | accuracy |
| Age | UTKFace, MORPH-II | UTKFace, MORPH-II | MAE |
| Gen | FairFace, UTKFace | FairFace, UTKFace | accuracy |
| Race | FairFace, UTKFace | FairFace, UTKFace | accuracy |
| Vis | COFW | COFW | recall at 80 percent precision |
| Exp | RAF-DB, AffectNet | RAF-DB | accuracy |
| FR | MS1MV3 | LFW, CFP-FP, AgeDB, CALFW, CPLFW | verification accuracy |

## UFaceNet FRec Datasets

Use a staged plan because licenses and data access differ.

### Stage A: Paired 2D Reconstruction

Candidate data:

- FFHQ
- CelebA-HQ
- CelebAMask-HQ
- VGGFace2, if license and access permit

Metrics:

- rFID
- FID-face
- LPIPS
- PSNR
- SSIM
- ID cosine
- task consistency

### Stage B: 3D Reconstruction

Candidate data and benchmarks:

- NoW benchmark
- MICC Florence
- AFLW2000-3D
- FaceScape, if license and access permit
- FaceVerse-style synthetic or fitted data, if license and access permit

Metrics:

- median/mean/std error in mm
- RMSE
- NME
- render consistency

### Stage C: Controlled Generation

Candidate data:

- FFHQ or CelebA-HQ aligned crops for distribution metrics.
- Identity-labeled sets only if licenses permit identity preservation evaluation.

Metrics:

- FID-face
- ArcFID or FaRLFID
- detector failure rate
- identity diversity and preservation
- prompt/control consistency if controls are used

## Required Dataset Metadata

Every dataset entry needs:

- name
- version
- root path
- license status
- download URL or acquisition note
- train/val/test split file
- number of images
- label schema
- preprocessing pipeline
- allowed publication use

## Validation Script Requirements

The dataset validator must check:

- path exists
- expected split files exist
- image count matches expected or known local count
- required labels exist
- license note exists
- sample can be loaded
- preprocessing produces expected tensor shape

When validation fails, write `runs/<run_id>/dataset_blocker.md` with exact missing paths and next steps.
