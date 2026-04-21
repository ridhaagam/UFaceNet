# Dataset Registry Plan

This file defines the intended dataset registry. Do not hard-code these paths in training or benchmark code.

## Current Status

This repo does not yet implement full paper-grade multi-task training. What exists now is:

- a one-pass UFaceNet model scaffold
- a smoke-trainable FRec path
- task registry, configs, validators, metrics scaffolding, and tests

What does not exist yet:

- full supervised training loops for all ten analysis tasks
- a paper-grade reconstruction pipeline with identity/perceptual/3D supervision
- final benchmark protocol code for each dataset

So the correct statement today is:

`UFaceNet is partially set up, but it is not yet in a state where all tasks become research-standard just by dropping in the datasets.`

## Master Dataset Table

| Group | Dataset | Tasks | Role | Priority | Expected Path | Main Labels / Assets | Core Metrics | Access | Official Source |
|---|---|---|---|---|---|---|---|---|---|
| Analysis | CelebAMask-HQ | FP, FRec | train/eval | required | `data/raw/CelebAMask-HQ` | face parsing masks, aligned HQ faces | mIoU, F1, parsing consistency | manual license | https://github.com/switchablenorms/CelebAMask-HQ |
| Analysis | 300W | LD | train/eval | required | `data/raw/300W` | 68-point landmarks | NME | manual license | https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/ |
| Analysis | 300VW | LD | eval | recommended | `data/raw/300VW` | video landmarks | NME, temporal stability | manual license | https://ibug.doc.ic.ac.uk/resources/300-VW/ |
| Analysis | 300W-LP | HPE, LD, FRec geometry warm start | train | required | `data/raw/300W-LP` | pose, 3D-aware landmarks | pose MAE, NME | manual license | http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm |
| Analysis | BIWI | HPE | eval | required | `data/raw/BIWI` | head pose labels | pitch/yaw/roll MAE | manual license | https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html |
| Analysis | CelebA | Attr | train/eval | required | `data/raw/CelebA` | 40 binary attributes | mean accuracy | manual or torchvision | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html |
| Analysis | LFWA | Attr | eval | optional | `data/raw/LFWA` | attribute labels | mean accuracy | manual license | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html |
| Analysis | FairFace | Age, Gen, Race | train/eval | required | `data/raw/FairFace` | age, gender, race | age MAE, gender acc, race acc | manual license | https://github.com/joojs/fairface |
| Analysis | UTKFace | Age, Gen, Race | train/eval | recommended | `data/raw/UTKFace` | age, gender, race | age MAE, gender acc, race acc | public | https://susanqq.github.io/UTKFace/ |
| Analysis | MORPH-II | Age | train/eval | optional | `data/raw/MORPH-II` | age labels | MAE | manual license | https://www.faceaginggroup.com/morph/ |
| Analysis | COFW | Vis, LD | train/eval | required | `data/raw/COFW` | occlusion / visibility labels, landmarks | recall@80P, NME | manual license | https://data.caltech.edu/records/bc0bf-nc666 |
| Analysis | RAF-DB | Exp | train/eval | required | `data/raw/RAF-DB` | expression labels | accuracy, F1 | manual license | http://www.whdeng.cn/RAF/model1.html |
| Analysis | AffectNet | Exp | train/eval | recommended | `data/raw/AffectNet` | expression, valence, arousal | accuracy, F1 | manual license | http://mohammadmahoor.com/affectnet/ |
| Analysis | MS1MV3 | FR | train | required | `data/raw/MS1MV3` | identity labels | verification / identification | manual license | https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_ |
| Analysis Eval | LFW | FR, smoke FRec only | eval | required | `data/raw/lfw_home` | identity pairs | verification accuracy | public | https://vis-www.cs.umass.edu/lfw/ |
| Analysis Eval | CFP-FP | FR | eval | required | `data/raw/CFP-FP` | frontal-profile pairs | verification accuracy | manual license | http://www.cfpw.io/ |
| Analysis Eval | AgeDB-30 | FR | eval | required | `data/raw/AgeDB-30` | age-separated identity pairs | verification accuracy | manual license | https://ibug.doc.ic.ac.uk/resources/agedb/ |
| Analysis Eval | CALFW | FR | eval | recommended | `data/raw/CALFW` | cross-age pairs | verification accuracy | manual license | http://www.whdeng.cn/CALFW/ |
| Analysis Eval | CPLFW | FR | eval | recommended | `data/raw/CPLFW` | cross-pose pairs | verification accuracy | manual license | http://www.whdeng.cn/CPLFW/ |
| FRec 2D | FFHQ | FRec | train/eval | required | `data/raw/FFHQ` | aligned HQ face images | rFID, FID-face, LPIPS, PSNR, SSIM | public via official tooling | https://github.com/NVlabs/ffhq-dataset |
| FRec 2D | CelebA-HQ | FRec | train/eval | recommended | `data/raw/CelebA-HQ` | aligned HQ face images | rFID, FID-face, LPIPS | manual / TFDS route | https://www.tensorflow.org/datasets/catalog/celeb_a_hq |
| FRec 2D | VGGFace2 | FRec, FR | train/eval | recommended | `data/raw/VGGFace2` | identity-rich face images | ID cosine, FID-face | manual license | https://github.com/ox-vgg/vgg_face2 |
| FRec 3D | NoW | FRec geometry | eval | required | `data/raw/NoW` | 3D scan benchmark protocol | median scan error, mm | manual license | https://now.is.tue.mpg.de/dataset.html |
| FRec 3D | AFLW2000-3D | HPE, FRec geometry | eval | required | `data/raw/AFLW2000-3D` | 3D landmarks / pose | NME, MAE | manual license | http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm |
| FRec 3D | MICC Florence | FRec geometry | eval | recommended | `data/raw/MICC_Florence` | 3D face scans | RMSE, scan error | manual license | https://www.micc.unifi.it/resources/datasets/florence-3d-faces/ |
| FRec 3D | FaceScape | FRec geometry, expression-aware recon | train/eval | optional | `data/raw/FaceScape` | high-quality 3D faces, expressions | geometry + render metrics | manual license | https://nju-3dv.github.io/projects/FaceScape/ |

## Minimum Download Set By Goal

| Goal | Minimum Datasets |
|---|---|
| Bring up parsing, landmarks, pose, attributes, age/gender/race, visibility, expression, recognition | CelebAMask-HQ, 300W, 300W-LP, BIWI, CelebA, FairFace, COFW, RAF-DB, MS1MV3, LFW |
| Build a credible 2D FRec baseline | FFHQ, CelebAMask-HQ, VGGFace2 or CelebA-HQ |
| Claim strong 3D reconstruction | NoW, AFLW2000-3D, MICC Florence, optionally FaceScape |
| Run only smoke tests | LFW |

## What LFW Can And Cannot Do

LFW is acceptable for:

- smoke-loading the dataloader
- verifying that the FRec branch can optimize
- debugging image save paths
- running crude identity-preserving reconstruction sanity checks

LFW is not acceptable for:

- research-standard high-fidelity reconstruction claims
- strong generation quality claims
- geometry reconstruction claims
- final paper tables for FRec

The earlier LFW outputs were therefore not valid as research-grade reconstructions. They were only proof that the code path executed.

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
