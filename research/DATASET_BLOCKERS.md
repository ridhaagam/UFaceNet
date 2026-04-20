# Dataset Readiness And Blockers

Canonical repo: https://github.com/ridhaagam/UFaceNet

## Downloaded Locally

- LFW through sklearn: `data/raw/lfw_home`
- LFW funneled FRec training split: `data/processed/aligned_faces/train`
- FaceXFormer upstream checkpoint: `checkpoints/facexformer/ckpts/model.pt`

These files are intentionally not tracked by git because datasets and checkpoints are large. Recreate them with:

```bash
python scripts/download_datasets.py --download-lfw --download-facexformer-ckpt
python scripts/prepare_lfw_frec.py --max-images 512
```

## Manual Access Required

The following are required for full FaceXFormer/UFaceNet paper-grade training or evaluation, but require licenses, requests, or manual terms acceptance:

| Dataset | Expected Path | Tasks | Source | Blocker |
|---|---|---|---|---|
| CelebAMask-HQ | `data/raw/CelebAMask-HQ` | FP, FRec | https://github.com/switchablenorms/CelebAMask-HQ | Follow upstream and CelebA license constraints. |
| 300W | `data/raw/300W` | LD | https://ibug.doc.ic.ac.uk/resources/300-W/ | Requires acceptance of dataset terms. |
| BIWI | `data/raw/BIWI` | HPE | https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html | Academic access constraints. |
| CelebA | `data/raw/CelebA` | Attr | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html | Large Google Drive style distribution often needs manual handling. |
| FairFace | `data/raw/FairFace` | Age, Gen, Race | https://github.com/joojs/fairface | Follow upstream license and download instructions. |
| RAF-DB | `data/raw/RAF-DB` | Exp | http://www.whdeng.cn/RAF/model1.html | Requires request or registration. |
| AffectNet | `data/raw/AffectNet` | Exp | http://mohammadmahoor.com/affectnet/ | Requires request or registration. |
| MS1MV3 | `data/raw/MS1MV3` | FR | https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_ | Large identity data with redistribution constraints. |
| NoW | `data/raw/NoW` | FRec geometry | https://github.com/soubhiksanyal/now_evaluation | Requires benchmark data access and official protocol. |

## Validation

Run:

```bash
python scripts/validate_datasets.py --output runs/dataset_report_final.json --blocker runs/dataset_validation_blocker_final.md
```

Current local validation shows LFW is available and the restricted datasets above are still blocked until manually acquired.
