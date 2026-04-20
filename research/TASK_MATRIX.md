# Task Matrix

Source baseline: FaceXFormer v3 Table 1, https://arxiv.org/pdf/2403.12960v3

Column names:

- FP: face parsing
- LD: landmark detection
- HPE: head pose estimation
- Attr: facial attribute recognition
- Age: age estimation
- Gen: gender estimation
- Race: race estimation
- Vis: face visibility prediction
- Exp: facial expression recognition
- FR: face recognition
- FRec: face reconstruction/generation, the new UFaceNet output

Important naming rule: `FR` is already face recognition. Use `FRec` for face reconstruction.

## Publication-Framing Matrix

This matrix is the one to use for the UFaceNet paper framing. It preserves the FaceXFormer v3 ten-task comparison and adds the new FRec column.

| Method | FP | LD | HPE | Attr | Age | Gen | Race | Vis | Exp | FR | FRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DML-CSR | Y | N | N | N | N | N | N | N | N | N | N |
| FP-LIIF | Y | N | N | N | N | N | N | N | N | N | N |
| SegFace | Y | N | N | N | N | N | N | N | N | N | N |
| Wing | N | Y | N | N | N | N | N | N | N | N | N |
| HRNet | N | Y | N | N | N | N | N | N | N | N | N |
| WHENet | N | N | Y | N | N | N | N | N | N | N | N |
| TriNet | N | N | Y | N | N | N | N | N | N | N | N |
| img2pose | N | N | Y | N | N | N | N | N | N | N | N |
| TokenHPE | N | N | Y | N | N | N | N | N | N | N | N |
| SSPL | N | N | N | Y | N | N | N | N | N | N | N |
| VOLO-D1 | N | N | N | N | Y | N | N | N | N | N | N |
| DLDL-v2 | N | N | N | N | Y | N | N | N | N | N | N |
| 3DDE | N | N | N | N | N | N | N | Y | N | N | N |
| MNN | N | N | N | N | N | N | N | Y | N | N | N |
| KTN | N | N | N | N | N | N | N | N | Y | N | N |
| DMUE | N | N | N | N | N | N | N | N | Y | N | N |
| CosFace | N | N | N | N | N | N | N | N | N | Y | N |
| ArcFace | N | N | N | N | N | N | N | N | N | Y | N |
| AdaFace | N | N | N | N | N | N | N | N | N | Y | N |
| SSP+SSG | Y | N | N | Y | N | N | N | N | N | N | N |
| Hetero-FAE | N | N | N | Y | Y | Y | Y | N | Y | N | N |
| FairFace | N | N | N | N | Y | Y | Y | N | N | N | N |
| MiVOLO | N | N | N | N | Y | Y | N | N | N | N | N |
| MTL-CNN | N | Y | N | Y | N | N | N | N | Y | N | N |
| ProS | Y | Y | N | Y | N | N | N | N | N | N | N |
| FaRL | Y | Y | N | Y | Y | Y | Y | N | N | N | N |
| HyperFace | N | Y | Y | N | N | Y | N | Y | Y | N | N |
| AllinOne | N | Y | Y | N | Y | Y | N | Y | Y | N | N |
| SwinFace | N | Y | N | Y | Y | N | Y | N | Y | N | N |
| QFace | N | Y | N | Y | Y | Y | Y | N | Y | N | N |
| Faceptor | Y | Y | N | Y | Y | Y | Y | N | Y | N | N |
| FaceXFormer | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | N |
| UFaceNet target | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |

## UFaceNet Addendum Table

This smaller table should appear near the method figure to make the new output explicit.

| Model | Unified analysis tasks | Reconstruction/generation | Reconstruction metrics | Real-time target |
|---|---:|---:|---|---:|
| FaceXFormer v3 | 10 | N | none | 33.21 FPS FP32 reported |
| UFaceNet baseline | 10 | Y | rFID, FID-face, LPIPS, ID cosine, geometry error | within 25 percent of baseline unless otherwise stated |
| UFaceNet full | 10 | Y | all baseline metrics plus ablations and robustness | report FP32 and FP16 |

## Verification TODO

Before paper submission, re-check every coverage mark against the camera-ready FaceXFormer paper and each cited method. The table above is a scaffold for ACCV planning, not a substitute for final citation verification.
