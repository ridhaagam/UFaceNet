# Benchmarks And Metrics

## Source Baseline

FaceXFormer v3 evaluates ten tasks with task-specific metrics:

- FP on CelebAMask-HQ: F1-score.
- LD on 300W and 300VW: NME.
- HPE on BIWI: MAE.
- Attr on CelebA/LFWA: accuracy.
- Age on UTKFace/MORPH-II style age sets: MAE.
- Gen/Race on FairFace/UTKFace: accuracy.
- Exp on RAF-DB/AffectNet: accuracy.
- FR on LFW, CFP-FP, AgeDB, CALFW, CPLFW: 1:1 verification accuracy.
- Vis on COFW: recall at 80 percent precision.

UFaceNet must keep these metrics and add reconstruction/generation metrics.

## New FRec Metrics

### rFID

Definition: reconstruction FID between real aligned face crops and their paired reconstructions.

Protocol:

- Use a fixed evaluation split.
- Use the same face detector, alignment, crop size, and image resolution for real and reconstructed images.
- Compute FID between all real crops and all reconstructed crops.
- Report sample count.
- Lower is better.

Recommended label in tables:

```text
rFID(face crops) down
```

### FID-face

Definition: FID computed on aligned face crops for generated or reconstructed face images. This is different from whole-image FID when backgrounds dominate.

Protocol:

- Detect and align generated faces.
- Use the same crop transform for real and generated sets.
- Report detector failure rate separately.
- Compute Inception FID on the aligned face crops.
- Lower is better.

Recommended label:

```text
FID-face down
```

### ArcFace/FaRL Embedding FID

Definition: Frechet distance in a face embedding space instead of Inception space.

Why: Inception FID can miss identity and face-manifold drift. ArcFace or FaRL features are more face-specific.

Protocol:

- Use a frozen face embedding model.
- Extract embeddings for real and reconstructed/generated crops.
- Compute the same Frechet formula over embedding means and covariances.
- Report the embedding model name and checkpoint.

Recommended labels:

```text
ArcFID down
FaRLFID down
```

Do not replace ordinary FID-face with this metric. Report both.

### Paired Reconstruction Fidelity

Use these for image-to-image reconstruction:

- LPIPS down
- PSNR up
- SSIM up
- L1 or L2 down

These metrics do not prove perceptual realism or identity preservation, so they must be reported with identity and distribution metrics.

### High-Fidelity FRec Requirements

FRec outputs should be evaluated as high-fidelity face outputs, not only as pixel reconstructions. A valid high-fidelity packet includes:

- aligned RGB reconstruction or generation;
- optional refined RGB output when a generative refiner is enabled;
- identity preservation through ArcFace or equivalent embeddings;
- perceptual quality through LPIPS and FID-face;
- reconstruction distribution quality through rFID;
- structural consistency through landmarks, pose, parsing, expression, and geometry metrics;
- detector failure rate for generated/refined faces.

The high-fidelity refiner must be reported separately from the lightweight reconstruction branch so runtime and quality tradeoffs are clear.

### Identity Preservation

Use a frozen face recognition model:

- ArcFace cosine similarity up.
- Verification acceptance at fixed thresholds up.
- Optional false match/false non-match analysis if identities are labeled.

Report identity preservation for:

- full validation set;
- pose slices;
- expression slices;
- demographic slices where labels and licenses permit.

### Task Consistency

Run the same or frozen task evaluators on input and reconstruction:

- landmark NME between input-derived landmarks and reconstruction-derived landmarks;
- head pose MAE between input and reconstruction;
- parsing F1 or mIoU between input-derived parsing and reconstruction-derived parsing;
- expression agreement;
- attribute agreement.

These are not ground-truth metrics unless labels exist. Mark them as consistency metrics.

### 3D Reconstruction Metrics

Use when the FRec branch predicts geometry:

- NoW benchmark: median, mean, and std error in mm.
- MICC Florence: RMSE in mm, if protocol is available.
- AFLW2000-3D: NME for dense/sparse alignment.
- FaceScape/FaceVerse-style held-out scans: point-to-surface or vertex error after documented alignment.
- Render metrics: silhouette IoU, depth MAE, normal angular error if ground truth exists.

## FID Formula

For feature distributions approximated by Gaussians:

```text
FID = ||mu_r - mu_g||_2^2 + Tr(Sigma_r + Sigma_g - 2 * sqrt(Sigma_r * Sigma_g))
```

Where `r` is real and `g` is generated or reconstructed.

## Primary Tables For ACCV

### Table 1: Task Coverage

Use `research/TASK_MATRIX.md`.

### Table 2: Original Face Tasks

Rows:

- specialized models from FaceXFormer paper;
- Faceptor;
- FaceXFormer;
- UFaceNet without FRec training;
- UFaceNet full.

Columns:

- FP mean F1
- LD NME
- HPE MAE
- Attr accuracy
- Age MAE
- Gen accuracy
- Race accuracy
- Vis recall at 80 percent precision
- Exp accuracy
- FR mean verification accuracy
- FPS FP32
- params

### Table 3: Reconstruction And Generation

Rows:

- simple autoencoder baseline;
- DECA/MICA/3DDFA-style reconstruction baseline where applicable;
- FaceXFormer plus separate recon model;
- UFaceNet isolated FRec decoder;
- UFaceNet with reconstruction-consistency block;
- UFaceNet with generative refiner.

Columns:

- rFID down
- FID-face down
- ArcFID down
- LPIPS down
- PSNR up
- SSIM up
- ID cosine up
- NoW median mm down
- detector failure rate down
- FPS FP32

### Table 4: Ablations

Rows:

- no FRec token
- FRec token only
- plus reconstruction-consistency block
- plus identity loss
- plus landmark/pose consistency
- plus parsing consistency
- geometry tokens
- generative refiner

Columns:

- rFID
- FID-face
- ID cosine
- NoW median
- average original task delta
- FPS

### Table 5: Robustness And Bias

Rows:

- occlusion
- profile
- expression-rich
- low resolution
- low light
- demographic slices where labels permit

Columns:

- rFID
- ID cosine
- landmark consistency
- pose consistency
- task deltas

## Benchmark Execution Rules

- Fixed evaluation split per dataset.
- Fixed crop/alignment code.
- Fixed image resolution.
- Fixed face detector and detector confidence threshold.
- Report detector failure rates.
- Report sample counts.
- Report bootstrap confidence intervals for key metrics when feasible.
- Preserve raw per-sample metrics for error analysis.

## Minimum Viable Benchmark Packet

For a first ACCV-quality internal checkpoint:

1. FaceXFormer baseline inference and runtime.
2. UFaceNet one-pass output on a fixed validation split, including analysis tasks and FRec from the same model interface.
3. rFID(face crops), FID-face, LPIPS, PSNR, SSIM.
4. ArcFace ID cosine.
5. Landmark and pose consistency.
6. Original task regression on at least FP, LD, HPE, Attr, Age/Gen/Race, and Vis.
7. Qualitative grid with no cherry-picking.

## Sources

- FaceXFormer v3: https://arxiv.org/pdf/2403.12960v3
- FID paper: https://arxiv.org/abs/1706.08500
- NoW evaluation repository: https://github.com/soubhiksanyal/now_evaluation
- DECA: https://arxiv.org/abs/2012.04012
- MICA: https://arxiv.org/abs/2204.06607
- EMOCA: https://arxiv.org/abs/2204.11312
- 3DDFA-V2: https://arxiv.org/abs/2009.09960
