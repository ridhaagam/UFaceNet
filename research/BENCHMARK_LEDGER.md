# Benchmark Ledger

Canonical repo: https://github.com/ridhaagam/UFaceNet

Purpose: every benchmark, table value, figure metric, leaderboard comparison, and ablation number must be recorded here. Keep failed, weak, invalid, and superseded values. Do not silently delete results.

## Status Labels

- `planned`: benchmark is defined but not run.
- `running`: run is in progress.
- `valid`: result passed protocol checks.
- `failed`: run crashed or did not produce valid metrics.
- `invalid`: result exists but protocol was wrong or data was flawed.
- `superseded`: result was replaced by a newer valid protocol or run.
- `removed`: value was removed from a table or figure, with reason logged in `research/REMOVAL_LOG.md`.

## Entry Template

```text
id:
date:
status:
run_id:
variant:
task_or_table:
dataset_split:
metric:
value:
direction:
sample_count:
command:
config:
checkpoint:
artifact_dir:
protocol_notes:
removal_or_supersedes:
owner:
notes:
```

## Planned Benchmark Packet

### BENCH-0001

```text
id: BENCH-0001
date: 2026-04-20
status: planned
run_id: pending
variant: FaceXFormer upstream baseline
task_or_table: original task regression baseline
dataset_split: pending dataset registry
metric: FP F1, LD NME, HPE MAE, Attr accuracy, Age MAE, Gen accuracy, Race accuracy, Vis recall@80P, Exp accuracy, FR verification
value: pending
direction: mixed
sample_count: pending
command: pending
config: pending
checkpoint: pending
artifact_dir: runs/<run_id>/
protocol_notes: baseline must be documented even if local upstream inference release lacks all ten task groups
removal_or_supersedes: NA
owner: future implementation agent
notes: If a task cannot run, log blocker instead of leaving blank.
```

### BENCH-0002

```text
id: BENCH-0002
date: 2026-04-20
status: planned
run_id: pending
variant: UFaceNet FRec lightweight branch
task_or_table: reconstruction and generation table
dataset_split: fixed aligned face validation split
metric: rFID, FID-face, ArcFID, LPIPS, PSNR, SSIM, ID cosine, detector failure rate
value: pending
direction: mixed
sample_count: pending
command: python scripts/eval_reconstruction.py --config configs/ufacenet_frec_frozen.yaml
config: configs/ufacenet_frec_frozen.yaml
checkpoint: pending
artifact_dir: runs/<run_id>/
protocol_notes: one-pass output from ufacenet/ required
removal_or_supersedes: NA
owner: future implementation agent
notes: Report high-fidelity refiner separately from lightweight branch.
```

### BENCH-0003

```text
id: BENCH-0003
date: 2026-04-20
status: planned
run_id: pending
variant: UFaceNet one-pass all-task output
task_or_table: one-pass output contract
dataset_split: smoke random tensor and fixed sample image
metric: output shape validation, params, FPS
value: pending
direction: pass/fail and higher FPS better
sample_count: pending
command: pytest ufacenet/tests
config: configs/ufacenet_base.yaml
checkpoint: optional
artifact_dir: runs/<run_id>/
protocol_notes: must request analysis tasks and FRec in one model invocation
removal_or_supersedes: NA
owner: future implementation agent
notes: This is a smoke benchmark, not a paper-quality metric.
```

### BENCH-0004

```text
id: BENCH-0004
date: 2026-04-20
status: valid
run_id: inference_smoke
variant: UFaceNet tiny random weights
task_or_table: one-pass output contract
dataset_split: random tensor
metric: output shape validation
value: pass
direction: pass/fail
sample_count: 1
command: python scripts/run_inference.py --tasks all --image-size 64 --output-dir runs/inference_smoke --refiner
config: inline UFaceNetConfig image_size=64 backbone=tiny refiner=true
checkpoint: none
artifact_dir: runs/inference_smoke/
protocol_notes: verifies simultaneous analysis and FRec output dictionary
removal_or_supersedes: BENCH-0003 planned smoke benchmark
owner: Codex
notes: Not a paper metric.
```

### BENCH-0005

```text
id: BENCH-0005
date: 2026-04-20
status: valid
run_id: frec_train_lfw_2step
variant: UFaceNet tiny FRec frozen-start sanity run
task_or_table: FRec training smoke
dataset_split: 512 LFW funneled images under data/processed/aligned_faces/train
metric: loss, PSNR, SSIM after 2 steps
value: loss=0.27934759855270386; psnr=9.928491592407227; ssim=0.010136268101632595
direction: lower loss better, higher PSNR/SSIM better
sample_count: 512 available; 2 train steps
command: python scripts/train_frec.py --config configs/ufacenet_frec_frozen.yaml --data-root data/processed/aligned_faces/train --output-dir runs/frec_train_lfw_2step --max-steps 2
config: configs/ufacenet_frec_frozen.yaml
checkpoint: runs/frec_train_lfw_2step/model.pt
artifact_dir: runs/frec_train_lfw_2step/
protocol_notes: verifies real-data training loop and checkpoint save
removal_or_supersedes: NA
owner: Codex
notes: Not a paper-quality training run.
```

### BENCH-0006

```text
id: BENCH-0006
date: 2026-04-20
status: valid
run_id: reconstruction_eval_lfw_2step
variant: UFaceNet tiny FRec 2-step sample
task_or_table: reconstruction smoke metric
dataset_split: real=512 LFW symlinks; generated=1 sample reconstruction
metric: color_moments_smoke Frechet score
value: 0.6524366837130209
direction: lower better
sample_count: real_count=512; generated_count=1
command: python scripts/eval_reconstruction.py --real-dir data/processed/aligned_faces/train --generated-dir runs/frec_train_lfw_2step --output runs/reconstruction_eval_lfw_2step/metrics.json
config: none
checkpoint: runs/frec_train_lfw_2step/model.pt
artifact_dir: runs/reconstruction_eval_lfw_2step/
protocol_notes: smoke-only feature model, not Inception FID or paper rFID
removal_or_supersedes: NA
owner: Codex
notes: Use only to verify evaluator plumbing.
```
