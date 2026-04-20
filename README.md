# UFaceNet Research Scaffold

UFaceNet is the working research direction for an independent unified face analysis and reconstruction/generation model trained from scratch.

Canonical publication repository: https://github.com/ridhaagam/UFaceNet

FaceXFormer v3 is used only as a cited baseline paper for task coverage and benchmark framing. UFaceNet does not vendor, import, adapt, or load FaceXFormer code or checkpoints. The current FaceXFormer v3 paper describes ten tasks: face parsing, landmark detection, head pose estimation, attributes, age, gender, race, visibility, expression, and face recognition. UFaceNet implements its own model for those tasks and adds an eleventh output family: face reconstruction/generation.

Use these files as the operating surface for autonomous research:

- `SOUL.md`: project principles, claim discipline, and completion standard.
- `CLAUDE.md`: operational instructions for Claude or any coding agent.
- `program.md`: autoresearch-style loop adapted from the original `autoresearch/` project.
- `research/FACEXFORMER_V3_NOTES.md`: paper read notes and implementation gaps.
- `research/TASK_MATRIX.md`: task coverage table with the new reconstruction column.
- `research/UFACENET_ARCHITECTURE.md`: proposed block-level design.
- `research/BENCHMARKS.md`: metrics and benchmark protocol, including rFID and face FID.
- `research/ACCV2026_PLAN.md`: submission-focused research plan and deadlines.
- `research/EXPERIMENT_QUEUE.md`: first autonomous experiment sequence.
- `research/EVIDENCE_LEDGER.md`: proof and claim evidence ledger.
- `research/BENCHMARK_LEDGER.md`: benchmark, table, and leaderboard value ledger.
- `research/REMOVAL_LOG.md`: record of removed, deprecated, or superseded code, claims, metrics, and experiments.
- `research/results.tsv`: experiment ledger template.

Primary rule: do not treat face reconstruction as a small auxiliary head. It must be a measurable output family with its own metrics, ablations, failure modes, and paper figures.

Implementation rule: implement UFaceNet fully inside the main `ufacenet/` package. Do not use `facexformer/`, FaceXFormer checkpoints, or upstream FaceXFormer runtime code.

One-pass rule: UFaceNet should support unified task prompting in a single model call. When FRec is requested, the same forward path should be able to return analysis outputs and high-fidelity reconstruction/generation outputs instead of launching a separate unrelated face generator.

Evidence rule: every proof, claim, benchmark result, table value, figure value, and removal must be documented before it is used in the paper or pushed to the canonical GitHub repo.

## Current Training-Ready State

The repo contains an active `ufacenet/` package with a one-pass task-token model, FRec reconstruction outputs, geometry outputs, a lightweight high-fidelity refiner interface, metrics, configs, scripts, and tests.

License-safe assets prepared locally:

- LFW downloaded through sklearn under `data/raw/lfw_home`.
- 512 LFW funneled images prepared as symlinks under `data/processed/aligned_faces/train`.

Large datasets and checkpoints are ignored by git. The expected paths and manual access blockers are documented under `runs/dataset_blocker.md` and `runs/dataset_validation_blocker_after_download.md`.
The committed blocker summary is [research/DATASET_BLOCKERS.md](research/DATASET_BLOCKERS.md).

## Quick Commands

```bash
python scripts/validate_environment.py
python scripts/download_datasets.py --download-lfw
python scripts/prepare_lfw_frec.py --max-images 512
python scripts/validate_datasets.py
python -m pytest -q
python scripts/run_inference.py --tasks all --image-size 64 --refiner
python scripts/train_frec.py --config configs/ufacenet_frec_frozen.yaml --data-root data/processed/aligned_faces/train --max-steps 2
python scripts/check_image_outputs.py runs/inference_fixed_image_v2 runs/frec_train_fixed_v2_2step --min-std 0.03
```

For full FRec training, increase `training.max_steps` in `configs/ufacenet_frec_frozen.yaml` or pass a larger `--max-steps` value after the dataset paths are confirmed.
