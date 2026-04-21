# Codex 5090 Handoff Prompt

You are working in `/home/agam/documents/UFaceNet` on the helper node with a 5090-class GPU.

Your job is to continue UFaceNet as a serious research codebase, not as a smoke-test scaffold.

Read these files first:

1. `README.md`
2. `SOUL.md`
3. `CLAUDE.md`
4. `program.md`
5. `research/DATASETS.md`
6. `research/DATASET_BLOCKERS.md`
7. `research/UFACENET_ARCHITECTURE.md`
8. `research/BENCHMARKS.md`
9. `research/EXPERIMENT_QUEUE.md`
10. `research/EVIDENCE_LEDGER.md`
11. `research/BENCHMARK_LEDGER.md`
12. `research/REMOVAL_LOG.md`

Current truth:

- UFaceNet is independent and must not use FaceXFormer runtime code or checkpoints.
- The repo currently contains a one-pass model scaffold, task registry, smoke-safe FRec path, configs, validators, and tests.
- The previous LFW reconstruction outputs were not research-standard reconstructions.
- LFW is acceptable only for smoke validation, not for final FRec claims.
- Full multi-task supervised training is not implemented yet.
- Paper-grade FRec training is not implemented yet.

What you must do:

1. Verify the environment on this machine.
2. Verify which datasets are already present under `data/raw/`.
3. Do not claim readiness for any task unless the training loop, dataset adapter, and benchmark runner all exist.
4. Turn the repo from scaffold into a research-grade training codebase.

Priority implementation order:

1. Build real dataset adapters and manifests for:
   - FFHQ
   - CelebAMask-HQ
   - 300W
   - 300W-LP
   - BIWI
   - CelebA
   - FairFace
   - COFW
   - RAF-DB
   - MS1MV3
   - NoW
   - AFLW2000-3D
2. Replace the current FRec smoke loss with a real staged objective:
   - pixel loss
   - LPIPS
   - identity loss
   - parsing consistency
   - landmark consistency
   - pose consistency
   - optional geometry supervision when data is available
3. Add a real FRec trainer with:
   - train/val split support
   - checkpoint resume
   - mixed precision
   - gradient accumulation
   - metric logging
   - qualitative sample grids
4. Add benchmark runners for:
   - rFID / FID-face
   - LPIPS / PSNR / SSIM
   - identity cosine
   - NoW / AFLW2000-3D geometry evaluation
5. Add actual supervised training pipelines for the ten analysis tasks.
6. Keep the one-pass UFaceNet design so analysis tasks and FRec remain part of one unified forward path.

Hard rules:

- Do not reintroduce `facexformer/`.
- Do not use FaceXFormer checkpoints.
- Do not describe smoke outputs as research results.
- Do not leave fake benchmark numbers or placeholder paper claims.
- Every completed result must be logged in `research/results.tsv`.
- Every claim or benchmark value must be recorded in `research/EVIDENCE_LEDGER.md` and `research/BENCHMARK_LEDGER.md`.
- Every dropped idea, removed file, or rejected baseline must be recorded in `research/REMOVAL_LOG.md`.

Expected first commands:

```bash
cd /home/agam/documents/UFaceNet
python scripts/validate_environment.py
python scripts/validate_datasets.py --output runs/helper_dataset_report.json --blocker runs/helper_dataset_blocker.md
python -m pytest -q
```

Definition of progress:

- not just passing tests
- not just producing nonblank images
- not just loading data

Real progress means a task can be trained and evaluated with documented data, metrics, and reproducible commands.

Before stopping, report:

- what is genuinely implemented
- what is still blocked
- which datasets are present
- which training loop is now real
- which benchmark can now be run end to end
