# Full Claude Prompt For UFaceNet Repo Build

Copy this prompt into Claude from `/home/dsic/Documents/agam/ufacenet`.

```text
You are working in /home/dsic/Documents/agam/ufacenet.

Your job is to turn this workspace into a full UFaceNet research repo based on FaceXFormer, with one unified model that keeps FaceXFormer-style facial analysis and adds high-fidelity face reconstruction/generation as a first-class output.

Canonical publication repository: https://github.com/ridhaagam/UFaceNet

Do not stop at a plan. Read the repo, implement the scaffold/code, add validators, add metrics, add smoke tests, and leave clear run commands. If a dataset/checkpoint is missing, create the code path and a precise blocker file instead of stopping.

First read these files in order:
1. README.md
2. SOUL.md
3. CLAUDE.md
4. program.md
5. research/FACEXFORMER_V3_NOTES.md
6. research/TASK_MATRIX.md
7. research/UFACENET_ARCHITECTURE.md
8. research/BENCHMARKS.md
9. research/EXPERIMENT_QUEUE.md
10. research/EVIDENCE_LEDGER.md
11. research/BENCHMARK_LEDGER.md
12. research/REMOVAL_LOG.md
13. facexformer/network/models/facexformer.py
14. facexformer/network/models/transformer.py
15. facexformer/inference.py

Core objective:
Build UFaceNet: a FaceXFormer-based unified facial model with the original FaceXFormer task-token idea plus a new FRec task family for face reconstruction/generation. FR means Face Recognition. The new task must be called FRec, not FR.

Critical migration requirement:
Do not use facexformer/ as the active runtime package. Treat facexformer/ as a read-only upstream snapshot and baseline reference. Copy and adapt all FaceXFormer pieces needed for UFaceNet into the main project package under ufacenet/. New scripts, tests, training code, inference code, and evaluators must import from ufacenet, not from facexformer. Direct imports from facexformer are allowed only in migration or baseline comparison scripts.

Evidence and benchmark documentation requirement:
Every proof, claim, benchmark, table value, figure metric, leaderboard value, and removal must be documented before it is used in the README, paper, figures, or pushed to GitHub. Use these ledgers:
- research/EVIDENCE_LEDGER.md for proof, claim, source-backed facts, assumptions, and verification status
- research/BENCHMARK_LEDGER.md for benchmarks, table values, figure metrics, run ids, splits, configs, commands, and status
- research/REMOVAL_LOG.md for removed, deprecated, renamed, superseded, rejected, or intentionally excluded code, claims, metrics, methods, datasets, and results

Do not silently delete weak or failed results. Mark them failed, invalid, superseded, removed, or rejected with a reason and replacement.

One-pass requirement:
UFaceNet must behave like a FaceXFormer-style unified task-token model. One model invocation should be able to request all tasks and return analysis outputs plus FRec outputs. Do not implement face generation as a separate unrelated pipeline. The high-fidelity reconstruction/generation path must be conditioned by UFaceNet shared features and FRec/task tokens.

The final repo should support:
- Face parsing
- Landmark detection
- Head pose estimation
- Attributes
- Age
- Gender
- Race
- Visibility
- Facial expression recognition
- Face recognition
- High-fidelity face reconstruction/generation via FRec

Important reality check:
The local facexformer code is an inference release and may not expose all ten FaceXFormer v3 tasks. Do not pretend it does. Preserve the existing local behavior as a documented baseline, then create the extensible UFaceNet code needed to support the full task set.

Implementation requirements:
1. Do not destroy or rewrite the original facexformer folder.
2. Create the active UFaceNet code in ufacenet/.
3. Copy/adapt required FaceXFormer code into ufacenet/:
   - encoder/backbone wrapper
   - FaceX decoder
   - two-way transformer utilities
   - task heads for released local tasks
   - preprocessing and serializers
   - checkpoint compatibility loader
4. Make all new runtime scripts import from ufacenet.
5. Add a task registry so task ids, names, heads, losses, metrics, serializers, and datasets are not hard-coded everywhere.
6. Add FRec tokens:
   - Start with T_frec.
   - Leave clean support for optional T_geom, T_tex, and T_render.
7. Add a reconstruction-consistency block:
   - It must connect FRec to face identity, landmarks, pose, parsing, expression, and geometry consistency.
   - It must not be just a detached image decoder.
8. Add high-fidelity FRec outputs:
   - reconstructed RGB
   - optional generated/refined RGB
   - optional depth
   - optional normals
   - optional mesh or 3DMM parameters
   - optional UV texture/albedo
   - optional face mask
   - camera/render metadata when geometry is enabled
9. Add a high-fidelity generation/reconstruction path:
   - Baseline: convolutional RGB reconstruction decoder from UFaceNet features.
   - Geometry path: depth/normal/mesh or 3DMM-compatible outputs if feasible.
   - Optional/refiner path: VAE/VQ/diffusion-compatible conditioned generator interface.
   - If pretrained generation weights are unavailable, build the interface, config, smoke test, and blocker docs.
10. Preserve old checkpoint loading:
   - Existing FaceXFormer checkpoints should load for old tasks through a migration-aware compatibility loader.
   - New FRec weights may initialize randomly.
   - Use explicit warnings for missing new keys and unexpected old keys.

Repository structure to create or complete:
- ufacenet/
  - __init__.py
  - tasks.py
  - model.py
  - decoder.py
  - transformer.py
  - reconstruction.py
  - checkpoint.py
  - losses.py
  - metrics/
  - data/
  - configs/
  - scripts/
  - tests/
- scripts/
  - validate_environment.py
  - validate_datasets.py
  - run_inference.py
  - train_frec.py
  - eval_reconstruction.py
  - eval_original_tasks.py
  - make_qualitative_grid.py
- configs/
  - ufacenet_base.yaml
  - ufacenet_frec_frozen.yaml
  - ufacenet_frec_consistency.yaml
  - datasets.example.yaml

Code quality standard:
- Research-grade Python only.
- No decorative section banners.
- No commented-out experiments or dead code.
- No stale TODOs without a precise blocker file.
- No comments that simply restate obvious code.
- Comments should explain why a constraint exists, how a shape convention works, numerical stability, checkpoint migration, or benchmark protocol.
- Public modules/classes/functions need concise docstrings.
- Scripts need argparse help text and useful errors.
- Remove unused imports, unused functions, unused classes, unused configs, and dead branches before completion.
- Prefer explicit names and small functions over long explanatory comments.

Metrics that must be implemented or stubbed with precise blockers:
- rFID on aligned face crops
- FID-face on aligned face crops
- ArcFace or FaRL embedding FID if a checkpoint is available
- LPIPS
- PSNR
- SSIM
- ArcFace identity cosine
- landmark consistency NME
- head pose consistency MAE
- parsing consistency
- expression consistency when labels/model exist
- NoW/MICC/AFLW2000-3D geometry metrics when data is available
- original FaceXFormer task deltas
- FPS and parameter count

Evaluation rules:
- Use fixed splits.
- Use fixed crop/alignment.
- Report detector failure rate.
- Report sample count.
- Do not cherry-pick qualitative samples.
- Save all outputs under runs/<run_id>/.
- Append every experiment to research/results.tsv.

High-fidelity output goal:
The FRec branch should be designed to produce publication-grade face outputs, not just blurry autoencoder reconstructions. Use staged implementation:
1. frozen UFaceNet features + RGB decoder
2. identity and LPIPS losses
3. landmark/pose/parsing consistency
4. geometry-aware outputs if possible
5. optional generative refiner conditioned on UFaceNet tokens

Training stages to encode:
- Stage 0: baseline FaceXFormer behavior validator
- Stage 1: frozen FRec reconstruction decoder
- Stage 2: reconstruction-consistency losses
- Stage 3: partial joint fine-tuning
- Stage 4: optional high-fidelity generative refiner

Smoke tests:
- ufacenet imports without importing facexformer
- model instantiates
- task registry resolves all task names
- FRec enabled and disabled modes both work
- one forward call can request all analysis tasks and FRec
- old FaceXFormer checkpoint compatibility path is present and tested with a dummy state dict
- forward pass on random tensor works
- reconstruction decoder returns image-shaped output
- optional geometry outputs have documented shapes when enabled
- metrics can run on tiny synthetic image folders
- scripts print useful errors for missing datasets/checkpoints

Documentation updates:
- Update README.md with setup, repo structure, commands, migration policy, one-pass output policy, and current blockers.
- Update research/EXPERIMENT_QUEUE.md with what was completed and what remains.
- Update research/results.tsv with any actual smoke runs.
- Update research/EVIDENCE_LEDGER.md for every proof, claim, source-backed statement, and paper assertion.
- Update research/BENCHMARK_LEDGER.md for every benchmark result, table value, figure metric, and leaderboard comparison.
- Update research/REMOVAL_LOG.md for every removed, deprecated, renamed, superseded, rejected, or intentionally excluded item.
- Add runs/<run_id>/blocker.md if data or weights are missing.
- Keep SOUL.md and CLAUDE.md consistent if implementation changes the workflow.

GitHub policy:
- Target repo is https://github.com/ridhaagam/UFaceNet.
- Do not push unless explicitly instructed and the current workspace is a git repository with the correct remote.
- Before any push or PR, verify the evidence, benchmark, and removal ledgers are current.
- If this local workspace is not a git repo, prepare the files and state that they are ready to be pushed rather than claiming they were pushed.

Do not stop after writing files. Run the validators and tests you create. If something fails, debug and fix it. If failure is due to missing external assets, document the exact missing file, expected path, and command the user should run.

Safety and ethics:
This is biometric research. Do not build a face-swapping or impersonation demo. The FRec/generation branch is for reconstruction, representation analysis, controlled generation research, and benchmarked face fidelity.

When finished, report:
- files created/changed
- commands run
- tests/validators passed
- blockers
- next experiment to run
```
