# Claude Operational Instructions

## Project Identity

- Name: UFaceNet
- Goal: Extend FaceXFormer into a unified face analysis plus face reconstruction/generation model.
- Upstream reference: `facexformer/`
- Active package target: `ufacenet/`
- Research control files: `SOUL.md`, `program.md`, and `research/*.md`
- Target venue framing: ACCV 2026
- Canonical publication repository: `https://github.com/ridhaagam/UFaceNet`

## Current Repository Reality

- `facexformer/` contains the official-style FaceXFormer inference release.
- The local model code currently exposes parsing, landmarks, head pose, attributes, age/gender/race, and visibility inference groups.
- The FaceXFormer v3 paper describes ten tasks, adding facial expression recognition and face recognition.
- UFaceNet must not assume the local code already implements all ten paper tasks.
- The original `autoresearch/` project is a language-model experiment loop. Use its autonomous loop style, not its model code or metric.

## Migration Policy

- Copy and adapt the FaceXFormer model pieces needed for UFaceNet into `ufacenet/`.
- Treat `facexformer/` as read-only provenance and baseline reference.
- New scripts, tests, training code, and evaluators must import from `ufacenet`, not from `facexformer`.
- Compatibility scripts may read `facexformer/` only to validate baseline behavior or migrate checkpoint keys.
- Preserve original FaceXFormer checkpoint loading through an explicit migration or compatibility loader.
- Do not modify `facexformer/` unless the user explicitly asks to patch the upstream snapshot.

## Primary Research Target

Add an eleventh output family to the FaceXFormer task-token design:

- Existing paper tasks: FP, LD, HPE, Attr, Age, Gen, Race, Vis, Exp, FR
- New UFaceNet task: FRec, meaning face reconstruction/generation

Do not call the new task `FR`, because `FR` already means face recognition in the FaceXFormer paper.

The FRec output must be available in the same unified forward path as the analysis tasks. It can include high-fidelity reconstruction/generation modules, but those modules must be conditioned by UFaceNet tokens/features and callable as part of one model invocation.

## Required Reading Before Edits

Read these files before architecture or benchmark changes:

1. `SOUL.md`
2. `program.md`
3. `research/FACEXFORMER_V3_NOTES.md`
4. `research/UFACENET_ARCHITECTURE.md`
5. `research/BENCHMARKS.md`
6. `research/EVIDENCE_LEDGER.md`
7. `research/BENCHMARK_LEDGER.md`
8. `research/REMOVAL_LOG.md`
9. `facexformer/network/models/facexformer.py`
10. `facexformer/network/models/transformer.py`

## Operating Mode

- Work autonomously after setup.
- Prefer concrete code, configs, benchmark scripts, and logs over discussion.
- If a dataset or checkpoint is missing, create a stub validator and document the exact missing asset.
- If a benchmark cannot run, preserve the command, error, and next fix in `runs/<run_id>/blocker.md`.
- Never silently skip a metric.

## Implementation Order

1. Reproduce or document local FaceXFormer baseline behavior from the upstream snapshot, if weights and images are available.
2. Create the `ufacenet/` package and migrate the needed FaceXFormer encoder, FaceX decoder, transformer utilities, task heads, preprocessing, serializers, and inference adapters into it.
3. Add a task registry so task ids, names, outputs, losses, and metrics are not hard-coded in multiple files.
4. Add UFaceNet reconstruction tokens without breaking old checkpoint loading.
5. Add a reconstruction/generation head behind a config flag.
6. Add a high-fidelity FRec path with RGB reconstruction, optional geometry outputs, and an optional VAE/VQ/diffusion-compatible refiner interface.
7. Add paired reconstruction output saving.
8. Add metric scripts for rFID, FID-face, LPIPS, identity cosine, and task consistency.
9. Add ablations from `research/EXPERIMENT_QUEUE.md`.
10. Generate tables and figure assets for the ACCV paper plan.

## Experiment Rules

- Every run writes to `runs/<run_id>/`.
- Every run appends one row to `research/results.tsv`.
- Every run saves `config.yaml`, `metrics.json`, `stdout.log`, and `stderr.log`.
- Every claim or proof used in docs/paper text must be added to `research/EVIDENCE_LEDGER.md`.
- Every benchmark result, table value, figure metric, or leaderboard comparison must be added to `research/BENCHMARK_LEDGER.md`.
- Every removed, deprecated, renamed, superseded, or intentionally excluded item must be added to `research/REMOVAL_LOG.md`.
- Keep changes if they improve the primary reconstruction metrics without unacceptable degradation on original tasks.
- Discard or isolate changes that improve rFID by damaging identity, geometry, or original FaceXFormer tasks.

## GitHub Documentation Policy

- The target public repo is `https://github.com/ridhaagam/UFaceNet`.
- Do not push unless explicitly instructed and the workspace is a git repository with the correct remote.
- Before any push or PR, verify that evidence, benchmark, and removal ledgers are current.
- Do not remove weak or failed results from history. Mark them `failed`, `invalid`, `superseded`, or `removed` with evidence and rationale.
- If a result is removed from a paper table, keep its old entry in `research/BENCHMARK_LEDGER.md` and add the removal reason to `research/REMOVAL_LOG.md`.

## Metrics That Must Travel Together

For reconstruction/generation:

- rFID on aligned face crops
- FID-face on aligned face crops
- optional ArcFace/FaRL embedding FID
- LPIPS
- PSNR and SSIM for paired reconstruction
- ArcFace identity cosine
- landmark NME consistency
- head pose MAE consistency
- expression consistency when expression labels are available
- NoW or MICC geometry error when 3D ground truth is available

For original tasks, keep the FaceXFormer paper metrics:

- FP: mean F1 on CelebAMask-HQ
- LD: NME on 300W
- HPE: MAE on BIWI
- Attr: accuracy on CelebA
- Age: MAE
- Gen/Race/Exp: accuracy
- Vis: recall at 80 percent precision on COFW
- FR: 1:1 verification accuracy on LFW, CFP-FP, AgeDB, CALFW, CPLFW

## Completion Criteria

Do not stop at the first working run. Stop only when all required items are complete or precisely blocked:

1. Environment validation exists.
2. Dataset validation exists.
3. FaceXFormer baseline has a reproducible command.
4. UFaceNet reconstruction/generation inference produces files.
5. rFID and FID-face evaluators run or have precise blockers.
6. Original task regression metrics are run or have precise blockers.
7. At least one ablation table is generated.
8. `research/results.tsv` has rows for completed attempts.
9. `research/EVIDENCE_LEDGER.md`, `research/BENCHMARK_LEDGER.md`, and `research/REMOVAL_LOG.md` are current.
10. ACCV table/figure plan is updated with completed versus missing evidence.

## Git And File Safety

- This workspace may not be a git repository at the root.
- Do not use destructive git commands.
- Do not overwrite the original `autoresearch/program.md`; use root `program.md` for UFaceNet.
- Keep generated checkpoints, images, and large metric caches out of source docs.

## Style

- Research-grade Python.
- Config-driven runs.
- No hidden dataset paths.
- No paper claims without metric evidence.
- No unexplained magic constants in metrics or loss weights.
- No decorative comments, stale TODOs, commented-out code, or comments that restate obvious code.
- Comments should explain why a constraint exists, how a shape convention works, or what external protocol must be preserved.
- Remove unused imports, unused functions, unused classes, unused configs, and dead branches before reporting completion.
- Add command-line `--help` text for scripts and concise docstrings for public modules/classes.
