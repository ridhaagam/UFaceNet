# UFaceNet Autoresearch Program

This is the autoresearch-style operating program for UFaceNet. It adapts the loop in `autoresearch/program.md` from a fixed language-model benchmark into an autonomous face-analysis and reconstruction research loop.

Canonical publication repository: `https://github.com/ridhaagam/UFaceNet`.

## Setup

To start a new run:

1. Choose a run tag based on the date and hypothesis, for example `apr20-recon-token`.
2. Create `runs/<tag>/`.
3. Read the in-scope files:
   - `README.md`
   - `SOUL.md`
   - `CLAUDE.md`
   - `research/FACEXFORMER_V3_NOTES.md`
   - `research/TASK_MATRIX.md`
   - `research/UFACENET_ARCHITECTURE.md`
   - `research/BENCHMARKS.md`
   - `research/EXPERIMENT_QUEUE.md`
   - `research/EVIDENCE_LEDGER.md`
   - `research/BENCHMARK_LEDGER.md`
   - `research/REMOVAL_LOG.md`
4. Verify assets:
   - At least one test image.
   - Dataset roots for any benchmark being run.
   - CUDA availability if training or full inference is planned.
5. Verify that all active runtime imports come from `ufacenet/`.
6. Initialize or append to `research/results.tsv`.
7. Record the run hypothesis in `runs/<tag>/hypothesis.md`.

If the root is not a git repository, use `run_id` and file checksums instead of commit hashes.

## Research Goal

Optimize for a publishable UFaceNet result:

- Preserve the ten-task unified face analysis promise.
- Add a new FRec output family for face reconstruction/generation.
- Make FRec callable in the same one-pass task-token model path as the analysis tasks.
- Show that the new block is not cosmetic by reporting reconstruction metrics, face-specific generative metrics, original task regressions, and efficiency.

The primary target is not a single best metric. The target is a defensible Pareto point:

- Lower rFID and FID-face.
- Higher identity preservation.
- Good geometry consistency.
- Minimal original-task degradation.
- Acceptable FPS and parameter overhead.

## What You Can Modify

- Add UFaceNet runtime code in `ufacenet/`.
- Implement architecture, training, and evaluation independently from scratch.
- Add task registries, configs, evaluators, dataloaders, and scripts.
- Add reconstruction/generation heads.
- Add high-fidelity FRec modules: RGB decoder, geometry/depth/normal outputs, identity-preserving losses, perceptual losses, and optional VAE/VQ/diffusion-compatible refiner interface.
- Add metric implementations.
- Add figure and table generation scripts.
- Add documentation when it keeps the research loop reproducible.
- Add or update evidence, benchmark, and removal ledgers.

## What You Must Not Do

- Do not vendor, import, adapt, or load FaceXFormer code or checkpoints.
- Do not recreate a local `facexformer/` runtime folder.
- Do not compare against baselines without citing the source and documenting the protocol.
- Do not implement FRec as a separate unrelated generator that bypasses UFaceNet tokens/features.
- Do not change benchmark splits after seeing results.
- Do not drop original task metrics.
- Do not call face reconstruction `FR`; reserve `FR` for face recognition.
- Do not report rFID or FID-face without stating crop, resolution, feature extractor, sample count, and split.
- Do not rely on undocumented manual image selection.
- Do not leave unused imports, unused code, commented-out experiments, decorative section comments, or comments that only repeat what the next line does.

## Code Quality Standard

- Research code must be reproducible, readable, and inspectable.
- Public modules, classes, and scripts need concise purpose docstrings.
- Comments are allowed when they explain shape contracts, benchmark protocol, numerical stability, or non-obvious design decisions.
- Prefer explicit names and small functions over explanatory comments.
- Every script should have `--help`, clear defaults, and useful errors for missing assets.
- Before marking work complete, run a dead-code/import check where practical and remove obvious unused material.

## Experiment Loop

Loop until interrupted or until the completion criteria in `CLAUDE.md` are met:

1. Read the latest `research/results.tsv`.
2. Select the next experiment from `research/EXPERIMENT_QUEUE.md` or create a new one if evidence points elsewhere.
3. Define the hypothesis in one sentence.
4. Make the smallest code or config change that tests the hypothesis.
5. Run smoke validation.
6. Run the relevant benchmark subset.
7. Save logs and outputs under `runs/<tag>/`.
8. Append one row to `research/results.tsv`.
9. Update `research/EVIDENCE_LEDGER.md` for any proof, claim, or source-backed statement.
10. Update `research/BENCHMARK_LEDGER.md` for any metric, table value, figure value, or benchmark comparison.
11. Update `research/REMOVAL_LOG.md` for any removed, deprecated, renamed, superseded, or intentionally excluded item.
12. Keep, discard, or quarantine the change based on evidence.
13. Update the ACCV table/figure plan if the result changes what the paper can claim.

## Evidence And Removal Ledgers

No paper claim, benchmark table value, figure value, or code removal is allowed to exist only in memory or chat.

- `research/EVIDENCE_LEDGER.md`: claims, proof sketches, source-backed facts, assumptions, and verification status.
- `research/BENCHMARK_LEDGER.md`: metrics, table values, run ids, data splits, commands, configs, and status.
- `research/REMOVAL_LOG.md`: removed or superseded code paths, metrics, datasets, claims, table values, methods, and why they changed.

If a result is bad, keep it and mark it honestly. If a value changes, record both the old and new values with the reason.

## Standard Output Row

`research/results.tsv` is tab-separated. Do not use commas inside fields.

Columns:

```text
run_id	date	variant	status	changed_component	data_split	rfid	fid_face	arcface_fid	lpips	id_cos	now_median_mm	task_delta	fps_fp32	params_m	notes
```

Use `NA` when a metric was not applicable. Use `blocked:<reason>` when a metric should have run but could not.

## Keep Or Discard Criteria

Keep a change when:

- rFID or FID-face improves and identity/geometry does not regress materially.
- Original analysis tasks stay within the declared regression budget.
- Runtime and parameter overhead remain defensible for a unified model.
- The change simplifies the system while preserving metrics.

Discard or quarantine a change when:

- It improves image realism but damages identity or task consistency.
- It requires undocumented dataset filtering.
- It makes original tasks fail without a clear recovery plan.
- It cannot be reproduced from saved configs and logs.

## Regression Budgets

Initial budgets for UFaceNet prototypes:

- FP mean F1: no more than 0.5 absolute drop.
- LD NME: no more than 0.15 increase.
- HPE MAE: no more than 0.15 degree increase.
- Attr/Gen/Race/Exp accuracy: no more than 0.5 absolute drop.
- FR verification mean: no more than 0.25 absolute drop.
- FPS FP32: no more than 25 percent slower than the matching baseline unless reconstruction quality is the point of the experiment.

Tighten these budgets after the first stable baseline.

## Never Stop Rule

Do not stop after:

- writing docs,
- adding a new token,
- getting the model to instantiate,
- producing one reconstructed image,
- running only rFID,
- running only original task metrics,
- getting one attractive qualitative grid.

Continue until the complete benchmark packet exists or each missing piece has a precise blocker and next command.
