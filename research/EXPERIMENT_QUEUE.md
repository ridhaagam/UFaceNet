# Experiment Queue

Use this queue with `program.md`. Append results to `research/results.tsv`.

## E000: Baseline Inventory

Hypothesis: The local repository can run FaceXFormer inference for released task groups once weights and one image are available.

Actions:

- Validate Python environment.
- Check checkpoint path.
- Run `facexformer/inference.py` on one image for each locally supported task.
- Save outputs under `runs/<run_id>/baseline_inference/`.

Keep criteria:

- At least one task inference works.
- Missing weights or image paths are documented precisely.

## E000A: FaceXFormer Source Migration

Hypothesis: UFaceNet should own its active runtime code while preserving FaceXFormer provenance and checkpoint compatibility.

Actions:

- Create `ufacenet/`.
- Copy and adapt required FaceXFormer modules into `ufacenet/`.
- Add provenance comments/docstrings only where useful.
- Add a compatibility loader for old FaceXFormer checkpoints.
- Update scripts so new runtime imports come from `ufacenet`, not `facexformer`.
- Add a smoke test that imports `ufacenet` without importing `facexformer`.

Keep criteria:

- Existing baseline path is still documented.
- UFaceNet forward pass can be tested from `ufacenet`.
- No new training/evaluation script depends on `facexformer` imports.
- Obvious unused imports, dead code, and stale comments are removed.

## E000B: One-Pass FRec Output Contract

Hypothesis: UFaceNet can expose analysis and FRec outputs through one FaceXFormer-style task-token interface.

Actions:

- Define the output dictionary contract for simultaneous analysis and FRec outputs.
- Add task-set parsing that can request all tasks in one call.
- Add a random-input smoke test that returns analysis placeholders and FRec image-shaped tensors in one forward pass.

Keep criteria:

- One model call can request `all` tasks including FRec.
- FRec does not bypass UFaceNet shared tokens/features.
- Output keys are stable and documented.

## E001: Task Registry

Hypothesis: A task registry reduces implementation risk before adding FRec.

Actions:

- Define task ids, names, output serializers, and metric names in one place.
- Preserve existing inference behavior.
- Add a smoke test that all existing task ids resolve.

Keep criteria:

- No behavior change for existing inference tasks.
- FRec can be registered without touching multiple hard-coded lists.

## E002: FRec Token Instantiation

Hypothesis: The model can add an FRec token while preserving old checkpoint loading via non-strict or migration-aware loading.

Actions:

- Add `T_frec`.
- Add config flag `enable_frec`.
- Ensure old checkpoint still loads for old tasks.
- Verify forward pass shape with random input.

Metrics:

- model instantiation
- params
- forward latency

Keep criteria:

- Existing task outputs unchanged when `enable_frec=false`.
- Forward pass works when `enable_frec=true`.

## E003: Frozen RGB Reconstruction Decoder

Hypothesis: A frozen migrated UFaceNet encoder/FaceX decoder plus a small RGB decoder can reconstruct aligned face crops enough to establish a baseline.

Actions:

- Freeze migrated backbone and FaceX modules inside `ufacenet/`.
- Train FRec decoder on aligned face crops.
- Save paired reconstruction grids.

Metrics:

- rFID
- FID-face
- LPIPS
- PSNR
- SSIM
- ID cosine
- FPS

Keep criteria:

- Produces valid nonblank reconstructions.
- rFID and LPIPS are finite.

## E004: Identity Loss

Hypothesis: ArcFace identity loss improves identity preservation without materially hurting rFID.

Actions:

- Add frozen identity encoder.
- Add `1 - cosine` identity loss.
- Compare against E003.

Metrics:

- ID cosine
- ArcFID
- rFID
- LPIPS
- task consistency

Keep criteria:

- ID cosine improves.
- rFID does not degrade beyond declared tolerance.

## E005: Landmark And Pose Consistency

Hypothesis: Landmark and pose consistency improve geometric faithfulness.

Actions:

- Use frozen or detached UFaceNet predictions, with FaceXFormer baseline predictions only for comparison when available.
- Add consistency losses for LD and HPE.

Metrics:

- landmark consistency NME
- pose consistency MAE
- LPIPS
- ID cosine
- rFID

Keep criteria:

- landmark or pose consistency improves without damaging ID cosine.

## E006: Parsing Consistency

Hypothesis: Face parsing consistency improves semantic region reconstruction and reduces artifacts near eyes, lips, nose, and hair.

Actions:

- Add parsing consistency on masks or logits.
- Evaluate per-region reconstruction.

Metrics:

- parsing consistency F1
- rFID
- LPIPS
- qualitative region crops

Keep criteria:

- better semantic consistency with acceptable rFID/LPIPS.

## E007: Geometry Head

Hypothesis: Predicting geometry parameters/depth/normal makes reconstruction more controllable and improves 3D metrics.

Actions:

- Add geometry output path.
- Start with depth/normal if FLAME assets are not available.
- Add NoW/MICC/AFLW2000-3D evaluator when data is available.

Metrics:

- NoW median/mean/std mm
- MICC RMSE
- AFLW2000-3D NME
- render consistency

Keep criteria:

- geometry metric improves or the branch exposes a clear improvement path.

## E008: Reconstruction-Consistency Block

Hypothesis: Letting reconstruction interact with task tokens through a dedicated block improves the Pareto point versus an isolated decoder.

Actions:

- Add block between FaceX outputs and reconstruction decoder.
- Compare against isolated FRec decoder.

Metrics:

- rFID
- FID-face
- ID cosine
- geometry metric
- original task delta
- FPS

Keep criteria:

- better reconstruction/identity/geometry with task regression inside budget.

## E009: Multi-Token FRec

Hypothesis: Geometry, texture, and render tokens reduce task conflict compared with one FRec token.

Actions:

- Add `T_geom`, `T_tex`, and `T_render`.
- Compare with one-token FRec.

Metrics:

- same as E008
- parameter overhead
- latency overhead

Keep criteria:

- measurable improvement justifies complexity.

## E010: Generative Refiner

Hypothesis: A generative refiner improves face realism and FID-face while preserving identity through conditioning.

Actions:

- Add optional VAE/VQ/diffusion refiner.
- Keep lightweight branch as separate baseline.
- Wrap the refiner behind the same one-pass FRec interface.
- Ensure the refiner is conditioned by UFaceNet tokens/features, not a standalone face generator.

Metrics:

- FID-face
- rFID
- ArcFID
- ID cosine
- detector failure rate
- FPS

Keep criteria:

- improved realism without unacceptable identity or speed cost.

## E011: Original Task Regression Packet

Hypothesis: UFaceNet can add reconstruction/generation without substantial degradation to the original tasks.

Actions:

- Run original task benchmarks or smallest valid subsets.
- Compare FaceXFormer baseline and UFaceNet variants.

Metrics:

- FP F1
- LD NME
- HPE MAE
- Attr accuracy
- Age MAE
- Gen/Race/Exp accuracy
- Vis recall at 80 percent precision
- FR verification accuracy

Keep criteria:

- all metrics within regression budget or tradeoff is explicitly justified.

## E012: ACCV Figure And Table Generation

Hypothesis: Results are reproducible enough to generate paper assets from scripts.

Actions:

- Generate task matrix table.
- Generate architecture figure draft.
- Generate qualitative reconstruction grid.
- Generate ablation plot.
- Generate runtime table.

Keep criteria:

- every asset can be regenerated from commands and saved configs.
