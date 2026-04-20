# Evidence Ledger

Canonical repo: https://github.com/ridhaagam/UFaceNet

Purpose: every proof, claim, source-backed statement, and paper assertion must be traceable. Do not use a claim in the README, paper, figures, or GitHub release unless it has an entry here or in a linked run artifact.

## Status Labels

- `planned`: claim is intended but not verified.
- `verified`: claim is supported by source, run artifact, or reproduced result.
- `partial`: claim is partly supported and needs more evidence.
- `rejected`: claim is false or not defensible.
- `superseded`: claim was replaced by a newer, documented claim.

## Entry Template

```text
id:
date:
status:
claim:
evidence_type:
source_or_run:
files:
verification_command:
scope:
limitations:
owner:
notes:
```

## Initial Entries

### EVID-0001

```text
id: EVID-0001
date: 2026-04-20
status: verified
claim: FaceXFormer v3 describes ten facial analysis tasks: FP, LD, HPE, Attr, Age, Gen, Race, Vis, Exp, and FR.
evidence_type: external paper
source_or_run: https://arxiv.org/pdf/2403.12960v3
files: research/FACEXFORMER_V3_NOTES.md, research/TASK_MATRIX.md
verification_command: manual PDF review
scope: paper framing and task matrix
limitations: this is a paper-source claim only and does not imply UFaceNet uses FaceXFormer code
owner: Codex
notes: FR means face recognition. UFaceNet uses FRec for face reconstruction/generation.
```

### EVID-0002

```text
id: EVID-0002
date: 2026-04-20
status: planned
claim: UFaceNet adds a first-class FRec output family to an independent tokenized unified face model.
evidence_type: architecture plan
source_or_run: research/UFACENET_ARCHITECTURE.md
files: README.md, CLAUDE.md, program.md
verification_command: pending implementation smoke test
scope: project objective
limitations: not yet implemented in code
owner: future implementation agent
notes: Must be implemented in ufacenet/ and callable in one forward path with analysis tasks.
```

### EVID-0003

```text
id: EVID-0003
date: 2026-04-20
status: verified
claim: The target public repository for this project is https://github.com/ridhaagam/UFaceNet.
evidence_type: user instruction
source_or_run: user request in current workspace
files: README.md, CLAUDE.md, program.md, research/CLAUDE_FULL_REPO_PROMPT.md
verification_command: local documentation review
scope: repository governance
limitations: current local workspace is not a git repository
owner: Codex
notes: Do not claim that docs were pushed unless a git push succeeds.
```

### EVID-0004

```text
id: EVID-0004
date: 2026-04-20
status: verified
claim: The active UFaceNet package can run a one-pass request for all analysis tasks plus FRec without importing external FaceXFormer code.
evidence_type: local smoke tests
source_or_run: pytest and runs/inference_smoke/report.json
files: ufacenet/, scripts/run_inference.py, ufacenet/tests/test_model.py
verification_command: python -m pytest -q; python scripts/run_inference.py --tasks all --image-size 64 --output-dir runs/inference_smoke --refiner
scope: starter implementation readiness
limitations: smoke uses randomly initialized tiny backbone, not trained paper-quality weights
owner: Codex
notes: This verifies interface and output shapes, not benchmark accuracy.
```

### EVID-0005

```text
id: EVID-0005
date: 2026-04-20
status: verified
claim: License-safe startup assets are available locally for FRec smoke training with LFW.
evidence_type: local download
source_or_run: data/raw/lfw_home, data/processed/aligned_faces/train
files: scripts/download_datasets.py, scripts/prepare_lfw_frec.py, runs/dataset_report_after_download.json
verification_command: python scripts/download_datasets.py --download-lfw; python scripts/prepare_lfw_frec.py --max-images 512; python scripts/validate_datasets.py --output runs/dataset_report_after_download.json
scope: training startup readiness
limitations: restricted datasets still require manual license/access steps
owner: Codex
notes: LFW is a smoke/verification set, not the full UFaceNet multi-task training corpus.
```

### EVID-0006

```text
id: EVID-0006
date: 2026-04-20
status: verified
claim: UFaceNet no longer tracks or uses a local FaceXFormer runtime folder or checkpoint download path.
evidence_type: local git and smoke validation
source_or_run: git status, runs/inference_independent_smoke/report.json
files: README.md, CLAUDE.md, scripts/download_datasets.py, ufacenet/data/download.py, research/REMOVAL_LOG.md
verification_command: find . -maxdepth 2 -type d -name facexformer; git ls-files facexformer; python -m pytest -q; python scripts/run_inference.py --tasks all --image-size 64 --output-dir runs/inference_independent_smoke --refiner
scope: independence policy
limitations: FaceXFormer remains cited in literature/benchmark docs only
owner: Codex
notes: Local ignored FaceXFormer checkpoint files were removed from the workspace.
```

### EVID-0007

```text
id: EVID-0007
date: 2026-04-20
status: verified
claim: FRec smoke images are no longer blank after adding the input-guided reconstruction skip and geometry bootstrap.
evidence_type: local image-output validation
source_or_run: runs/inference_fixed_image_v2 and runs/frec_train_fixed_v2_2step
files: ufacenet/reconstruction.py, scripts/check_image_outputs.py, scripts/run_inference.py, scripts/train_frec.py
verification_command: python scripts/check_image_outputs.py runs/inference_fixed_image_v2 runs/frec_train_fixed_v2_2step --min-std 0.03
scope: visual smoke reliability
limitations: validates nonblank output and reconstruction plumbing, not paper-quality fidelity
owner: Codex
notes: The input skip is trainable and initialized to preserve visible image structure during from-scratch startup.
```
