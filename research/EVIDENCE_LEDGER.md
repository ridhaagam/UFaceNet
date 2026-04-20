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
limitations: local facexformer code exposes fewer inference task groups than the v3 paper describes
owner: Codex
notes: FR means face recognition. UFaceNet uses FRec for face reconstruction/generation.
```

### EVID-0002

```text
id: EVID-0002
date: 2026-04-20
status: planned
claim: UFaceNet adds a first-class FRec output family to the FaceXFormer-style tokenized model.
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
claim: The active UFaceNet package can run a one-pass request for all analysis tasks plus FRec without importing the upstream facexformer package.
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
claim: License-safe startup assets are available locally: LFW smoke data and the upstream FaceXFormer checkpoint.
evidence_type: local download
source_or_run: data/raw/lfw_home, data/processed/aligned_faces/train, checkpoints/facexformer/ckpts/model.pt
files: scripts/download_datasets.py, scripts/prepare_lfw_frec.py, runs/dataset_report_after_download.json
verification_command: python scripts/download_datasets.py --download-lfw --download-facexformer-ckpt; python scripts/prepare_lfw_frec.py --max-images 512; python scripts/validate_datasets.py --output runs/dataset_report_after_download.json
scope: training startup readiness
limitations: restricted datasets still require manual license/access steps
owner: Codex
notes: LFW is a smoke/verification set, not the full FaceXFormer multi-task training corpus.
```
