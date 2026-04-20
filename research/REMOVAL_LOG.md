# Removal Log

Canonical repo: https://github.com/ridhaagam/UFaceNet

Purpose: document every removed, deprecated, renamed, superseded, or intentionally excluded code path, result, table value, metric, claim, dataset, or experiment. Nothing should disappear without a reason.

## Status Labels

- `removed`: item was deleted or excluded.
- `deprecated`: item remains but should not be used for new work.
- `renamed`: item changed name but remains conceptually active.
- `superseded`: item was replaced by a better protocol/result/implementation.
- `rejected`: item was considered and rejected.

## Entry Template

```text
id:
date:
status:
item:
item_type:
previous_location:
new_location:
reason:
evidence_or_benchmark_id:
replacement:
impact:
owner:
notes:
```

## Initial Entries

### REM-0001

```text
id: REM-0001
date: 2026-04-20
status: renamed
item: Face Reconstruction abbreviation
item_type: terminology
previous_location: user-facing discussion
new_location: FRec throughout UFaceNet docs
reason: FR is already used for Face Recognition in FaceXFormer v3
evidence_or_benchmark_id: EVID-0001
replacement: FRec
impact: avoids ambiguity in task matrix, code registry, and tables
owner: Codex
notes: Do not use FR for reconstruction in code, docs, or figures.
```

### REM-0002

```text
id: REM-0002
date: 2026-04-20
status: removed
item: FaceXFormer code and checkpoint dependency
item_type: code path
previous_location: facexformer/
new_location: ufacenet/ independent implementation
reason: user requested UFaceNet to be fully our own and trained from scratch rather than using FaceXFormer code or checkpoints
evidence_or_benchmark_id: EVID-0003
replacement: independent UFaceNet package under ufacenet/
impact: FaceXFormer remains a cited baseline paper only
owner: Codex
notes: Do not recreate facexformer/ or checkpoint-download paths.
```

### REM-0003

```text
id: REM-0003
date: 2026-04-20
status: removed
item: FaceXFormer checkpoint downloader
item_type: script feature
previous_location: scripts/download_datasets.py and ufacenet/data/download.py
new_location: NA
reason: UFaceNet will train from scratch and must not load FaceXFormer checkpoints
evidence_or_benchmark_id: EVID-0003
replacement: UFaceNet checkpoints produced by scripts/train_frec.py and future multi-task training scripts
impact: dataset downloader now prepares datasets only
owner: Codex
notes: Local ignored checkpoint files may exist from earlier setup but are not committed or used.
```
