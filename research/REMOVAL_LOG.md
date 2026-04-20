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
status: deprecated
item: Direct runtime imports from facexformer/
item_type: code path
previous_location: facexformer/
new_location: ufacenet/
reason: user requested FaceXFormer code copied into the main project so active work does not depend on the upstream folder
evidence_or_benchmark_id: EVID-0003
replacement: migrated UFaceNet package under ufacenet/
impact: facexformer/ remains read-only provenance and baseline reference only
owner: Codex
notes: Baseline or migration scripts may inspect facexformer/ but new runtime code must import ufacenet.
```
