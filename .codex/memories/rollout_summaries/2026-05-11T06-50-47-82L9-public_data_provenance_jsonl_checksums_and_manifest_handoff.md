thread_id: 019e15cd-7907-76b0-902a-83af0aaee1f3
updated_at: 2026-05-11T07:31:08+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/05/11/rollout-2026-05-11T06-50-47-019e15cd-7907-76b0-902a-83af0aaee1f3.jsonl
cwd: /data/CoordExp
git_branch: main

# Added Git-tracked public_data provenance manifests and JSONL-only checksum tests for the core COCO 1024 processed datasets

Rollout context: The user first asked to clean up `public_data` and keep only a small set of COCO/LVIS/VG data products; later they asked to align `public_data` behavior with a newer cross-node sync/provenance principle, then tightened the contract to use cheap checksum alignment on training JSONL files rather than disk-level sync. The key durable topic became: `public_data` processed datasets should be reproducible from raw data plus Git-tracked provenance, and only the model-facing JSONL files should be checksum-verified across nodes.

## Task 1: Clean up `public_data` and preserve only the requested dataset groups

Outcome: success

Preference signals:
- The user said in Chinese: “我现在需要对codebase的`public_data`做一层清理、整理和归档，把暂时不需要的数据给完全删掉以节省磁盘。” -> the user wanted destructive cleanup, not just guidance or classification.
- After the assistant initially treated VG as a deletion candidate, the user corrected: “`VG`raw也可以保留” -> in similar cleanup tasks, VG raw should be treated as a keep candidate unless the user says otherwise.
- The user later clarified that `coco,1024,60` was actively used by two training processes and said: “你先不用动这一个数据集的源数据。” -> if the user says a data root is in active use, future agents should not modify or regenerate that root without explicit permission.

Key steps:
- Inventory of `public_data` showed large old derived trees and caches; Git tracking was checked so only untracked local data was removed.
- Deleted old 768 variants, VG-Ref, VG derived data, `public_data/output`, `__pycache__`, and nonessential caches; preserved raw COCO/LVIS/VG and COCO 1024 base/max60/max60-lvis-proxy.
- Verified post-cleanup that only the intended directories remained and that `git status -- public_data` was clean.

Failures and how to do differently:
- The first pass considered VG raw and some COCO 1024 related data too aggressively; the user corrected the scope. In similar cleanups, ask or infer active-use exceptions before deleting anything that might be training input.
- One requested dataset variant (`public_data/coco/rescale_32_1024_bbox_lvis_proxy`) did not exist on disk. It was initially called out as missing, then later correctly removed from the tracking set once the user clarified that nonexistent variants should not be tracked as current artifacts.

Reusable knowledge:
- `public_data` processed trees can be very large; `du -sh` plus `find`/`git ls-files` is enough to separate raw inputs from disposable derived data.
- This repo keeps raw datasets and processed derived trees under `public_data`, but the derived trees may be safely removed if they are not part of the current working set.
- Active training use is a stop signal: if the user says a processed dataset is currently feeding training, do not rewrite or delete it.

References:
- `public_data/coco/raw`, `public_data/lvis/raw`, `public_data/vg/raw` were preserved.
- Deleted examples: `public_data/coco/rescale_32_768_bbox`, `public_data/lvis/rescale_32_1024_bbox`, `public_data/vg_ref`, `public_data/output`, `public_data/**/__pycache__`.
- The missing requested variant was explicitly confirmed as absent: `public_data/coco/rescale_32_1024_bbox_lvis_proxy`.

## Task 2: Add provenance manifests and tests for core `public_data` datasets

Outcome: success

Preference signals:
- The user asked: “请浏览记忆。” and described a desire for a separate `public_data*`-style record of training-data changes and meta information so that cross-cluster exports from raw data are identical. -> future agents should treat Git-tracked provenance as the contract, not disk-level sync.
- The user explicitly said not to touch `public_data/coco/rescale_32_1024_bbox` because two training processes were using it. -> future provenance or validation changes should avoid rewriting active data roots.
- After seeing that the full-object LVIS-proxy variant was not actually produced, the user asked why it was tracked and then said: “如果本身就还没有生产这个数据集，则不用追踪/记录它了，将其删除即可” -> only materialized durable processed datasets should appear in the current provenance set; wishlist variants should not be recorded as current artifacts.

Key steps:
- Read the accepted design/standard docs describing `manifests/public_data_provenance/` as the Git-tracked location for processed public_data regeneration contracts.
- Added JSON manifests for the three actual COCO 1024 processed datasets that matter for the current environment:
  - `public_data/coco/rescale_32_1024_bbox`
  - `public_data/coco/rescale_32_1024_bbox_max60`
  - `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy`
- Created `tests/test_public_data_provenance_manifests.py` to enforce:
  - schema shape and required fields,
  - manifest path mirroring under `manifests/public_data_provenance/`,
  - repo-relative paths,
  - current materialized COCO1024 processed dirs having matching manifests,
  - regeneration-vs-sync policy.
- Added and then removed the nonexistent `rescale_32_1024_bbox_lvis_proxy` manifest after the user clarified it should not be tracked.

Failures and how to do differently:
- The first manifest draft included a not-yet-materialized full-object LVIS-proxy variant. The user corrected that. In similar work, distinguish clearly between “desired future dataset” and “currently durable processed dataset.”
- Initial tests treated the provenance layer as metadata-only and forbade checksums; the user then clarified that per-JSONL checksums were desired because they are cheap and useful. Future agents should not assume checksums are out of scope; ask whether to hash the model-facing JSONL files.

Reusable knowledge:
- The canonical provenance location is `manifests/public_data_provenance/<dataset>/<processed-dir>.json`, mirroring the `public_data` path.
- `docs/standards/OUTPUT_SYNC_AND_DATA_PROVENANCE.md` and `docs/superpowers/specs/2026-05-11-output-sync-and-public-data-provenance-design.md` encode the project policy: raw `public_data` is not Baidu-sync territory; processed `public_data` should be reproducible from manifests.
- The user wants these manifests to survive cross-node environments where only raw COCO/LVIS is guaranteed.

References:
- Added manifests:
  - `manifests/public_data_provenance/coco/rescale_32_1024_bbox.json`
  - `manifests/public_data_provenance/coco/rescale_32_1024_bbox_max60.json`
  - `manifests/public_data_provenance/coco/rescale_32_1024_bbox_max60_lvis_proxy.json`
- Added test file: `tests/test_public_data_provenance_manifests.py`
- Verified by targeted test run: `conda run -n ms python -m pytest tests/test_public_data_provenance_manifests.py -q` -> `6 passed`.

## Task 3: Add JSONL-only checksums for materialized processed datasets and keep nonexistent variants untracked

Outcome: success

Preference signals:
- The user asked: “file_hashes, checksums可以是对整个文件夹做吗？我想要对`*.jsonl`(训练样本）做hash或者 checksums 层面的对齐，因为比较‘便宜’” -> future agents should use file-level JSONL checksums, not whole-folder or image-tree checksums.
- The user later said: “没毛病，按照这样修改！” -> the checksum scope decision was accepted.
- When the assistant proposed keeping the nonexistent full-object LVIS-proxy variant tracked, the user corrected that it should be removed because it had not been produced. -> do not track unmaterialized dataset variants in the active provenance set.

Key steps:
- Computed SHA256, size_bytes, and nonempty line counts for the JSONL training sample files in the three materialized COCO 1024 roots.
- Extended the provenance schema to allow a `checksums` field with:
  - `scope = jsonl_training_samples_only`
  - `algorithm = sha256`
  - an `aggregate_sha256` over sorted per-file records
  - a `files` list containing repo-relative `*.jsonl` paths only.
- Updated the README example to explain that only model-facing JSONL files should be hashed, not images, cache directories, or whole trees.
- Added checksum payloads to the three materialized manifests and updated the test to recompute and compare per-file and aggregate hashes.
- Deleted the unmaterialized `rescale_32_1024_bbox_lvis_proxy` manifest after the user said nonexistent datasets should not be tracked.

Failures and how to do differently:
- The first implementation used a blanket “no checksums” posture; the user explicitly wanted cheap JSONL hashing. In future, when a user asks for cross-node equivalence, treat per-file hashes of the model-facing artifacts as a default option unless there is a reason not to.
- The attempt to keep a manifest for the nonexistent full-object LVIS-proxy variant was corrected away. Future agents should avoid recording future-wish datasets as current canonical training artifacts.

Reusable knowledge:
- For these `public_data` contracts, the cheap/effective checksum target is the JSONL sample files only; the user did not want raw images or entire directories hashed.
- The three current materialized roots that now carry checksum blocks are:
  - `public_data/coco/rescale_32_1024_bbox`
  - `public_data/coco/rescale_32_1024_bbox_max60`
  - `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy`
- The test recomputes per-file SHA256 and an aggregate SHA256 from sorted `path sha256 size_bytes records` lines, which is a good cross-node equality sentinel.

References:
- Schema: `manifests/public_data_provenance/schema.json`
- README: `manifests/public_data_provenance/README.md`
- Checksummed manifests:
  - `manifests/public_data_provenance/coco/rescale_32_1024_bbox.json`
  - `manifests/public_data_provenance/coco/rescale_32_1024_bbox_max60.json`
  - `manifests/public_data_provenance/coco/rescale_32_1024_bbox_max60_lvis_proxy.json`
- Test logic: `tests/test_public_data_provenance_manifests.py`
- Verified by repeated targeted test runs: `6 passed in ~2.1s`.

## Task 4: Commit and push the provenance/checksum change set

Outcome: success

Preference signals:
- The user explicitly requested: “请commit and push这些 changes。” -> future agents should not stop at local edits when the user asks for commit/push.
- The user also said they would “先`pull`再执行校验” in the other environment, so pushing the canonical provenance state was important for handoff.

Key steps:
- Confirmed current branch `main`, HTTPS remote `origin`, and that `github_personal_token.txt` is ignored/untracked.
- Staged only the intended provenance-related files, not unrelated docs modifications that existed earlier in the broader rollout context.
- Committed with a small, scoped message: `chore(public-data): add provenance checksums`.
- Pushed successfully to `origin/main`.

Failures and how to do differently:
- There were pre-existing docs changes elsewhere in the broader session context, but by the time of commit the worktree relevant to this task was cleanly narrowed to the provenance files. In similar situations, do a focused `git add` on only the intended files.

Reusable knowledge:
- The repo uses HTTPS remote `https://github.com/Pein2017/CoordExp.git`.
- `github_personal_token.txt` is ignored by `.gitignore` and was not tracked.
- Final commit pushed cleanly to `main`.

References:
- Commit: `82d5b26e62b57471c70282eca0db8e4875b74766`
- Commit message: `chore(public-data): add provenance checksums`
- Push result: `main -> origin/main`
- Final verification before commit: `conda run -n ms python -m pytest tests/test_public_data_provenance_manifests.py -q` -> `6 passed`.

## Handoff prompt created for another environment

Outcome: success

Preference signals:
- The user wanted a prompt for another Codex agent running in an environment that may only have COCO/LVIS raw data. That implies a durable handoff should emphasize pull-first, manifest-first verification, and regeneration from raw if needed.
- The user specifically said the other environment would “先`pull`再执行校验” -> the prompt should start with a `git pull --ff-only` and then verification.

Key steps:
- Drafted a standalone handoff prompt that instructs the other agent to:
  - pull the latest commit,
  - inspect the provenance manifests,
  - run the provenance checksum test,
  - regenerate only missing datasets from raw COCO/LVIS if needed,
  - avoid Baidu Netdisk or disk-level sync,
  - report mismatches rather than silently rewriting manifests.
- The prompt explicitly frames the three core datasets as the canonical set and tells the agent not to invent a manifest for the unmaterialized full-object LVIS-proxy variant.

Reusable knowledge:
- The ideal cross-environment validation workflow is: pull latest -> inspect manifests -> run targeted provenance test -> regenerate only missing processed datasets -> rerun test.
- The handoff should stress JSONL-only checksums and manifest-driven reproducibility, not whole-tree synchronization.

References:
- Handoff commit to pull: `82d5b26e62b57471c70282eca0db8e4875b74766`
- Canonical manifest paths to validate: `manifests/public_data_provenance/coco/rescale_32_1024_bbox.json`, `..._max60.json`, `..._max60_lvis_proxy.json`
- Validation command: `conda run -n ms python -m pytest tests/test_public_data_provenance_manifests.py -q`
