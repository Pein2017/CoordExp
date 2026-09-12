## 1. Freeze the contract and review

- [x] 1.1 Validate proposal, design and delta specs with `openspec validate optimize-production-training-and-eight-gpu-smoke --strict`; freeze their hashes and record the resolved grilling decisions.
- [x] 1.2 Run one independent Astra review round with separate semantic/resource and lifecycle/compatibility reviewers; record findings and lead dispositions, resolve material counterexamples, and revalidate before implementation.

## 2. Establish the real baseline

- [x] 2.1 Freeze the eight-rank four-update config and COCO prefixes in a run-specific launch packet; verify production EBS48, FA2/BF16, exact-state publication mode with null parent checkpoint, actual train/eval pack counts, at least eight eval packs and valid save/eval cadence through the real preparation entry.
- [x] 2.2 Execute the pre-change source eight-rank baseline with preserved logs and exit status; verify successful updates, evaluation and published checkpoints, recording any first failing evidence before a scoped correction.
- [x] 2.3 Preserve immutable pre-change adapter/embedding payload bytes and baseline HF outputs for a frozen inference set; verify hashes and identify the fixture as a baseline-produced payload if no suitable historical checkpoint exists.

## 3. Implement bounded preparation

- [x] 3.1 Add a caller-facing gated-executor regression that exceeds the worker-proportional submission limit on unchanged source; retain RED evidence plus canonical-order and worker-failure cases.
- [x] 3.2 Replace eager submission with a bounded local window in the existing cache owner; verify GREEN and existing order, worker, failure and payload-identity tests, keeping semantic cache determinants unchanged.
- [x] 3.3 Compare baseline/candidate preparation on the frozen workload through the real entry with unchanged 16 materialization workers and planner worker_count=1; verify actual worker metadata, identical encoded/packed identities and warm cache admission, recording timing/resource observations without inferring whole-pipeline streaming or unproven speedup.

## 4. Integrate the production smoke and compatibility checks

- [x] 4.1 Add the maintained eight-GPU smoke overlay without changing production precision, optimizer, EBS or one-through-four-process interface; verify config resolution and actual launch batch arithmetic.
- [x] 4.2 Verify supported old component routes and new authenticated payloads using existing inference reload/artifact tests; add a targeted regression only for a demonstrated missing compatibility invariant and retain fail-closed identity checks.
- [x] 4.3 Run the relevant cache, loss/masking, checkpoint and exact-resume regression suites; preserve command results and fix any reproduced task-related failure at its current owner.

## 5. Execute the eight-GPU vertical acceptance

- [x] 5.1 Run the candidate through cache preparation, all-hit verification and four applied eight-rank optimizer updates; verify finite loss, FA2 evidence, sharded forward evaluation, step-2/step-4 checkpoints and successful final publication, preserving exact configs/logs/resource observations.
- [x] 5.2 Resume the candidate step-2 checkpoint in a fresh eight-process run with the unchanged four-update schedule; compare the final model/optimizer/scheduler/RNG/cursor state and steps 3–4 accounting against uninterrupted candidate execution under the existing exact-resume contract.
- [x] 5.3 Consume immutable pre-change combined components and the candidate final checkpoint with real HF inference; verify baseline/candidate old-payload outputs and scoring, token trace and artifact finalization on the frozen request set.
- [ ] 5.4 Record and independently review the user-accepted one-grid BF16 composition contract; implement its bounded, authenticated admission with RED/GREEN while preserving exact structural/replay checks and honest numeric diagnostics. Run and admit the current composed-vLLM qualification for the final candidate model/config identity, then execute real vLLM inference; preserve failed attempts and require the unchanged runtime/concurrency/forced-replay criteria.
- [ ] 5.5 Run the direct detection evaluator on the completed scored inference artifact family; verify metrics/conversion artifacts, lineage and downstream completion without interpreting smoke metrics as model quality. HF baseline-old/candidate-old/initial-new/final-new are complete; the qualified vLLM family remains pending with 5.4.

## 6. Close the production change

- [x] 6.1 Record baseline/candidate timings, resource bounds, semantic comparison and each roadmap trigger disposition; require repeated comparable evidence for any speedup claim and explicitly mark inconclusive performance results.
- [x] 6.2 Update current training/operations documentation with the eight-rank command, few-step evidence boundary, old-inference/new-resume compatibility and links to the run receipt; verify referenced commands and local links.
- [x] 6.3 Inspect the exact final diff and retained acceptance artifacts, rerun affected verification after corrections, and run strict OpenSpec validation; mark only completed gates checked and report any remaining limitation without claiming full acceptance.
