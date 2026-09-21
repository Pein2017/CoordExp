# Root exception: exactly two additional position-correct parity forwards

Decision: APPROVED. This is an explicit narrow amendment to the preceding no-second-repair boundary and the parent's8-forward gate cap, not a retrospective claim that the failed readbacks passed.

Root inspected technical-gate-v2/run_gate.py: compact_forward returns[batch,6,vocab], while stage compares source[target_index,-1] to target[0,0] for row_boundary. The latter is five sequence positions earlier. Forced-description x1 compares both final positions. The observed row-boundary discrepancy is therefore an invalid comparison, not demonstrated native-runtime divergence. Existing v1/v2 receipts and failed status remain immutable; qualify them in a new record as measurement-invalid.

Authorize EXACTLY2 additional native tied-model forwards, for aggregate10 across this gate lineage:
1. Original image417044, original tied model and literal native prefix through source_row.end119, target-only batch.
2. Same target prefix in its original heterogeneous source group with unchanged companion histories/positions.

No generation, new scientific cells, policy changes, model changes or new tolerance. Use logits_to_keep=1 and assert output shape[batch,1,full_vocab]. Save BOTH complete FP32 score vectors and actual input/prompt/media/prefix identities, masks and positions BEFORE any pass/fail comparison. Record source batch target index and causal alignment explicitly:119 consumed action tokens; next predicted action index119; the selected score is from the final consumed input position. Do not confuse that next-action index with an input-position index. Compare source[target_index,0] and target[0,0] only after proving identical unpadded target token/position alignment.

Before these calls, parent owns/checks the position-coded CPU RED/GREEN witness on the actual readback consumer: the archived6-position wrong selector must fail, and the corrected single-position selector plus saved-trace offset must pass. Do not delegate another unreviewed local-index guess. No new reviewer chain is needed.

Retain full-vocabulary max-absolute tolerance2e-4, exact winner, selected source saved-trace agreement and exact target positions/identities. Missing position evidence is not a passing value. Once vectors are on disk, perform the comparison again in the independent CPU consumer; preserve them even if it fails. Reuse the bound passed forced-x1 evidence and untied C qualification within their existing scope; do not rerun them or claim they qualify every spatial state.

If this correctly aligned check passes and the previous pixel/history/stopping/window repairs remain qualified, the earlier authorization to execute the remaining43 frozen states is active without another root question. Preserve all14 corrected pilot cells and their two LOCAL HOLDs. Remaining states retain their own original admission rules; total corrected panel remains45states/up to315unique cells. No substitutes or transform tuning.

If the two-forward gate still fails (including technical inability to establish alignment), stop B and return its unanswered/unexecuted portion with the unaffected lanes. No further GPU qualification attempt is authorized by this amendment.

The overall24allocated-GPU-hour ceiling and prior corrected-attempt budget remain unchanged. Include all prior failed gate calls, these2 calls, loading/occupancy and subsequent work in the conservative budget. No wake-me-up or new arm. Parent independently integrates A/C/D and returns one candidate with exact invalid/qualified/unexecuted distinctions. Root acceptance remains pending.
