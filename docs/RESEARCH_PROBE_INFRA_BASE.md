# Research mechanics

Use [operator entries](../probes/README.md) and [core architecture](SYSTEM_OVERVIEW.md)
from a current task, not a full historical producer chain.

| Operation | Owner | Caller still decides |
|---|---|---|
| Current config resolution | `src.config` | Explicit science/policy values |
| Original token/text mapping | `src.inference.token_text` | Parser evidence and target eligibility |
| Native image/history planning | `src.inference.inputs`, `src.qwen.native` | Literal histories, companions, masks and budgets |
| Bound image relocation | `src.inference.input_materialization` | The frozen input and new JSONL origin |
| Aligned token scores | `src.losses.token_scores` | Selection, reduction, dose and credit |
| Adapter/model assembly | `src.adapters`, `src.qwen` | Composition, mode, precision and checkpoint |
| Owner/geometry accounting | `src.eval.saved_rows`, `src.eval.assignment` | Threshold, label version and physical interpretation |
| Exclusive publication | `src.artifacts` | Payload meaning and a fresh destination |
| Continuation source gate | `src.artifacts.git_identity` | Exact required source closure and separate data/runtime identity |

There is no shared experiment DSL or mandatory historical admission runner.
Two functions with different loss denominators or conditioning are not duplicate
implementations to merge. Do not place fixed populations/checkpoints into generic
loaders. New abstraction needs a real retained consumer, not hypothetical reuse.

The two retained numerical methods are output-QP and readout norm. Their current
qualification commands load no model. Core CPU tests cover primitive semantics;
real-model forward/gradient/cache parity remains a separate claim requiring
explicitly scoped evidence. Historical detailed methods are in Git via catalog.
