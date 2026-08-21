# Task 5 failure-mode matrix — bounded CPU correction

| boundary | failure | durable evidence | admitted counters | stop rule |
| --- | --- | --- | --- | --- |
| CUDA identity | indexless `cuda` resolves only under the frozen world-one/current-index/visibility contract | `Human13CudaLogicalDeviceIdentity` hash with raw accelerator, parameter, current-index, process, and visibility fields | unchanged | reject before K16 |
| CUDA identity | explicit index conflict, multi-device trainables, unavailable current index, or ambiguous visibility | typed `Human13LiveModelError`; no ownership receipt | unchanged | reject before K16 |
| training open / loader | checkpoint or placement fails before handle publication | append-only `ActionAttemptReceipt` and failed phase with exception hash and bounded redacted output | `model_loads=0`, `gpu_allocations=0`; no close claim | preserve primary failure |
| phase publication | journal write fails after a backend handle is returned | primary phase error plus best-effort close note; no caller-visible handle is published | admitted counter remains zero until durable handle publication | close the unpublished handle exactly once |
| audit open | audit boundary raises before handle publication | failed action-attempt phase; no admitted audit handle | no increment | close only an existing handle |
| audit evaluator | evaluator/loader raises during source/proposal audit | failed evaluator attempt with exception/provenance; existing session remains separately closable | admitted open counter unchanged | rollback/close according to lifecycle |
| terminal binding | service attempt ledger differs from terminal action list | terminal publication rejects before exclusive write | unchanged | retain phase evidence and primary error |
| historical terminal | v1 terminal lacks attempt list | explicit v1 schema dispatch and original content hash | never inferred | immutable reload only |

The matrix is a production-admission correction record, not algorithm evidence.
No GPU/model/K16/retry/update/checkpoint action is authorized by these rows.
