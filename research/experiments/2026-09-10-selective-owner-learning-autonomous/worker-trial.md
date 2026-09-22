# One paired DeepSeek-max versus Astra-max subagent trial

This is an operational side study requested by the user, not a model-quality
gate for selective owner learning. Both workers received the same bounded
read-only question and allowed files: find silent loss/mask/receipt mistakes
in the frozen first iteration and propose exactly three deterministic tests.
Neither owned code, launches, research decisions or further delegation.

| Observation | DeepSeek V4.1-Flash, max | Astra, max |
|---|---|---|
| Route | Persistent Codex CLI `ds-flash-max`; provider/model/effort verified in runtime metadata | Native `gpt-6-astra`, max, fresh context |
| Observed task interval | 73 seconds including CLI startup,16:34:47 to16:36:00UTC | Approximately211 seconds from assignment delivery to final candidate,16:35:37 to16:39:08UTC |
| Correct preservation counts134/41 | No: proposed138/44 and an incompatible index-set test | Yes: exact ranges, excluded five positions per image, EOS included |
| KL/weighting test correctness | No: reversed gradient sign/wrong differentiation variable; required formula replaced per-state mean by sum | Yes: analytic expected total2.00344455595 and per-state gradient5/134*(q-p) |
| Obsolete stopping helper risk | Correctly identified old schema/margin stop incompatibility | Correct, plus concrete early-stop and update22 consumer counterexamples |
| Full-trajectory CE indexing risk | Not identified | Correctly identified last-position helper would supervise EOS if its assertion were weakened |
| Lead disposition | Candidate not accepted for loss/mask/test decisions; useful already-known stop-rule warning retained | Candidate advice accepted after independent mask arithmetic and analytic loss/gradient checks |

Root independently projected the exact retained trajectories: image368 has
139 tokens, excludes97..101 and preserves134; image7116 has46, excludes22..26
and preserves41. EOS138/45 is included. A direct Torch autograd calculation
confirmed the Source-to-candidate KL logit gradient is candidate probability
minus Source probability, not the reverse. The Astra numeric loss fixture was
independently recomputed as2.0034445559487057. Newly implemented training and
evaluation tests are accepted separately; neither adviser self-report replaces
the actual model/consumer checks.

DeepSeek returned substantially sooner on this task but proposed tests that
would enforce the wrong scientific objective. Astra was more reliable on the
specific alignment and calculus questions. The observed intervals have different
CLI/native startup and tool overhead, and provider prompts/cache differ despite
the common brief; this is not a controlled throughput benchmark, general
intelligence ranking or a universal speed ratio. No numerical RMB charge was
available. DeepSeek's turn completed without quota/error events; its reported
usage was465532 input tokens (400128 cached),11166 output including7809 reasoning
tokens. Token accounting is provider-reported, not a bill.

The DeepSeek session remains available for a future suitable bounded task;
there is no live invocation and no automatic retry or refill. Root continues
to own all decisions. One successful CLI turn does not establish broad worker
reliability, and one poor scientific review does not rule out mechanical tasks.

Invocation receipt and preserved original response:

- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/deepseek-trial-invocation.json`
- `/tmp/deepseek-worker.heIGg4/reply.txt`
- DeepSeek session: `01a08c2b-fdba-7630-8aee-74e91390c103`.
