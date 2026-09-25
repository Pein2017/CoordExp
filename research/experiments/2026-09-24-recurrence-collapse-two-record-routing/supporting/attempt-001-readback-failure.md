# Two-record attempt 001: cold-readback qualification failure

**Status: technical HOLD / scientific outcome unanswered.** No repair, second model attempt, or successor was run. Protocol SHA-256 `cf4f84f9f4142fdbfad5a07a9db3ae0de7d007ccdac846fca283f7d4c94ded40`; admission SHA-256 `0c44d25fd542043ef543eb31fa284549edd07dd5d0fd05dd796a3093d7b53668`.

CPU preflight passed 225 actual writer/caller/parser/mutation checks, original four-request source/position/companion identity, widths 1380–1395, and 27 direct source captures. [Preflight](attempt-001-preflight.json) SHA-256 `77f19245d84d880bc07428d985bd48ef4a50727c4a21f5f98a9ce4462a56fd54`; captured [producer](/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/recurrence_collapse_two_record_routing/run.py) SHA-256 `0e701dea6634f2dcc39f11b64d99f990b4af1b406cd7c258bee9396d6cd9f943`.

The one GPU child completed all five fixed arms and its production receipt reports 45 model, 45 vision, 45 emitted target tokens, zero reused. Native/sham row2 and source trace, 28 native-mask consumers, and companion comparisons were checked during that run. The saved raw arm token sequences are:

| Arm | Emitted IDs (provisional; cold readback incomplete) |
| --- | --- |
| native_AF | `151646,8987,151647,151648,151670,151670,151699,151756,151649` |
| identity_write_AF | same as native_AF |
| FF | `151646,8987,151647,151648,151671,151683,152206,152669,151649` |
| AA | `151646,8987,151647,151648,151875,151775,152077,152169,151649` |
| FA | `151646,8987,151647,151648,151670,151670,151717,151756,151649` |

The independent CPU readback stopped on its first arm **before loading any saved vector**. The receipt encodes `written_records` as JSON list `['A','F']`; the frozen producer's `PATTERNS['native_AF']` is Python tuple `('A','F')`. Its direct comparison at `run.py:517` therefore raises `ValueError: cold written donors changed`. This is a representation mismatch in the verifier, not observed evidence of a changed donor or source. It prevents the required separate-process vector, input, mask, parser and region qualification. Neither `readback.json` nor `overlay.png` exists. The captured producer, preflight, raw files and receipt were left unchanged; the readback command was not rerun.

[Raw production receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-collapse-two-record-routing/attempt-001/receipt.json) SHA-256 `0c1132a8efe2ebde2ddc3ae180d84d13de66f0c4f1cf5329ec4cbcc4b484a974` binds all 45 saved vectors and actual inputs. [Outer terminal record](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-collapse-two-record-routing/attempt-001/outer.json) SHA-256 `6eee4194dc16ddebc671029651759a8409069a30b7c3bef74b79d1ed73d038ce`: child PID 2094235 absent, exit 0, **129.19039377570152 outer seconds**; 123.0920315310359 internal seconds. Final raw root 700,761,281 bytes, peak RSS 11,518,120 KiB, GPU allocated/reserved 9,952,862,720/11,075,059,712 bytes. Prior sequence `0.5012880232967056` plus this charge gives `0.537174243789956` GPU-hours. [Stdout](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-collapse-two-record-routing/attempt-001/stdout.log) SHA-256 `102aab0d7c48c90417a8efa9b2a61b2175727b93233fd72d30be26d05b70a3bd`.

The narrow possible next action, if the lead authorizes it, is CPU-only readback of the existing saved vectors with tuple/list normalization at this check, followed by the original untouched gates. No GPU re-execution is needed to inspect the saved evidence. The present packet does **not** score the frozen FF/AA/FA conjunction or claim a scientific result.
