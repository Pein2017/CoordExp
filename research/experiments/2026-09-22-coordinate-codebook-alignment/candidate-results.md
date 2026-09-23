# Coordinate codebook alignment — final worker candidate

Status: candidate for final lead acceptance. Technical qualification and the
nominal fitted-image milestone are lead-accepted; the full package is not
self-accepted. Both fresh training seeds and all 352 fixed native cells are
complete. All 63 owned producers are terminal and their PIDs absent. No next
stage model work, conditional LR fit, architecture ablation, commit or
publication was performed.

## Scientific evidence

| Fitted-image condition | Clean images | Class-consistent IoU50 | Class-consistent IoU80 | Caps |
| --- | ---: | ---: | ---: | ---: |
| source | 4/32 | 264/679 | 158/679 | 3 |
| nominal-step32 | 5/32 | 341/679 | 220/679 | 3 |
| nominal-step128 | 21/32 | 584/679 | 526/679 | 0 |
| nominal-step512 | 32/32 | 679/679 | 679/679 | 0 |
| nominal-repeat-step32 | 5/32 | 367/679 | 243/679 | 2 |
| nominal-repeat-step128 | 22/32 | 582/679 | 515/679 | 0 |
| nominal-repeat-step512 | 32/32 | 679/679 | 679/679 | 0 |

Both step512 checkpoints produce exactly 679 predictions with natural EOS on
all32 fitted images and zero exact/annotation-owner revisits, invalid geometry,
parser drops or UNKNOWN. This replicates finite-panel natural-completion
feasibility across the two authorized seeds. It does not isolate address-module
benefit, superiority to ordinary SFT, transfer or a burst mechanism. Selected
seed1729/step512 remains fixed; seed2718 never replaces it.

The64-image monitor comparison remains separate. Source versus selected has
363 versus328 class-consistent IoU50 matches of630; clean completion23 versus21;
CE(token-weighted)1.68460 versus2.93591; EOS62 andcaps2 each. There are12 improved,
19 worse and33 equal images by match count. Dense images contribute net-35;
ordinary aggregate matches68/82 are unchanged despite individual gains/losses.

Do not interpret aggregate invalid/revisit reductions as broad improvement.
Source cap images65798/143572 account for invalid540->2 and exactrevisits109->0;
the other62 images worsen2->8 and6->8. Malformed drops rise2->257 and affected
parser-drop images4->25; flagged malformed-span characters159->79409. Image458325
newly caps with a35352-character malformed span. Total parser drops544->267 are
not fixed-size error units. Parsed predictions752->793, generatedtokens11879->14574,
UNKNOWN386->460. All owner matches/revisits remain annotation proxies; no new
physical adjudication was conducted. Specialization/forgetting is a working
explanation, not an established cause or address-component indictment.

## Execution and reusable implementation

The maintained `src.train` -> pipeline -> `SupervisedTrainer` route now supports
opt-in independent input/output selected rows, the concrete main-merger live
output-codebook module, explicit optimizer coverage, atomic composition/state
checkpointing and config-driven resume. Existing tied/no-injection defaults
remain neutral. Scale, data paths, seed, epochs/updates, batch and group LRs remain
configurable. Prepared full-data1/2/4epoch YAMLs remain unexecuted.

Both fits start fresh from the mature untied+axis001 source, with BF16/FA2 and
explicit promoted live PEFT adapters. All512updates per seed are finite/applied;
the global512x8pack schedules are checked against actual four-rank ordering.
Source and trained inference use the same declared runtime. Historical Mixin
BF16 outputs are not claimed equivalent to it.

The accepted qualification manifest is
`qualification/complete-v1/manifest.json` (SHA256
`7d99eed4bb83fa7ed6bbc6e79584bf70295a36925fc96ba3ab5284461c81507c`).
It retains the first-branch CPU/CUDA norm witness, new-only post-placement
initialization, exact three-case OFF/reload logits,903restored tensors,
real single/two-rank loss/gradient checks and mutation failures. Mature and
restored tensors are not reset. Resume is numerically checked, not bitwise:
adaptermax3.3587e-6, embeddingmax1.6494e-6, momentmax.0001432681,
relativeL2.0054508049; the strict failure and unresolved cause remain visible.

Final fresh-process evaluation readback checks all saved-to-loaded parameter
hashes against exact payloads. The saved reducer replays an explicit352-file
list byte-identically, including telemetry. Earlier monitor restricted replay
had identical scientific fields but different cells_loaded(320 versus128);
those original reductions remain unchanged. The unrelated JSON-list scan
failure was repaired only at CLI schema filtering; accepted nominal reduction
still replays byte-identically. A test-only compatibility projection preserves
historical config digests for neutral new defaults while detecting tie=false.
An outdated pipeline test double was updated to current neutral defaults.

## Cost, checks and handoff

Total model-execution wall time:2.537312hours. Allocated GPU time:6.775323hours
(24391.161394GPU-seconds), including eight failed qualification attempts and
all loading, fitting and evaluation. Both fits total1024updates. Training
forward counts are microstep-based and exclude checkpoint recomputation;
evaluation call counters are actual. Terminal artifact inventory was about
6.899GB before final closeout records. The original clock and limits were never
reset. All scientific cells completed; no unfinished denominator is hidden.

Final required knowledge/layout/diff checks and exact test logs are bound by the
candidate manifest. The unrelated historical active-profile inventory test
still fails because current profiles exceed its frozen inventory; the package
changes no active profile. Focused compatibility, source, checkpoint, optimizer,
address, resume and reducer checks pass. Failed CPU/model attempts remain saved.

The CPU-only1024/256proposal is in [next-stage-preparation.md](next-stage-preparation.md)
and the separate scale-preparation root's `candidate-v2.json`. It proposes one
fresh four-epoch trajectory with1/2/4epoch checkpoints, training-first ranking,
and explicit source-relative format/duplication tolerances for lead review.
959/1024training and248/256validation identities occur in the mature SFT source;
validation excludes currentfit/monitor and nexttrain, but is not untouched.
No next-stage launch is authorized by this candidate.
