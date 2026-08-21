# P2 Triage — 2026-08-21 Codex Adversarial Review

Builder D (verify-only). All source read from the working tree; `git status
--short -- src/` was clean at read time (no dirty files from Builders A/B/C),
so the working tree equals `HEAD` for every file cited below — no `git show
HEAD:<path>` fallback was needed. Repo root:
`/data/CoordExp/.worktrees/CoordExp-swift`.

---

## Claim 1 — Terminal row loses known finite truth

**VERDICT: CONFIRMED (narrow — `finite_status` only; the nullable booleans the
spec calls out by name are correctly preserved).**

### Evidence

`docs/ARTIFACTS.md:98-106` and
`openspec/specs/coordexp-swift-training-artifacts/spec.md:433-443` both
require the terminal train row to carry "truthful nullable `applied` and
`step_was_skipped`, `mutation_state`, finite/update status" — i.e. null only
where genuinely unknown, not as a blanket default.

For the two *known* boolean cases the spec calls out
(`spec.md:397-401`, "unanimous result that contradicts the expected action"),
the code is correct:

- `src/runtime/optimizer_boundary.py:337-375`
  (`AppliedUpdateReceipt.terminal_post_wrapper`) — `apply + all_skipped` sets
  `applied=False, step_was_skipped=True` (line 345-346); `scaler_skip +
  none_skipped` sets `applied=True, step_was_skipped=False` plus the real
  pre-call LRs (line 353-360). Only the true `mixed` outcome nulls both
  (line 338-342).
- `src/training/reporting.py:249-258` (`_boundary_truth_fields`) projects
  `receipt.applied` / `receipt.step_was_skipped` / `receipt.mutation_state`
  verbatim into the row — no re-nulling.

But `finite_status` is a different story. `src/runtime/finite_gates.py:216`
declares `GateDecision.finite_status: str`, and it is **always** populated —
including on the terminal branch — as a pure function of the all-rank-gathered
`all_safe`:

```
src/runtime/finite_gates.py:339   finite_status="finite" if all_safe else "non_finite",
```

(same pattern at line 275 for the pre-backward-scalar stage). This value is
known and all-rank-converged *before* `_reduce_boundary_action`
(`finite_gates.py:368-386`) even decides whether the boundary is terminal —
`all_safe` is computed first, `terminal_reason` second.

`src/training/supervised_trainer.py:366-367` and `:396-400` capture this
correctly into a local `finite_status` variable on the very call that reaches
the terminal boundary:

```
367:  finite_status = pre_decision.finite_status
400:  finite_status = post_decision.finite_status
```

But the boundary call one line later
(`src/training/supervised_trainer.py:407-410`,
`self.runtime.execute_optimizer_boundary(...)`) raises
`OptimizerBoundaryTerminal` on the terminal path
(`src/runtime/train_runtime.py:250-260` and `:305-316`), and that exception
carries **only** the receipt:

```
src/runtime/optimizer_boundary.py:400   def __init__(self, receipt: AppliedUpdateReceipt) -> None:
```

`AppliedUpdateReceipt` has no `finite_status` field at all
(`optimizer_boundary.py:108-127`), and neither `terminal_not_attempted`
(`:267-304`) nor `terminal_post_wrapper` (`:306-375`) accepts one. So the
known, locally-computed `finite_status` in `supervised_trainer.py` is dropped
on the floor — it never reaches the receipt or the exception. By the time
`build_terminal_boundary_row` runs, it hardcodes:

```
src/training/reporting.py:749   "finite_status": "unavailable",
```

with a comment claiming "the gate decision is gone by this point" — true
*given the current object design*, but only because the exception was built
to not carry it, not because the value is inherently unknowable.

The existing test locks this in and reveals the actual original intent:
`tests/training/test_reporting.py:1173-1204`, specifically the comment at
1194-1196: *"Never the previous COMPLETED step's finite status relabeled as
this planned step's own truth."* This shows the `"unavailable"` hardcode was
built to fix a **different**, real bug — stale `lifecycle["finite_status"]`
(from the last completed step) leaking into the terminal row — and overshot by
discarding the terminal boundary's own freshly-known value along with it.

### Severity / reachable scenario

Concrete: a post-backward gradient consensus finds all ranks agree gradients
are non-finite (`all_safe=False` → `finite_status="non_finite"`, known,
all-rank-converged), and the boundary happens to hit any terminal branch (e.g.
`TERMINAL_PRE_WRAPPER_SCALER_CANDIDACY_DIVERGENT`,
`finite_gates.py:380-386`, where scaler-active ranks disagree but gradient
safety was already decided). The published `logging.jsonl` terminal row will
read `finite_status: "unavailable"` even though the runtime computed
`"non_finite"` moments earlier. This does not corrupt any decision (the
correct action/mutation_state are still preserved) — it is an observability
gap: a human or tool triaging *why* a run terminated loses a known, cheap
diagnostic signal. Bounded severity, matches P2.

### Cheap-fix proposal

- Add a `finite_status: str` field to `AppliedUpdateReceipt`
  (`src/runtime/optimizer_boundary.py`), threaded through
  `terminal_not_attempted` / `terminal_post_wrapper` (and the normal
  constructors, for symmetry — they can pass `"finite"` since a normal
  boundary only reaches those constructors when `all_safe` was true).
- Pass `decision.finite_status` at both terminal-raise sites in
  `src/runtime/train_runtime.py:254-259` and `:309-315`.
- Replace `src/training/reporting.py:749`'s hardcode with
  `receipt.finite_status`, keeping `"unavailable"` only as a genuine fallback
  if a future construction path omits it.
- Update `tests/training/test_reporting.py:1196` (currently asserts
  `"unavailable"`) and `tests/artifacts/test_observation_publisher.py:454-460`
  to assert the real threaded value.

**Size**: small (~4 files, one new dataclass field, two call-site edits, one
hardcode removal, two test updates).

**Overlap flag**: `src/runtime/train_runtime.py` is Builder B's owned file
(P1-2, declared-fp16/GradScaler fail-closed), and `src/runtime/finite_gates.py`
is also Builder B's, though this fix does not need to touch
`finite_gates.py` itself (only reads the already-existing
`GateDecision.finite_status`). `tasks.md` already anticipates this: *"terminal-
row P2 lands only after B, same files"* — confirmed correct sequencing.
`src/runtime/optimizer_boundary.py` and `src/training/reporting.py` are
**not** owned by any Wave-1 builder — safe to touch directly, but land after B
to avoid a merge race on `train_runtime.py`.

---

## Claim 2 — `TrainingExecutionPlan` is only shallowly immutable

**VERDICT: CONFIRMED (structurally true), but no reachable mutation path
exists in current production code — theoretical, not actual.**

### Evidence

`src/training/execution_plan.py:76-88` — `TrainingExecutionPlan` is
`@dataclass(frozen=True)`. Its own container fields are properly deep-frozen:
`measurement_context` and `entry_resources` are wrapped in
`MappingProxyType(dict(...))` at construction
(`execution_plan.py:108, 111`), and the underlying dict is never aliased
elsewhere (a fresh `dict(...)` copy feeds the proxy).

But `resolved_config: ResolvedTrainConfig` is itself a frozen dataclass
(`src/config/models.py:579-588`) whose own fields include two **plain mutable
dicts**, not proxies:

```
src/config/models.py:582   config_dict: dict[str, Any]
src/config/models.py:588   path_origins: dict[str, PathOrigin]
```

`frozen=True` only blocks *rebinding* `plan.resolved_config = X`; it does
nothing to stop `plan.resolved_config.config_dict["foo"] = "bar"` or
`plan.resolved_config.path_origins.clear()`. The rest of `ResolvedTrainConfig`
is properly hardened for comparison: `sources: tuple[ConfigSource, ...]`
(`:587`) and the `path_origins` *values* are themselves frozen dataclasses
(`ConfigSource` at `models.py:556-561`, `PathOrigin` at `:564-576`) — only the
two dict *containers* are the gap.

`to_artifact_dict()` (`models.py:590-604`) makes this worse in kind, not just
in theory: it returns `"config": self.config_dict` — the **live internal
dict**, not a copy. Any caller of `resolved_config.to_artifact_dict()` that
mutated the returned payload (rather than only reading/serializing it) would
silently corrupt the plan's own state for every other holder of the same
instance.

I checked for an actual mutation path and found none in production code:

```
grep -rn 'config_dict\[.*\]\s*=|config_dict\.update|path_origins\[.*\]\s*=|...' src/
  -> src/config/inference.py:675 and src/config/paths.py:70
```

Both hits are `path_origins[field] = PathOrigin(...)` writes happening
**during construction** of a fresh `path_origins` dict *before* it is passed
into `ResolvedTrainConfig(...)` — not a mutation of an already-constructed,
already-frozen plan. The two callers of `to_artifact_dict()`
(`src/config/writer.py:30`, `src/training/session.py:1049, 2112`) only read
the payload for JSON/YAML serialization; none mutates it.

### Severity / reachable scenario

No currently-reachable mutation exists. The risk is latent: `config_dict`
travels as a live reference through `to_artifact_dict()`, and any future
caller that treats the returned payload as an owned copy (e.g. to patch a
field before re-dumping it) would silently corrupt the shared frozen plan for
every other consumer in the same process — a hard-to-diagnose spooky-action
bug precisely because the type signature (`@dataclass(frozen=True)`) implies
full immutability. Bounded severity today; real API-safety debt.

### Cheap-fix proposal

- In `ResolvedTrainConfig.__post_init__` (needs adding —
  `src/config/models.py`), wrap `config_dict` and `path_origins` in
  `MappingProxyType`, mirroring the pattern `execution_plan.py` already uses
  for `measurement_context`/`entry_resources`. Frozen dataclasses can still
  assign in `__post_init__` via `object.__setattr__`.
- Alternatively/additionally, have `to_artifact_dict()` return a shallow copy
  instead of the live dict if full proxy-wrapping is judged too invasive
  (would need `dict(self.config_dict)`).

**Size**: small, isolated to `src/config/models.py` (one `__post_init__`,
type annotations become `Mapping[str, Any]` at the dataclass field level, or
kept as `dict` with only the instance wrapped). Type-checker fallout is the
main cost — every current reader of `.config_dict`/`.path_origins` that
assumes a plain `dict` (e.g. `.get`, iteration) still works against
`MappingProxyType`; only a caller that assumes write access would break, and
none currently does per the grep above.

**Overlap flag**: none. `src/config/models.py` is not owned by any Wave-1
builder.

---

## Claim 3 — Phase finalization has dual owners

**VERDICT: PARTLY — a local mirror exists alongside the durable owner, but
their writes cannot disagree *within one process's run*; the real gap is
narrower (a crash-then-resume path with no explicit reconciliation), not
concurrent dual-writer disagreement.**

### Evidence

`src/artifacts/run_writer.py:1-10` (module docstring) explicitly claims sole
ownership: *"this module remains the import and I/O facade for `RunWriter`
[...] it is the only owner that reads/writes run files and sequences
transitions."* It owns the durable `measurement["active_phase"]`,
`measurement["phases"]`, `measurement["phase_order"]` state via three methods:
`begin_phase` (`:360-398`), `finish_phase` (`:687+`), and a second write path,
`record_phase_summary` (`:543-685`), used for aggregate phases recorded
without ever being opened (steady_state / evaluation_execution summaries).

`src/training/session.py:1-14` (module docstring) also claims to "own" **"run
finalization"** among a long list of what `TrainingSession` owns — worded
ambiguously enough that a reviewer reading both docstrings side by side would
reasonably flag competing ownership claims.

In practice, `session.py` does not maintain an independent durable phase
record. `_start_run_phase` (`session.py:169-197`) calls `writer.begin_phase()`
**first** (line 190-194) and only then sets the local
`lifecycle["active_phase"] = phase` (line 195) — strictly sequenced, same
thread, same call. `_finish_run_phase` (`session.py:199-252`) similarly
guards on the local mirror (line 211), then calls `writer.finish_phase()`
(line 240-250), then clears the mirror (line 251). Within one process's
synchronous execution, the writer call and the mirror update cannot diverge —
there is no concurrency and no exception window between them wide enough to
leave them disagreeing in a way that matters (if `writer.begin_phase()`
itself raises, `lifecycle["active_phase"]` is never set, so the two stay
consistent).

The second write method, `record_phase_summary`
(`run_writer.py:543-685`), is not a second *owner* — it is a second *method
on the same owner file*, and it re-validates against the same durable state
before writing: `if measurement["active_phase"] is not None: raise`
(`:656-661`) and `if phase in measurement["phases"]: raise` (`:662-667`). It
cannot silently overwrite what `begin_phase`/`finish_phase` already wrote.

`src/training/session.py:961-990`
(`_record_evaluation_summary_after_failure`) confirms `session.py` treats
`run_writer.py` as ground truth rather than trusting its own state: it calls
`writer.read_run()["measurement"]["phases"]` (line 967) to check whether a
phase is already recorded *before* deciding to call
`writer.record_phase_summary(...)` — the durable writer state, not the local
`lifecycle` dict, is consulted.

The one real gap: I checked `src/artifacts/run_state.py` and
`src/training/exact_resume.py` for `active_phase` handling on resume and
found none (`grep -n "active_phase" src/artifacts/run_state.py
src/training/exact_resume.py` returns nothing). If a process crashes while
`measurement["active_phase"]` is non-null (no chance to call `finish_phase`),
a resumed process starts with a fresh, empty `lifecycle = {}` (mirror reset to
None) with no reconciliation step. This is not silent, though: the *next*
`begin_phase()` call on the resumed process will hit
`run_writer.py:373-379`'s own guard (`"a run phase is already active"`) and
raise `ArtifactContractError` — fail-closed, but as a late, possibly
confusing failure deep in the next phase transition rather than an explicit,
early resume-time diagnostic.

### Severity / reachable scenario

Within a single run: not reachable — sequencing prevents divergence. Across a
crash-then-resume boundary: a stale `active_phase` left by a hard crash
surfaces only when the resumed process next calls `begin_phase()`, as a
generic `run_writer.phase_already_active` contract error, not a targeted
"resume found dangling phase state" diagnostic. This is closer to a
"confusing failure mode" than data corruption or a decision-changing bug — the
run does fail loudly, just later and less specifically than it could.

### Cheap-fix proposal / DEFER

**DEFER.** This is not "confirmed and cheap" in the sense of the P1s: fixing
it properly means deciding what resume *should* do with a dangling phase
(clear it? refuse to resume entirely with a named diagnostic? attempt
recovery?) — that's a resume-semantics decision, not a mechanical fix, and
belongs with whoever owns `src/training/exact_resume.py` /
`src/artifacts/run_state.py` resume-admission logic. A follow-up would need:
(1) an explicit decision on resume-time handling of a dangling
`active_phase` (likely: refuse resume with a named contract error citing the
stuck phase, rather than deferring to the next `begin_phase()` call), (2) a
2-rank or crash-simulation test fixture that leaves `measurement` mid-phase
and exercises resume. Separately, the docstring wording collision (`session.py`
claiming to "own... run finalization" vs. `run_writer.py` claiming to be "the
only owner that reads/writes run files") is a cheap, no-risk wording fix if
the lead wants it: reword `session.py`'s docstring to "orchestrates run
finalization by calling the run_writer owner" — but that alone does not
resolve the resume gap and shouldn't be conflated with fixing it.

**Overlap flag**: `src/training/session.py` and `src/artifacts/run_writer.py`
are not owned by any Wave-1 builder; `src/training/exact_resume.py` /
`src/artifacts/run_state.py` likewise free.

---

## Claim 4 — Import-boundary guard misses `from package import member`

**VERDICT: CONFIRMED — reproducible from the guard's own test fixture, and a
matching import style already exists in production code.**

### Evidence

The guard lives in
`tests/training/test_training_module_boundaries.py`. `imported_modules()`
(`:218-238`) AST-parses each production file and, for `ast.ImportFrom` nodes:

```python
228: elif isinstance(node, ast.ImportFrom):
229:     if node.level:
230:         base = package_parts[: max(0, len(package_parts) - (node.level - 1))]
231:         module = ".".join([*base, node.module] if node.module else base)
232:     else:
233:         module = node.module or ""
234:     if module:
235:         names.add(module)
```

For an **absolute** `from src.training.pipeline import run_training_pipeline`,
`node.module` is already the fully-dotted `"src.training.pipeline"`, so this
is caught correctly — `FORBIDDEN_UPWARD_TARGETS = ("src.training.pipeline",
"src.training.session")` (`:174-177`) matches it in
`observed_forbidden_upward_edges` (`:263-274`).

But when the imported *member* is itself the target submodule name — i.e.
`from src.training import pipeline` (absolute) or `from . import pipeline`
(relative, from inside `src/training/`) — `node.module` is only
`"src.training"` (absolute case) or resolves to `"src.training"` via the
`base`-only branch (relative case, since `node.module` is `None` for `from .
import X`). The specific submodule name (`pipeline`) lives only in
`node.names[i].name`, which this function never reads for `ImportFrom` nodes.
The result: `names.add("src.training")` — never
`"src.training.pipeline"` — so the forbidden-target check silently misses it.

This is not a hypothetical reading of the AST branch — the guard's **own**
test proves it, unintentionally documenting the gap as expected behavior.
`tests/training/test_training_module_boundaries.py:399-418`
(`test_import_parser_reads_absolute_and_relative_imports`) feeds the parser:

```
399:  "from . import pack_cache\n"
```

and asserts (`:412-417`) the observed module set is
`("src.artifacts", "src.qwen.parity", "src.training", "src.training.pipeline")`
— note `"src.training"`, **not** `"src.training.pack_cache"`. The test's own
docstring comment (`:434-438`) even explains the *intended* purpose of this
resolution (correctly resolving the package for relative imports) without
noticing it also erases the submodule name for exactly this import shape.

**Concrete bypass** (not added to the repo, per instructions): a file in
`LEAF_AND_DOMAIN_OWNERS` (e.g. `src/training/execution_plan.py`, which is
forbidden from reaching `src.training.pipeline`/`src.training.session`) could
add either

```python
from src.training import pipeline      # absolute form
# or, from inside src/training/:
from . import pipeline                 # relative form
```

and then reference `pipeline.run_training_pipeline(...)`. Both forms register
only `"src.training"` in `imported_modules()`'s output, which is not in
`FORBIDDEN_UPWARD_TARGETS`, so `observed_forbidden_upward_edges()` reports no
violation and `test_leaf_and_domain_owners_never_import_the_facade_or_session`
(`:384-385`) stays green despite the forbidden reverse edge existing.

This import style is not purely hypothetical — it is already idiomatic in
this codebase: `src/training/reporting.py:84` has
`from src.training import cache_workflow` (a `LEAF_AND_DOMAIN_OWNER` file
using exactly this collapsing form). `cache_workflow` is not a forbidden
target today, so this specific line is harmless, but it demonstrates the
style is already in use and would collapse identically to `"src.training"` if
the imported name were ever `pipeline` or `session` instead.

### Severity / reachable scenario

No current violation exists (checked: no `LEAF_AND_DOMAIN_OWNERS` file
currently imports `pipeline` or `session` via any form,
`observed_forbidden_upward_edges(REPO_ROOT)` would return `()` today). The
risk is a **regression that goes undetected**: a future edit that
reintroduces exactly the reverse dependency this guard exists to prevent, if
written in the collapsing style already used at `reporting.py:84`, would pass
CI silently. Given the codebase already has that style as precedent, this is
a real (if latent) coverage gap in a fault-detection surface, which per the
project's own contract ("Silent-corruption surfaces... are guarded by
invariant assertions") deserves closing even though nothing is broken today.

### Cheap-fix proposal

In `imported_modules()` (`tests/training/test_training_module_boundaries.py`,
inside the `ast.ImportFrom` branch), also register the member-qualified
candidate for every imported name, not just the resolved module:

```python
for alias in node.names:
    if alias.name != "*":
        names.add(f"{module}.{alias.name}" if module else alias.name)
```

added alongside the existing `names.add(module)`. This is a safe
over-approximation: the only consumers of `imported_modules()` output are
membership checks against small closed sets
(`FORBIDDEN_UPWARD_TARGETS`, `CACHE_WORKFLOW_FORBIDDEN_IMPORTS`, the
`src.qwen.parity` prefix check) — adding extra, mostly-harmless
member-qualified strings (e.g. `"src.qwen.parity.canonical_json_bytes"`)
cannot produce a false positive against those checks since they test for
either exact equality against the closed target set or a `.` prefix that
member-qualified names would only *additionally* satisfy when already true.
The two hardcoded-expectation tests
(`test_import_parser_reads_absolute_and_relative_imports:399-418`,
`test_import_parser_resolves_relative_imports_inside_a_package_init:420-438`)
would need their expected tuples extended to include the new
member-qualified entries (e.g. add `"src.training.pack_cache"` alongside the
existing `"src.training"`) — this is a RED-first opportunity: add a new test
asserting `from . import pipeline` (or `session`) *is* caught as a forbidden
edge, watch it fail against the current parser, then land the fix.

**Size**: small — one function, ~2-3 lines; two existing test expectations
widened; ideally one new RED test demonstrating the previously-missed
forbidden edge.

**Overlap flag**: none. `tests/training/test_training_module_boundaries.py`
is not owned by any Wave-1 builder and touches no file any of A/B/C are
editing. As a **verify-only** builder I did not construct this fix in the
repo — the snippet above is illustrative only, per the task instructions.

---

## Summary for the lead

| # | Claim | Verdict | Cheap fix? | Files | Builder-B overlap |
|---|---|---|---|---|---|
| 1 | Terminal row loses known finite truth | CONFIRMED (finite_status only) | Yes, small | `optimizer_boundary.py`, `train_runtime.py`, `reporting.py`, 2 tests | Yes — `train_runtime.py`; land after B per `tasks.md` |
| 2 | `TrainingExecutionPlan` shallow immutability | CONFIRMED (structural), no reachable mutation found | Yes, small | `src/config/models.py` | None |
| 3 | Phase finalization dual owners | PARTLY — sequencing prevents in-process divergence; real gap is resume-time reconciliation | DEFER (resume-semantics decision needed) | `session.py`/`run_writer.py` docstring wording is a free cheap fix if wanted | None |
| 4 | Import guard misses `from package import member` | CONFIRMED — reproducible from guard's own test fixture, matching style already at `reporting.py:84` | Yes, small | `tests/training/test_training_module_boundaries.py` | None |

**Observations on other builders' domains, for integration:**

- Builder B's files (`src/runtime/train_runtime.py`,
  `src/runtime/finite_gates.py`) are exactly where P1-2's declared-fp16
  refusal and Claim 1's cheap fix both land. `finite_gates.py`'s
  `GateDecision.finite_status` (already correct, already all-rank-converged)
  is the value Claim 1's fix threads through — Builder B does not need to
  change `finite_gates.py` for this, only `train_runtime.py`'s two
  terminal-raise call sites gain one new keyword argument each. Sequence
  Claim 1's fix strictly after B's P1-2 lands to avoid a merge race on the
  same two call sites (`train_runtime.py:250-260`, `:305-316`) that P1-2 is
  also likely to touch (declared-fp16-without-scaler is itself one of the
  `not_attempted`/terminal paths).
- Builder A (`src/losses/runner.py`) and Builder C
  (`src/training/cache_workflow.py`, `src/training/pack_cache.py`, new
  assembler module) have no overlap with any of the four P2 findings or their
  proposed fixes.
- `src/training/reporting.py`, `src/runtime/optimizer_boundary.py`,
  `src/config/models.py`, `src/artifacts/run_writer.py`,
  `src/training/session.py`, and
  `tests/training/test_training_module_boundaries.py` are all currently
  unowned by any Wave-1 builder — free for whoever picks up the confirmed
  cheap fixes (Claims 1, 2, 4) in Wave 2 integration.

---

## Lead dispositions (Wave 2 integration, 2026-08-21)

- **Claim 1 (terminal finite_status): FIXED.** RED-first (new test failed with
  `TypeError: unexpected keyword argument 'finite_status'` against the old
  constructor API, tests/training/test_reporting.py:1228). `AppliedUpdateReceipt`
  gained a closed-set `finite_status` field ("finite"/"non_finite"/"unavailable",
  default preserved for constructor paths without a gate decision); both
  terminal constructors accept it; both `train_runtime.py` terminal-raise sites
  thread `decision.finite_status`; `reporting.py` publishes
  `receipt.finite_status` instead of the hardcode. The original stale-lifecycle
  protection is retained and re-asserted (the value travels in the receipt,
  never from the lifecycle mirror). tests/training/test_reporting.py +
  tests/runtime/test_fp16_scaler_contract.py: 62 passed.
- **Claim 2 (shallow immutability): FIXED (alias-severing variant).** The
  proposed `MappingProxyType` wrap was implemented, then REJECTED on evidence:
  27 test errors (`TypeError: cannot pickle 'mappingproxy' object`) proved
  deep-copying `config_dict` is a legitimate existing consumer pattern.
  Landed instead: `ResolvedTrainConfig.__post_init__` deep-copies both mapping
  fields at construction (severs every caller alias) and `to_artifact_dict`
  returns a deep copy (severs consumer aliases). New isolation test in
  tests/config/test_train_config.py; tests/config/ + input attestation:
  263 passed. Sibling `ResolvedInferenceConfig` has the same latent pattern
  and was intentionally left out of scope (observed, not fixed).
- **Claim 4 (import-guard bypass): FIXED.** RED-first (both widened parser
  expectation tests failed against the old parser). `imported_modules` now
  also records `{module}.{member}` for every `ImportFrom` alias (safe
  over-approximation; `*` excluded). The widened guard is green against the
  whole production tree — no previously-hidden forbidden edge existed.
  tests/training/test_training_module_boundaries.py: 20 passed.
- **Claim 3 (phase finalization / dangling resume phase): DEFERRED** per
  Builder D's analysis — in-process disagreement is impossible (strict write
  ordering, single durable owner); the real gap is a resume-semantics decision
  (what exact-resume should do with a crash-dangled `measurement["active_phase"]`)
  owned by the user, needing an owning change with a crash-simulation test.
  Recorded here as the named follow-up; no code touched.
