# DORA owner learning

This maintained package contains Source256 preparation, raw-softmax bank
collection and CE/K4-RLOO updates. `owner_outcome.py` independently retains the
self-rollout coordinate/full-action credit objectives and a native replay
consumer. It does not migrate the autonomous experiment scheduler, N2 solver,
unfinished feedback studies, or historical evaluations.

## Explicit profiles

| Profile | Entry | Fixed reduction |
| --- | --- | --- |
| Source256 CE | `prepare --arm ce`, then `train` | Mean over images of each native GT action's mean token NLL, including EOS |
| Source256 K4-RLOO | `sample`, `prepare --arm rloo`, then `train` | `-sum(advantage * sum(action logp)) / (N * 4)`; advantage is reward minus the other three rewards' mean |
| Owner-outcome coordinate | `OwnerOutcomeScorer(..., scope=COORDINATE_SCOPE)` | `-sum(credit * selected_coordinate_logp) / (16 * 4)` |
| Owner-outcome full action | `OwnerOutcomeScorer(..., scope=FULL_ACTION_SCOPE)` | `-sum(credit * all_action_logp) / (16 * 4)` |

The training contributions multiply these global objectives by world size to
compensate for DDP averaging. Owner-outcome callers supply the retained generated
prefix, exactly four coordinate positions and the chosen credit; they own the
16-image/four-branch population and the forward/backward synchronization scope.
Its coordinate and full-action formulas are separate from Source256 CE/RLOO.

`configs/source256.yaml` is the fully resolved original Source256 model/input
profile. It uses the public `src.config.inference.load_research_infer_config`
loader for shared YAML/JSON resolution, `InferConfig` validation and fingerprinting,
without production-only config-directory and leaf-authoring restrictions or debug
mutations. The retained resolved fingerprint
is `7f8448a8d8e62442bea1e9b1ffa45921a876510d6c9f12370949c6a418d1a4bb`.
The old `generation`/`scoring`/`artifacts` settings describe the inherited strict
inference profile; native bank collection explicitly uses temperature 1, top-p 1,
top-k 0, repetition penalty 1, a fresh generation config, and **no traces**.
It keeps interior pad IDs and omits observed terminal EOS from bank bodies;
preparation appends EOS only for an observed `im_end` stop.

The model is loaded once through public Qwen/adapter/embedding owners. The
`source256_loaded_policy.v1` descriptor means loaded-model identity only;
`native_execution` separately records the actual replay or sampling behavior.
No inference session, canonical scored-greedy receipt, hidden model, or sibling
worktree import is used.

## CPU entry and checks

From an independent checkout with the documented shared input/model paths:

```bash
python -m probes.dora_owner_learning.preflight --rows 1
python -m pytest -q probes/dora_owner_learning/tests
python -m probes.dora_owner_learning.prepare --help
python -m probes.dora_owner_learning.sample --help
python -m probes.dora_owner_learning.train --help
```

Preflight checks the saved 256-image input and encodes one to eight rows without
loading model weights. `--plan /absolute/path/plan.json` additionally validates a
specified immutable plan and its input/continuation bindings. Tests include the
original Source256 oracles, real saved-input CE encoding, cold AdamW identity
errors, coordinate/full-action gradient comparisons, literal raw-softmax token
projection and two-process CPU DDP reduction/communication sensitivity.

The complete retained model path is:

```bash
python -m probes.dora_owner_learning.sample --output /absolute/path/bank.json
python -m probes.dora_owner_learning.prepare --arm rloo --round 1 \
  --rollout-artifact /absolute/path/bank.json --output /absolute/path/plan.json
torchrun --nproc_per_node=8 --module probes.dora_owner_learning.train \
  --plan /absolute/path/plan.json --output-root /absolute/path/update
```

For CE, preparation omits the bank. Subsequent rounds require a resolved profile
pointing to that arm's saved adapter, fresh RLOO banks where applicable, and the
immediately preceding `--previous-receipt`. The optimizer state must match ordered
parameters, hyperparameters, moment shapes/dtypes, step and adapter/embedding
identity. Mid-round resume remains unsupported. These model commands require a
separately authorized run; the migration's acceptance is CPU-only.

## Preserved provenance

Original DORA producers and tests are recoverable at
`archive/research-restructure-20260909/dora-prox-linear-n2`
(`ba801de514143d8fd25822180323e3e2b101bcd4`), under
`scripts/research/{prepare_source256_ce_rloo_round,train_source256_ce_rloo_round,run_current_seeded_sampled_rollouts}.py`.
Original self objectives are recoverable at
`archive/research-restructure-20260909/self-rollout-behavior`
(`d6de155fb5f80b178e438e3036462b58eda6420c`), under
`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-owner-outcome-autonomous/learning/train.py`.
These are provenance locators, not executable dependencies. Original receipts
remain unchanged; new plans/updates identify the current code/config inputs.
Source recovery and CPU checks do not establish real-model numerical parity or
new owner-recovery evidence. Research interpretation belongs to the
[investigation](../../research/investigations/qwen3-vl-dense-enumeration/).
