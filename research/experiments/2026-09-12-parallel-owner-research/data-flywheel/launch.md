# Frozen execution packet — waiting for root GPU grant

Active packet: raw `data-flywheel/preparation-v2/pilot.json`.
The six-shard preparation-v1 is superseded, never launched. Final endpoint
placement is all8physical GPUs, no further adaptive allocation in this packet.

## CPU acceptance already run

```bash
python -m pytest -q -p no:cacheprovider research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-12-parallel-owner-research/data-flywheel/test_pilot.py
python -m probes.parallel_owner_research.data_flywheel verify
python -m probes.parallel_owner_research.training verify --input /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/data-flywheel/preparation-v2/training-inputs.json
bash -n research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-12-parallel-owner-research/data-flywheel/run_endpoint.sh
git diff --check
```

Six focused tests use the real immutable417044 source: original34rows map to
25reviewed owners; repeated predictions cannot create a second owner;
changing a real owner box lowers25→24; all34literal token rows decode exactly;
the25compacted histories preserve exactly3original conditions and change22;
the cold endpoint is exactly384trained plus1freshStable50 native image.
The unchanged training engine was not requalified.

## Exact training and cold-check commands after grant

Run from `/data/CoordExp/.worktrees/research-probes`; preserve each real process
exit and full log under raw `data-flywheel/`.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m probes.parallel_owner_research.training launch \
  --input /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/data-flywheel/preparation-v2/training-inputs.json \
  --arm screened25 --world-size 2 \
  --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/data-flywheel/training-v1

CUDA_VISIBLE_DEVICES=0 python -m probes.parallel_owner_research.training cold-check \
  --input /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/data-flywheel/preparation-v2/training-inputs.json \
  --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/data-flywheel/training-v1

python -m probes.parallel_owner_research.data_flywheel prepare-endpoint
```

The last command mechanically binds the final32-update adapter and passed
cold receipt to the already sealed scientific packet. It does not select an
endpoint, modify labels or inspect natural quality. Before evaluation, it
requires matching input/arm/update/cold-adapter identities.

## Natural endpoint and readback

```bash
bash research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-12-parallel-owner-research/data-flywheel/run_endpoint.sh
python -m probes.parallel_owner_research.data_flywheel reduce
python scripts/visualize_detection.py compare \
  --left-run-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/data-flywheel/endpoint-v1/visual-inputs/Stable50 \
  --right-run-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/data-flywheel/endpoint-v1/visual-inputs/screened25 \
  --left-label Stable50 --right-label screened25 \
  --out-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/data-flywheel/endpoint-v1/visuals \
  --row-id coco2017_train_000000417044
```

The shell starts8fixed independent processes once and blocks on their actual
PIDs. It performs no polling, retries, output filtering or queue scheduling.
Each process has its own complete log, outer exit file, raw rows, native
model identity and terminal resource receipt. There are no distributed
collectives in this endpoint. Physical GPUs0..7 map literally to shards0..7.

The reducer re-decodes tokens with the local tokenizer and reruns the native
parser/global matcher. Legacy baseline384 is reused from its source-bound
artifact; the primary image gets a fresh original-prompt Stable50 decode.
Both primary outputs are rendered against unchanged originalGT. The separate
reviewed-owner sidecar is not written into GT.

## Frozen counters and invocation guards

| Phase | Exact work / bound |
|---|---|
| Training rank0 / rank1 | 1802 /1752model and image forwards |
| Training per rank | 1696backwards,32synchronized backwards,32optimizer updates |
| Training wall | ≤10,000s per rank |
| Training memory | ≤32GiB CUDA allocated/reserved;≤40GiB RSS |
| Cold reload | 25score forwards, one model load; exact discrete scores and existing≤1e-5live/cold tolerance |
| Endpoint shards | 49 /48 /48 /48 /48 /48 /48 /48natural decodes |
| Endpoint maximum per shard | ≤151,116model forwards;≤49image forwards;≤12,000s |
| Endpoint memory | ≤32GiB CUDA allocated/reserved and RSS |
| Generation | Original prompt, empty detection prefix, greedy,RP1,cap3084, noKV intervention |

Endpoint model loads are2onshard0 (Stable50 then trained) and1each onthe
other7shards. Resource guards are invocation bounds, not spending ceilings.
Actual measured counters/resources must be reported separately. The cold
check reuses the qualified engine as-is; its resource receipt is measured,
not claimed to have a new hard memory/wall guard.

## Final acceptance and stop

The primary25-owner strong criterion remains jointly gated by retained
baseline owners, EOS, noinvalid/malformed rows and actual visual confirmation
of no physical duplicate/group extents. The reducer deliberately leaves that
visual gate pending; an all25geometric score alone cannot auto-promote.

Legacy partitions are disjoint trained1 /normal56 /other327. Report allGT
IoU50/60/80gains/losses and counts, plus baseline/trained valid rows, strict
duplicates, geometry/other drops, caps/EOS and sequence lengths. No fresh256
selection/evaluation, no adaptive dose, no automatic promotion. Stop after
this single endpoint and root scientific acceptance, including useful partial
or negative outcomes.
