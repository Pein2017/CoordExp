# Source at a Sixteen-Row Budget: vLLM Collection Receipt

## Purpose

This receipt records the finite Source baseline used to compare greedy owner
coverage with the sixteen low-temperature sampled trajectories. `Source@B16`
means the first sixteen clean, complete object rows from a repetition-penalty
1.0 greedy trajectory, or all clean rows when the model naturally ends sooner.
It does not use any row after the semantic budget.

The baseline is one immutable realization under a frozen execution policy. It
is not claimed to be invariant to inference batching.

## Evidence Roots

Production Source root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/
source-b16-vllm/production-v1
```

Same-policy qualification roots:

```text
source-b16-vllm/pilot-64-c-batch16
source-b16-vllm/pilot-64-d-batch16
```

Earlier batch-sensitivity roots:

```text
source-b16-vllm/pilot-64-a
source-b16-vllm/pilot-64-b
```

## Qualification Result

Changing only the request partition from one 64-image call to two 32-image
calls changed meaning-bearing output:

- 16 of 64 projected token hashes matched exactly;
- 60 of 64 eligibility statuses matched;
- one image changed from ineligible to accepted;
- three images changed from reaching the sixteen-row budget to natural end.

The production policy was therefore frozen at sixteen images per persisted
request batch. Two independent runs on different NVIDIA A100 80-gigabyte
Graphics Processing Units then matched on all 64 raw token hashes, projected
token hashes, projected texts, row counts, and eligibility statuses. Both had:

- 19 `accepted_budget` images;
- 43 `accepted_natural_end` images;
- 2 `failed_invalid_before_budget` images.

This supports exact replay under the frozen per-engine execution policy while
preserving the claim that other batch partitions may realize a different
greedy route.

## Frozen Production Policy

- inference runtime: vLLM 0.14.1;
- model computation: Brain Floating Point 16-bit;
- model length: 4,096 tokens;
- raw generation allowance: 2,048 tokens;
- scheduler capacity: `max_num_seqs=32`;
- persisted request batch: `image_batch_size=16`;
- eight stable strided image shards, 304 images and 19 batches per shard;
- greedy temperature 0, nucleus probability 1.0, top-k 0, repetition penalty
  1.0, and one completion per image;
- one engine and one Graphics Processing Unit at a time per worker;
- materialized execution-model composition key
  `2ae15bcc86014bcee1eda6e9bf1c79ffc246243b938a140ac5b851760b26d6ce`;
- execution-model snapshot fingerprint
  `4b3cdcacd9452811281572cfd6f1a984fc89e93442625ffdc11c6dab32809eff`.

All devices were NVIDIA A100 80-gigabyte PCI Express models with driver
550.54.15. PyTorch was 2.9.1 with CUDA 12.8. External co-tenancy caused two
initialization attempts to fail before generating data; those shards were run
unchanged on released identical devices. Exact realized device mapping is in
the worker manifests.

The uncommitted execution inputs were frozen by content:

```text
collector SHA-256:
d3feaed5fa40330e8071e39944af51c28072211be013e0ad1bf8f426672d5f9d

configuration SHA-256:
9c6b804f54a7b38c4d7f25eed69104b6900a7fb3fdb93008eb5e2d2f3e0a043e
```

## Production Result

The production root contains eight manifests and 152 immutable batch
artifacts. All 152 artifact hashes match their manifest values. The artifact
image set contains exactly 2,432 unique identifiers and is identical to the
frozen candidate-pool image set.

Source status counts:

| Source status | Images |
|---|---:|
| `accepted_budget` | 548 |
| `accepted_natural_end` | 1,838 |
| `failed_invalid_before_budget` | 46 |
| `failed_token_limit_before_budget` | 0 |

Thus 2,386 images have a usable Source baseline and 46 do not.

Raw stopping reason must be interpreted jointly with Source status:

| Source status | Raw stop | Images |
|---|---|---:|
| `accepted_budget` | natural image-end | 527 |
| `accepted_budget` | token length | 21 |
| `accepted_natural_end` | natural image-end | 1,838 |
| `failed_invalid_before_budget` | natural image-end | 43 |
| `failed_invalid_before_budget` | token length | 3 |

A raw token-length stop is not itself a Source failure when sixteen clean rows
were already completed.

Split distribution:

| Frozen split | Accepted | Ineligible | Total |
|---|---:|---:|---:|
| training candidate | 2,004 | 44 | 2,048 |
| development | 254 | 2 | 256 |
| held out | 128 | 0 | 128 |

Object-count distribution:

| Annotated objects | Accepted | Ineligible | Total |
|---|---:|---:|---:|
| 1-3 | 608 | 0 | 608 |
| 4-7 | 605 | 3 | 608 |
| 8-15 | 594 | 14 | 608 |
| 16 or more | 579 | 29 | 608 |

The ineligibility is density-dependent and must remain visible in later
admission and interpretation.

## Claim Boundary and Next Action

An ineligible image has no Source baseline; it does not have an empty Source
owner set. It may not enter the training-admission join and may not be rerun
under another batch grouping to change its status.

The next gate is to join only the 2,004 Source-eligible training-candidate
images with the first-sixteen-row sampled owner evidence, then measure whether
at least 496 images provide both a trusted sampled owner event and a trusted
Source-preservation event. No training conclusion follows from this collection
receipt alone.
