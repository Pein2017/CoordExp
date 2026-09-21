---
title: N256 fixed-surface resource census v1
type: investigation
role: planning-receipt
status: counts_only_no_runtime_budget
updated: 2026-09-01
---

# N256 fixed-surface resource census v1

These are exact workload counts or conservative array-size projections from the
frozen policy. They are not wall-time, RSS, feasibility, or model-quality
evidence.

## Full ladder

- QP cells: 8 semantic + 29 unique null = **37 solves**. N2 contributes one
  unique null; each later stage contributes four.
- One exhaustive separator sweep over every cell covers **206,904,224**
  registered comparisons. Optimizer/separator repetition is not yet measured.
- If nested semantic captures are reused, train capture still requires at most
  **256 semantic + 2,034 null = 2,290 teacher-forced image-route forwards**.
- Screen readout requires **37 payload processes plus one Source process** and
  **4,864 image decodes**. At the frozen cap this is a worst-case ceiling of
  **15,000,576 generated tokens**; natural EOS should lower it, but no timing
  estimate is registered yet.

## N2/N4 production-shaped slice

- QP solves: 2 at N2 and 5 at N4 = **7**.
- Train capture: **22 teacher-forced image-route forwards** when the four
  semantic images are reused.
- Screen readout: Source + 2 N2 payloads + 5 N4 payloads = **8 fresh
  processes**, **1,024 image decodes**, and at most **3,158,016 generated
  tokens**.

This screen cost is the material launch decision; a green CPU verifier does not
authorize it.

## N256 CPU array floors

For `P=18,745`, `R=1,136`, hidden width 2,048:

- FP64 hidden/projected states: **292.89 MiB**;
- FP64 fixed-row base logits: **162.46 MiB**;
- top-1,137 token IDs plus logits at 8 bytes each: **325.21 MiB**;
- one Human13-compatible FP64 residual payload: **17.75 MiB**;
- current active-operator chunk upper bound: **128 MiB** per weighted/gather
  workspace at 8,192 constraints.

SVD/LAPACK, model state, Python objects, optimizer history, captures, and
artifact serialization add to these floors. Historical N13 used 8,079 active
constraints and 76,176 optimizer iterations; projected N256 active burden is
roughly 42k-61k, not an acceptance fact.

## Unfrozen gate

Before any model/GPU command, the user must accept wall time, host RSS, GPU
memory, active-constraint, artifact-byte, attempt, and corrected-launch limits.
Until then the only valid disposition is **HOLD before model construction**.

The separately authorized N2 plumbing smoke is smaller: four teacher-forced
train captures, two CPU solves, and six natural screen decodes across three
fresh screen processes. It has its own two-hour/40-GiB-GPU/64-GiB-host/10-GiB-
artifact/20k-active hard limits and cannot own a scientific verdict.
