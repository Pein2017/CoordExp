# S2 Observed-Alias Handbag Continuation Rescue Result

## Decision

`SCIENTIFIC_STOP_OBSERVED_ALIAS_HANDBAG_CONTINUATION`

The one-update run is mechanically valid, but it does not learn
`umbrella -> handbag`.  It restores the handbag at the umbrella's former
emission slot, removes the umbrella, and diverges before reaching the exact
supervised prefix.  This is same-slot reversal, not continuation.

Authoritative reduction:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-s2-observed-alias-handbag-continuation-rescue/analysis-v1.json`

SHA-256:
`37e9d3bd101276ec2c555c35733ada67295e33786a8b4a4bc039cc753bb00358`.
Deterministic replay reproduced it byte for byte.

## Materialization and training

The StateBank contains two identity-equivalent copies of one annotated event:
the canonical handbag after the exact 38-token S2 history through the natural
umbrella alias.

- StateBank ID:
  `7370ba9986776cc37eb5c6ef79134659bc94dd4e51095f04e0a39468c6053260`;
- manifest SHA-256:
  `8f3cf7299365ad47cc1cab4313e87a5405b20af7c1196e0d67dacf7a675398f7`;
- records SHA-256:
  `bb4fbaedf6def0c455e4c935946feca29a6dcfe1836cd57ae55110ada4d6a3ce`;
- receipt SHA-256:
  `d4c801d6ff69f295d8356b4981a4409c0793fad9b8ecb3e49b208d4e2df9e2be`;
- S2 prefix SHA-256:
  `8fc19eb788679a4edb5895590a34ab9328678720d46904961bb8a557182a7814`;
- canonical handbag-row SHA-256:
  `fa874f5b7ae73e83615a98bb9e714424b993f2c990fac8a17bc7f90ea8f99515`;
- forbidden word/geometry hybrid SHA-256:
  `2eb19cac85988384369042caee00166037dd2809b5d9df6bbaa64545c69e5186`;
- full sequence length: 1,410/12,000 tokens.

CPU rematerialization reproduced manifest and records byte for byte; the
receipt differed only by its intentionally recorded output root when replayed
in a temporary directory.

The canonical world-size-two BF16 run consumed one microstep per rank and
applied exactly one AdamW update at LR `2.5e-6`:

| total loss | annotated-row loss | site gate | grad norm |
|---:|---:|---:|---:|
| 1.494127 | 1.494126 | 0.000001 | 11.318727 |

- run receipt SHA-256:
  `5c4fc47e6e539e3a637bd908ce01b4306195bf33b75d0f817e4b606f1e09951b`;
- resolved-config fingerprint:
  `f95fe7348acab32bef666e6e19472beed15dd695a3de3b82a986095832bc1aca`;
- proposal adapter fingerprint:
  `42465d40eab2b28efce782af0d8bb2f0ba8ccfa32713b8c3dd43ac6a170b40c4`;
- 588 language-DoRA tensors and 18,006,016 trainable scalars;
- frozen selected-token embedding fingerprint:
  `635ec008a79fd2657c2acc75772a52cfda70eb0f664ef4f91c4f05c0aa931bc6`.

The checkpoint is unmerged DoRA plus an exact identity-copy of the already
required frozen embedding delta.  No merged checkpoint or base-weight export
was created.

## Cold natural behavior

The proposal was loaded in a fresh process with exact saved-to-materialized
adapter equality, `merged_adapters=[]`, HF greedy decoding, RP1, and the same
twelve-image panel.

| read | IoU50 | IoU60 | IoU80 | selected IoU50 | protected | predictions | duplicate candidates | parser drops | natural EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 137 | 127 | 77 | 0/11 | 3/3 | 204 | 1 | 1 | 12/12 |
| S2 source | 140 | 129 | 78 | 2/11 | 2/3 | 209 | 2 | 1 | 12/12 |
| proposal | 140 | 131 | 80 | 1/11 | 3/3 | 209 | 2 | 1 | 12/12 |

There are no invalid, malformed, or capped rows.  Ordering-monitor counts are
unchanged from S2: 14 violations over six rows.

Relative to S2, the proposal gains the handbag and one dense-car owner but
loses the umbrella and another dense-car owner.  Relative to C, it gains the
book, three car owners, and loses one car owner.  Thus its unchanged total of
140 hides owner exchange.

The lost C owner is on image `337111`, where many small, fence-obscured cars
make exact localization visually weak.  This is a diagnostic, not a retroactive
gate relaxation.  More importantly, it cannot change the decision: the trained
image itself loses the required umbrella.

## Exact transition result

On image `359310`:

- S2 matches umbrella owner `coco_ann:1426509` at generated order 3;
- the proposal instead matches handbag owner `coco_ann:1172698` at generated
  order 3;
- the proposal and supervised prefix agree for only 29 of 38 tokens;
- their first differing token is the first description token:
  expected umbrella `3551`, observed handbag `10661`;
- no matched umbrella precedes the handbag, so neither the behavioral
  continuation gate nor exact-prefix attribution passes.

The proposal does retain the untouched book and restores all three protected
owners.  It also keeps total IoU50 at 140.  Those positives do not rescue the
registered question because the putative conditioning state disappears.

## Interpretation

Observation: CE applied only to `handbag | p, umbrella` changed the earlier
decision at `p` enough to choose handbag instead of umbrella.

Inference: globally shared DoRA couples neighboring autoregressive states.
Teacher-forcing a desired continuation does not ensure that natural decoding
will reach that conditioning state; it can increase the same row's logits at
an earlier, competing slot instead.

This supports the same-prefix competition account and falsifies the narrower
hypothesis that one observed-alias continuation target alone is sufficient.
It does **not** prove that sequence supervision, explicit preservation, or a
joint owner-set objective cannot work.  The two objects are small and visually
ambiguous, so this single image must not select a universal architecture.

Per the frozen stop rule, do not add dose, sweep learning rate, expand aliases,
or implement the eleven-event paired profile.  The next program-level design
should step back from hard per-owner slots and ask how all annotated owners can
receive joint positive credit while unmatched predictions remain unknown.

## Claim boundary

This is a mechanically valid, same-panel, single-image mechanism negative from
one finite shared-DoRA update.  It establishes neither held-out transfer nor a
COCO-scale result, and it makes no production or architecture claim.
