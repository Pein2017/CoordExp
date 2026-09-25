# Current DoRA qualification contract

The public V1 adapter schema uses `adapter.type: dora`; the PEFT mechanism is
`LoraConfig(use_dora=True)`. Language/vision/aligner selection and excluded lm_head
remain the current explicit adapter configuration, not implicit research defaults.

The historical source study records: Wave 1B probe evidence completes task 2.3.
Its full rationale and execution details are recoverable at Git
`108dede0154abfd90a54d18234d9e0bac780a3ba`, original source-study document under the
June 27 architecture proposal. This condensed current contract is not a new model
run, a general numerical replay certificate, or continuation authority.

`src.adapters.source_gates` separately requires the external roundtrip evidence
at `outputs/probes/coordexp_swift/dora_roundtrip/receipt.json`. Actual saved A/B and
magnitude-vector counts, reload/logit equivalence and finite gradients remain
checked. Missing probe evidence fails; documentation text alone cannot qualify an
adapter. That external run directory was not copied or modified by cleanup.
