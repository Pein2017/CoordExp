# Deprecated

This completed change is superseded by `unify-inference-runtime`.

Active unified Stage-2 rollout-correction server training uses
`rollout_matching.vllm.mode=server`,
`rollout_matching.vllm.sync.mode=adapter`, and
`rollout_matching.vllm.enable_lora=true`, with coord-row updates handled by the
shared inference backend.

Archive this change with `--skip-specs`; do not sync its older Stage-2 AB
full-sync materialization requirements into stable specs unless a future
OpenSpec explicitly revives native full-sync as an active contract.
