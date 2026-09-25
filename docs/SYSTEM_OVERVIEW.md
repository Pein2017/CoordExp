# Current architecture

The maintained core is a one-way dependency stack:

```text
src.train / src.infer / saved eval and visualization commands
       -> training / inference / eval / visualization
       -> config / data / templates / qwen / adapters
       -> packing / losses / supervision / runtime / artifacts

probes.output_qp, probes.readout_norm -> explicit numerical inputs + artifacts
research index -> questions/story/assets -> historical catalog -> Git/artifacts
```

`src` never depends on research profiles. Ordinary functions own narrow mechanics;
the caller retains populations, denominators, prompt policy, scientific stopping
and interpretation. Model-state evidence and dataset identity are separate.
The reused saved-row evaluator does not promote IoU matches to physical truth.

Training uses the current config/Accelerate/packing interfaces; inference selects
its declared HF/vLLM backend and preserves actual likelihood/trace semantics.
Batch, precision, image geometry and cache semantics are not refactored away.
`openspec/specs` holds their detailed stable requirements and `tests` their
executable contract coverage.

`src.artifacts.git_identity` binds exact clean commit/tree/source bytes. Training
resume schema2 reads a JSON source gate and verifies payload bytes before loading
training state. Old envelopes cannot resume by filename or an updated hash.
Clean source is necessary, not sufficient for reproducible model numerics.

History detail is recovered through catalogued Git commits. Raw external outputs,
model tensors, data and current run evidence are not absorbed into the codebase.
