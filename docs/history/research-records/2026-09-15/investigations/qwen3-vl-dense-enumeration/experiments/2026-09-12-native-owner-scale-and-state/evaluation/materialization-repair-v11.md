# Candidate request materialization correction (accepted; execution complete)

Root closeout: this correction received its bounded regrant and completed the
640-row natural endpoint and cold consumer. See [result](result-v11.md).
The validation below preserves the prelaunch evidence, not a pending request.

The consumed v10 invocation failed before any model forward: legacy relative
image references were resolved against the fresh panel JSONL directory, yielding
`/public_data/...` rather than the frozen `/data/CoordExp/public_data/...` source.
The exact executed producer is preserved as `evaluation/executed-producer-v10.py`
under the unit output root (SHA `fb54f7e72d09d4e5bd371188a13afa11c4c9810681d756cb40c40dd342cf8d6a`).

The candidate-only fix verifies frozen absolute image file bytes, deep-copies
the case, and derives an ephemeral relative reference from that absolute path
against the request config JSONL parent. The raw-row contract requires relative
references, so inserting a literal absolute reference was rejected in CPU
testing and is not the implemented fix. Frozen source rows, annotations, prompt,
native media/grid checks, selection, and inference semantics remain unchanged.

Fresh validation: `python -m pytest -q probes/native_owner_scale/tests/test_evaluation.py`
reports **21 passed**. `materialization_check_v11.py` reproduces the original
real-entry missing-image failure, then checks all640 bound request prompt IDs
and image file hashes and the first8 CPU-executed media/grid/prompt projections.
No model loads or forwards. Output receipt: `evaluation/materialization-check-v11.json`.

Continuation packet `evaluation/candidate-panel-bound-v11.json` SHA
`951888372135721d8b1dbd4b02b6ed3a830311d4ccfc1370816838715eeca339` binds producer
`f4161a4f2a512e8ef0dc3d02b2607377889bea7f40341df7e16afb3c2d1e786d` and preserved
snapshot `candidate-producer-v11.py`. Its records are byte-equivalent JSON values
to v10 (digest `50291f812f57c499d482cee0e9e757a1f63b742a63b8f830c38716fe7e1292f7`).
The output target is the new `evaluation/candidate-natural-v11`; v10 is untouched.

Prelaunch status at original submission: correction candidate only, awaiting
root regrant. Superseded by the completed execution linked above; no promotion.
