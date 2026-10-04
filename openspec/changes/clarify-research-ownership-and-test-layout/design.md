# Ownership decision

src owns reusable execution and integrity mechanics. probes owns concrete
research-direction methods, not only the two oldest retained operators and not a
universal experiment registry. tests/probes owns their executable contracts.
research/questions owns distilled arguments; research/experiments owns bounded
units, evidence and conclusions. docs admits stable cross-cutting knowledge,
not copies of module APIs or the latest experimental state.

## Evidence and protection

Baseline HEAD: ca7ec3c28506304c1f58c82094b9aac51bd7f303.
The three relocated modules use explicit probes imports, no path-relative fixture
loads or relative test imports. A current-source search found no probes/tests
caller or source-receipt path. Historical records are not rewritten. Baseline
operator plus knowledge tests: 41 passed, CUDA hidden and offline.

Rename without changing bytes:
- probes/tests/test_output_qp.py -> tests/probes/test_output_qp.py
- probes/tests/test_readout_norm.py -> tests/probes/test_readout_norm.py
- probes/tests/test_runtime_compat.py -> tests/probes/test_runtime_compat.py

The old broad probes test root can discover ignored retired directories. Only the
current source-owned test root is selected; no ignored file is deleted. Numerical
certificate, unselected-column identity, input safety and catalog recovery tests
remain. No backwards import shim or second documentation owner is introduced.

## Validation

Compare renamed bytes and the same 41 baseline test cases, then run the wider
retained probes/knowledge/integrity contracts. Validate current knowledge and
historical recovery through the existing checker, OpenSpec metadata, exact diff,
unchanged implementation files and concurrent work. This is structural/CPU
validation, not numerical model qualification or scientific acceptance.
