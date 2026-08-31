# Dyadic norm-release distillation implementation plan

The execution reused the frozen Image2299 augmented-r32 route and its
production parser/matcher. It captured the 12 positive states, protected 664
states, solved one 108-variable constrained release, materialized the nine
selected FP64 residual rows, and tested one warm candidate followed by cold
reload. No new evaluator, model surface, optimizer, or training schedule was
introduced.

## Literal execution checklist

- [x] One solve, 108/108 feasible; minimum norm
      `1.0972734315870298 < 9/8`.
- [x] Full-vocabulary runtime recheck passed `12/12` states.
- [x] Two model loads, one solve, one warm candidate; wall time `840.67 s`.
- [x] Warm and cold route SHA exactly
      `c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e`,
      length 370, with exact selected IDs and payload parity.
- [x] Warm/cold frozen surfaces, protected null, gate, route, ledger, basis,
      payload, and successor-schema checks all true.
- [x] Nine FP64 residual rows of shape `9 x 2048`, payload SHA
      `10455e0587bfd103bf3c539cdc655f21b1ee778a02ef27be5f80a6e9f8c34704`,
      residual-row SHA
      `f85c0022306c426d5cc752d9ac383ea313c1af6fc737f42bee058b8891b1b1ec`.
- [x] Selected target IDs are exactly
      `[151645,151646,151820,151867,151935,152032,152190,152242,152305]`.
- [x] Cold natural greedy: 38 persons, 3 ties, 41 strict owners; eight exact
      gained persons; zero debt/counters; parser-accepted natural EOS.

The authoritative receipt and final interpretation are in [results.md](results.md).
