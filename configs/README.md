# Configs

This checkout's Swift training profiles live in `coordexp_infras/prod/`,
implementation smokes in `coordexp_infras/smoke/`, and inference examples in
`coordexp_infras/infer/`. The selected checkout's typed source and stable
OpenSpec contracts determine whether a profile is currently supported.

`stage1/`, `stage2/`, `_shared/` and `archive/` retain earlier training profiles
or authoring examples. Their historical pipeline IDs do not make them the
current main training route. Explicit legacy reproduction requires qualification
against the corresponding historical source; a retained YAML alone is insufficient.

`analysis/`, `infer/`, `eval/`, `postop/`, `bench/` and `public_data/` contain
tool inputs rather than training profile families. A retained config may outlive
its original tool; use source or CLI discovery before treating it as runnable.

Keep durable production profiles, reusable smokes and decision-bearing ablations
here. Historical results belong to their research owner; output placement follows
[storage ownership](../docs/storage-ownership.md).
