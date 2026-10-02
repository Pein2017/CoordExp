# Maintained command adapters

Saved detection evaluation and visualization are thin entries over `src.eval`
and `src.vis`. Use their `--help` for actual accepted fields; old launch
flags are not compatibility promises.

- `python -m scripts.evaluate_detection --help`
- `python -m scripts.visualize_detection --help`
- `python -m scripts.check_research_knowledge check`

`scripts/probes/coordexp_infras/` retains explicit engineering qualification
commands for adapter/token payloads, attention and inference backends. They are
not routine research runners and must not be executed merely because their
source is present. Their contracts and meaningful core tests remain supported.

Historical training managers, dataset conversion factories, experiment-specific
admission adapters and migration tools have retired. Git recovers old commands;
new commands do not pretend to accept their schemas. [Research operators](../probes/README.md)
use ordinary modules. External outputs, model and data resources retain their
original identities and are never recreated implicitly by an entry adapter.
