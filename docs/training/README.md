# Current training

Entry: `python -m src.train --config <current-yaml>`.
`src.training` owns the supervised loop, `src.runtime` the supported Accelerate
mechanics, `src.config` resolved configuration and `src.losses` the objectives.
[Core contracts](../coordexp_infras.md) and OpenSpec specs define detailed behavior.

A clean Git source qualification is required before decision-bearing execution.
Resume requires current source gate schema2 in addition to unchanged compatible
config, schedule, data and optimizer state. Missing legacy source rejects before
model initialization. [Artifacts](../ARTIFACTS.md) explains the JSON sidecar and
its trust boundary. This is not an exact arbitrary-GPU replay guarantee.

DoRA, geometry/CE losses and packing remain because current training consumes
them, not because old experiments need indefinite maintenance. Historical
training recipes are recovered from catalogued Git; no old Stage-2 runner is
implicitly supported. A model run still requires explicit resource authority.
