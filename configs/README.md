# Supported configurations

`coordexp_infras/prod` contains maintained production-shaped training profiles;
`coordexp_infras/smoke` contains reusable engineering qualification inputs;
`coordexp_infras/infer` contains the retained inference/test profiles. A small
number of additional current-schema profiles remain actual core-test consumers.
Declared `extends` parents remain part of each configuration closure.

A smoke config's presence is not an authorization to run GPU tests. Paths identify
external data/model roots and must be checked before execution. No YAML exists
solely as a diary of a closed research run. Historical config versions and exact
hyperparameters are recoverable from the research catalog's Git and artifacts.

Config validation rejects unsupported legacy keys; it does not silently migrate
old pipeline schemas. New clean-source qualification does not make an old receipt
continuable. Keep tested scientific values distinct from execution mechanics.
