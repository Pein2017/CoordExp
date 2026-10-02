# Dependency and knowledge boundaries

```text
entrypoints -> training / inference / evaluation / visualization
            -> config / data / templates / model and adapter boundaries
            -> packing / supervision / losses / runtime / artifacts
research operators -> reusable core mechanisms, never the reverse
```

This is a dependency sketch, not a maintained list of functions. Resolve actual
modules and callers through source and exact-checkout CodeGraph. A backend or
config option mentioned by a sibling branch is not thereby implemented here.

Keep mechanism separate from scientific policy. Core components own validation,
alignment, composition and publication; callers own cohorts, conditioning,
intervention, denominators, budgets and stopping rules. Model-state, source and
dataset identities are separate. Clean source is necessary for some execution
claims but is not a guarantee of reproducible model numerics.

Executable contracts live in local source/tests and stable OpenSpecs. Scientific
claims live at their research evidence owner. [Documentation](RETENTION.md) keeps
lasting reasons and distinctions, not duplicated implementation detail.
