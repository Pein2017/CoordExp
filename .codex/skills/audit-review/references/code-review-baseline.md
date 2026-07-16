# Code Review Baseline

Use this only for `diff/code review` mode. Repository standards override these
heuristics. Treat generic smells as judgment calls, not hard violations, and
skip formatting or lint issues already enforced by tooling.

## Engineering Smells

- **Mysterious name**: a changed name hides the concept or layer it owns.
- **Duplicated code**: the same nontrivial behavior is introduced twice.
- **Feature envy**: logic reaches deeply into another module's state instead of
  living with the concept it owns.
- **Data clump**: the same fields travel together and want a cohesive type.
- **Primitive obsession**: strings or loose dictionaries stand in for a stable
  domain or research contract.
- **Repeated switch**: the same mode dispatch is copied across owners.
- **Shotgun surgery**: one semantic change requires scattered edits because
  ownership is diffuse.
- **Divergent change**: one module changes for unrelated reasons.
- **Speculative generality**: abstractions, hooks, or knobs serve no current
  requirement or verified second use.
- **Message chain**: callers navigate internal structure that an interface
  should hide.
- **Middle man**: a module delegates without adding leverage or protecting a
  contract.
- **Refused inheritance**: an implementation accepts an interface or base class
  it cannot honestly satisfy.

## CoordExp Review Smells

- **Research-semantic hiding**: a refactor changes forward behavior, targets,
  loss normalization, sampling, geometry, metric meaning, or statistical scope
  behind an apparently structural edit.
- **Contract diffusion**: config values, token roles, metric keys, artifact
  names, or geometry rules gain a second owner.
- **Canonical/diagnostic mixing**: salvage or compatibility behavior can affect
  canonical training or evaluation claims.
- **Silent fallback**: unsupported or obsolete input becomes plausible output
  instead of failing fast.
- **Comparability drift**: a metric, dataset slice, parser, bbox surface, or
  checkpoint scope changes without an explicit claim boundary.
- **Test-seam mismatch**: tests exercise helpers while the real failure lives in
  orchestration, integration, distributed behavior, or artifact materialization.
- **Unrequested control surface**: a new knob transfers an implementation choice
  to the user instead of making the implementation correct.
- **Artifact amnesia**: behavior changes without enough resolved config,
  manifest, metric, or provenance evidence to reproduce and interpret it.

## Complexity And Proportionality

- Search the current repository owner, native platform or standard library,
  and installed dependencies before approving a new abstraction. A shorter
  replacement is acceptable only when it preserves research semantics,
  validation, ordering, compatibility, and provenance.
- Lines of code are evidence of review surface, not an optimization objective.
  Accept complexity that protects a real contract; classify unsupported
  complexity as `narrow`, `drop`, or `probe` rather than demanding cosmetic
  compression.
- Before declaring a definition unused, inspect direct callers, imports,
  re-exports, configuration, reflection, and dynamic loading where plausible.
  Use a narrow runtime probe when static evidence cannot decide the question.
- Distinguish duplicated current authority from preserved provenance. Remove or
  consolidate the former; retain source reviews, audits, and immutable receipts
  while assigning one current semantic owner.

## Axis Discipline

Place each finding on the axis it actually supports:

- **Engineering Standards**: maintainability, ownership, testability, depth,
  clarity, and proportionality.
- **Intent And Contract**: requested behavior, algorithm/data/loss semantics,
  reproducibility, evaluation validity, and artifact/metric contracts.

If one observation affects both axes, write two consequences or explicitly mark
the cross-axis impact. Never use a clean result on one axis to cancel a failure
on the other.
