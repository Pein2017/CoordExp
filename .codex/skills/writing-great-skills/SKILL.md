---
name: writing-great-skills
description: Use only when the user explicitly invokes $writing-great-skills to design or revise a Codex skill around clear trigger boundaries, judgment-aware instructions, expressive interfaces, progressive disclosure, and fresh-context validation.
---

# Writing Great Skills

Build the smallest durable interface that helps a capable model make the right
task-specific judgments. Optimize for reliable outcomes and clear boundaries,
not an identical process on every run.

## Design

1. **Bound one job.**
   - Put the positive trigger and meaningful near-miss boundary in the
     frontmatter description; the body is loaded only after selection.
   - Keep implicit invocation when autonomous discovery earns its catalog cost.
     Set `policy.allow_implicit_invocation: false` when deliberate user choice
     matters more.

2. **Choose the degree of freedom.**
   - Use judgment-oriented instructions when several approaches can work.
   - Use expressive parameters, schemas, or pseudocode when a preferred
     interface exists but context should choose the implementation.
   - Use a strict script or sequence only for fragile, deterministic, or
     safety-critical operations.
   - Keep a hard constraint only when it protects a high-impact boundary or
     addresses an observed recurring failure. State the desired behavior
     directly whenever possible.

3. **Design the interface before adding examples.**
   - Make inputs, selectable parameters, outputs, side effects, completion, and
     failure states inspectable.
   - Prefer code, schemas, tests, and expressive tool descriptions over a list
     of demonstrations. Keep an example only when it reveals a non-obvious
     contract that the interface cannot express more directly.

4. **Disclose context progressively.**
   - Keep the common decision path in `SKILL.md`.
   - Put branch-specific knowledge in a directly linked reference whose pointer
     says when to read it.
   - Bundle reusable scripts inside the skill, resolve their paths relative to
     the skill directory, and test them by execution.
   - Remove unlinked resources and references to repository paths unless the
     skill explicitly owns that repository interface.

5. **Prune against the model's current default.**
   - Remove explanations of general capabilities, repeated policy, memorized
     path lists, and sentences that do not change behavior.
   - Keep each meaning in one owner. Prefer revising that owner over adding a
     second reminder elsewhere.

## Validate

- Parse the frontmatter and `agents/openai.yaml`; confirm the skill name matches
  its directory, required fields exist, and the description states the intended
  trigger boundary. Run the platform skill validator when available.
- Exercise bundled scripts on a representative input.
- Forward-test one positive trigger and one plausible near-miss in fresh
  context without leaking the expected answer.
- Revise only when observed behavior, a real consumer, or a high-impact boundary
  justifies more context.

The skill is complete when its description routes correctly, every instruction
earns its load, every bundled resource has an entrypoint, and the claimed
behavior has proportionate evidence.
