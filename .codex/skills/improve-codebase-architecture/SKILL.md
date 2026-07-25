---
name: improve-codebase-architecture
description: Review CoordExp architecture read-only, rank evidence-backed deepening opportunities, or develop an approved interface design.
---

# Improve Codebase Architecture

Find **architectural friction** without silently changing research meaning.
Use `codebase-design` for interface judgment; this skill owns discovery,
ranking, and the approval boundary.

## Review

1. **Load current authority.**
   - Follow the repository's current authority index to the relevant owning
     guidance, contracts, configs, tests, artifacts, and source.
   - Use Serena for known Python symbols and raw search for non-code evidence.
   - Complete when the review follows live owners rather than memorized paths.

2. **Locate friction.**
   - Look for caller knowledge spread across modules, multiple owners for one
     policy, orchestration hidden behind shallow helpers, compatibility leaking
     into canonical behavior, or tests crossing internals because no honest seam
     exists.
   - Complete when each candidate has concrete call, config, test, or artifact
     evidence.

3. **Rank before designing.**
   - For each candidate report owner, friction, deepening direction, knowledge
     hidden and retained, semantic risk, benefit, verification, and confidence.
   - Give one top recommendation. Do not invent an interface for every
     candidate.
   - Complete when the user can choose at the level of promises and trade-offs.

4. **Design only the selected candidate.**
   - Use `codebase-design` and compare meaningfully different interfaces when
     the seam is consequential.
   - Pause on choices that alter algorithm, data, objective, statistics,
     metrics, artifacts, compatibility, or material migration cost.
   - Complete at `drop candidate`, `probe architecture assumption`,
     `ready for interface decision`, `ready for implementation approval`, or
     `needs user decision`.

Architecture review is read-only unless the user separately authorizes a durable
record or implementation.

## Report

Return ranked candidates with exact evidence, one recommendation, verification,
residual risk, and the approval state.
