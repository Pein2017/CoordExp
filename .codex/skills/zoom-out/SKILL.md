---
name: zoom-out
description: Use when the user asks for broader context, a module map, caller/callee orientation, or how an unfamiliar CoordExp area fits into the stack.
---

# Zoom Out

Give a higher-level map before diving back into details.

## CoordExp Read Order

Start from the current task and follow this repo's routing rules:

- `docs/AGENT_INDEX.md` and `docs/catalog.yaml` for entrypoint routing;
- `docs/PROJECT_CONTEXT.md`, `docs/SYSTEM_OVERVIEW.md`, and `docs/IMPLEMENTATION_MAP.md` for current architecture;
- relevant domain docs under `docs/`;
- `openspec/specs/` only for stable compatibility contracts;
- `progress/` only for historical evidence, diagnostics, or empirical context.

## Mapping Style

- Name the major modules, configs, artifacts, and call paths.
- For Python code, use Serena MCP for symbol-aware exploration after narrowing candidates with `rg` or `rtk`.
- Separate current stable behavior from legacy paths, active experiments, and historical notes.
- End with the smallest next investigation or implementation step.
