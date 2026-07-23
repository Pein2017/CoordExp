import assert from "node:assert/strict";

import { rewriteBashCommand } from "./extensions/codex-rtk-hook-adapter.ts";

assert.equal(
  await rewriteBashCommand("pytest -q tests"),
  "rtk pytest -q tests",
  "ordinary noisy commands should use the same RTK rewrite as Codex",
);

assert.equal(
  await rewriteBashCommand("git status --porcelain"),
  "git status --porcelain",
  "machine-readable commands must remain byte-compatible",
);

assert.equal(
  await rewriteBashCommand("RTK_HOOK_DISABLE=1 pytest -q tests"),
  "RTK_HOOK_DISABLE=1 pytest -q tests",
  "the existing command-level disable contract must remain effective",
);

const previousGlobalDisable = process.env.RTK_HOOK_DISABLE;
process.env.RTK_HOOK_DISABLE = "1";
assert.equal(
  await rewriteBashCommand("pytest -q tests"),
  "pytest -q tests",
  "the existing process-level disable contract must remain effective",
);
if (previousGlobalDisable === undefined) {
  delete process.env.RTK_HOOK_DISABLE;
} else {
  process.env.RTK_HOOK_DISABLE = previousGlobalDisable;
}

assert.equal(
  await rewriteBashCommand("pytest -q tests", async () => ""),
  "pytest -q tests",
  "empty hook output means no rewrite",
);

assert.equal(
  await rewriteBashCommand("pytest -q tests", async () => "not-json"),
  "pytest -q tests",
  "malformed hook output must fail open",
);

assert.equal(
  await rewriteBashCommand("pytest -q tests", async () => {
    throw new Error("simulated hook failure");
  }),
  "pytest -q tests",
  "hook execution errors must fail open",
);

assert.equal(
  await rewriteBashCommand(
    "pytest -q tests",
    async () =>
      JSON.stringify({
        hookSpecificOutput: {
          updatedInput: { command: "rtk pytest -q tests" },
        },
      }),
  ),
  "rtk pytest -q tests",
  "valid Codex updatedInput output must become the Pi bash command",
);

console.log("codex_rtk_hook_adapter_ok");
