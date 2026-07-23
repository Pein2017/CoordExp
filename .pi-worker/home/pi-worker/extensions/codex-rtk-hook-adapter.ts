import { spawn } from "node:child_process";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const HOOK_PYTHON = "/root/miniconda3/envs/ms/bin/python";
const HOOK_SCRIPT = "/data/CoordExp/.codex/hooks/rtk-pretooluse.py";
const HOOK_TIMEOUT_MS = 5_000;
const MAX_HOOK_OUTPUT_BYTES = 64 * 1024;

type HookRunner = (payload: unknown, signal?: AbortSignal) => Promise<string>;

export function runCodexHook(payload: unknown, signal?: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(HOOK_PYTHON, [HOOK_SCRIPT], {
      env: process.env,
      signal,
      stdio: ["pipe", "pipe", "ignore"],
      timeout: HOOK_TIMEOUT_MS,
      killSignal: "SIGKILL",
    });

    let stdout = "";
    let settled = false;

    const finish = (callback: () => void) => {
      if (settled) return;
      settled = true;
      callback();
    };

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
      if (Buffer.byteLength(stdout, "utf8") > MAX_HOOK_OUTPUT_BYTES) {
        child.kill("SIGKILL");
        finish(() => reject(new Error("Codex RTK hook output exceeded 64 KiB")));
      }
    });
    child.on("error", (error) => finish(() => reject(error)));
    child.on("close", (code, closeSignal) => {
      if (code === 0) {
        finish(() => resolve(stdout));
        return;
      }
      finish(() =>
        reject(
          new Error(
            `Codex RTK hook exited with code ${String(code)} signal ${String(closeSignal)}`,
          ),
        ),
      );
    });

    child.stdin.end(JSON.stringify(payload));
  });
}

export async function rewriteBashCommand(
  command: string,
  runner: HookRunner = runCodexHook,
  signal?: AbortSignal,
): Promise<string> {
  try {
    const rawOutput = await runner(
      {
        tool_name: "Bash",
        tool_input: { command },
      },
      signal,
    );
    if (!rawOutput.trim()) return command;

    const output = JSON.parse(rawOutput) as {
      hookSpecificOutput?: {
        updatedInput?: { command?: unknown };
      };
    };
    const rewritten = output.hookSpecificOutput?.updatedInput?.command;
    return typeof rewritten === "string" && rewritten.length > 0 ? rewritten : command;
  } catch {
    // Match the token-saving hook's fail-open contract: adapter, subprocess,
    // timeout, or payload errors must never block the real command.
    return command;
  }
}

export default function codexRtkHookAdapter(pi: ExtensionAPI): void {
  pi.on("session_start", (_event, ctx) => {
    ctx.ui.setStatus("codex-rtk-hook", "RTK hook: Codex parity");
  });

  pi.on("tool_call", async (event, ctx) => {
    if (event.toolName !== "bash") return;

    const input = event.input as { command?: unknown };
    if (typeof input.command !== "string" || input.command.length === 0) return;

    input.command = await rewriteBashCommand(input.command, runCodexHook, ctx.signal);
  });
}
