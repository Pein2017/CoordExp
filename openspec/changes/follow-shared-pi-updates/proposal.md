## Why

Pi CLI and Web already share an official managed release, but the installation is mixed into the agent profile and Web only binds it at startup. `pi update` changes the selected release without replacing Web's loaded SDK. The user wants one local runtime owner, automatic downstream adoption, and indexed official Pi and maintained Web source for reference/development.

## What Changes

- Place the unmodified official Pi source checkout at `codex-tools/pi-core`, pinned to the installed release for reference. Its ignored `.local` directory owns the official managed runtime and launcher.
- Keep shared profile, credentials, sessions and local compaction package selection in `.pi`; point CLI and Web at the canonical installation explicitly.
- Have local Next dev Web observe the official selected version and restart its worker only after sessions, starts, commands and terminal shells are idle. Close new work admission before disposing sessions and changing the binding.
- Initialize independent CodeGraph projects inside the active `codex-tools` repositories, including Pi core, Web and local extensions; remove the Pi-only parent index and retain independent histories.
- Document core versus extension update semantics, loaded-process boundaries, source-reference refresh, and local restart diagnostics.

## Capabilities

### New Capabilities
- `shared-pi-updates`: one local managed Pi installation with idle Web adoption and indexed source navigation.

### Modified Capabilities
- None.

## Impact

Changes the local CLI wrapper, Web managed-release binder, Web session/terminal admission and local launcher, plus runtime/source navigation documentation. Moves the existing managed installation without changing release bytes or official core source. No paid model requests, research changes, publication, or core fork. Existing live CLI processes retain their loaded version until reopened; future SDK API conflicts remain explicit adaptation failures.
