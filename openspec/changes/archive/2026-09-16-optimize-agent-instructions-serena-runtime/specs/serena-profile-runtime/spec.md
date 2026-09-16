## Purpose

Keep the DSH Web Serena integration predictable and portable by removing ineffective hook overhead and resolving profile-owned paths from runtime configuration rather than one machine's absolute layout.

## ADDED Requirements

### Requirement: The Web profile has one effective reminder owner

The DSH Web profile SHALL NOT install the Serena reminder hook by default when the hook runner is unavailable or duplicates the broader repeat-tool guard. The existing standalone Codex hook configuration SHALL remain independent of this profile choice.

#### Scenario: DSH Web starts with the optimized profile

- **WHEN** the Web profile is loaded
- **THEN** it SHALL mount the Serena MCP server and SHALL NOT register the profile-local Serena reminder hook

#### Scenario: The generic repeat guard remains available

- **WHEN** repeated tool calls reach the configured repeat thresholds
- **THEN** the existing generic repeat-tool guard SHALL remain eligible to provide its reminder behavior

### Requirement: Profile-owned paths are runtime-resolved

The Web profile SHALL resolve DSH-owned files from the active DSH home and SHALL allow the Serena executable and Serena state directory to be overridden by environment configuration. The default state location SHALL remain relative to the launch workspace's `.codex/serena` directory.

#### Scenario: DSH home moves

- **WHEN** the profile runs with a different configured DSH home
- **THEN** DSH-owned profile paths SHALL resolve under that active home without requiring an edit to the profile file

#### Scenario: Serena executable is overridden

- **WHEN** the Serena executable override is set
- **THEN** the profile SHALL launch that executable instead of a machine-specific fixed path

#### Scenario: Serena state override is absent

- **WHEN** no Serena state override is set
- **THEN** the profile SHALL use the launch workspace's `.codex/serena` state directory
