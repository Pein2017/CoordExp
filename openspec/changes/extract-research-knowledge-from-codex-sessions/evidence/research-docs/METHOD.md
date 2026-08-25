# Research-document census method

Capture: `2026-08-25` UTC, live filesystem. Product: a manifest and bounded
synthesis of Markdown research knowledge. No source research, code, config,
Notion page, experiment, or worktree was modified.

## Frozen and live boundary

- Root: `/data/CoordExp/research/**/*.md` (51 paths).
- Registered worktrees: `git worktree list --porcelain`, restricted to
  `/data/CoordExp/.worktrees/*`; for each worktree, every regular Markdown file
  whose path contains `/research/` (9,695 live paths).
- Combined manifest rows: 9,746. The contract snapshot states 9,694 worktree
  paths; the current census is one path above that snapshot. The extra path is
  retained, not discarded. Nested `docs/history/**/research/**` and
  `.pi-worker/**/source/research/**` are included because the declared broad
  discovery glob reaches them; they are marked historical/duplicate/skip.
- Canonical research authority remains the locked
  `/data/CoordExp/.worktrees/research-probes`; locked
  `/data/CoordExp/.worktrees/research-probe-infras` owns reusable mechanics;
  `/data/CoordExp/.worktrees/image2299-mechanism-microscope` is a dirty,
  uncommitted candidate surface.

## Exact discovery and identity commands

The mechanical enumeration was equivalent to:

```bash
find /data/CoordExp/research -type f -name '*.md'
find /data/CoordExp/.worktrees -path '*/research/*' -type f -name '*.md'
git worktree list --porcelain
git -C <each-worktree> status --porcelain
sha256sum <each-discovered-path>
```

The deterministic TSV generator walked the two roots with `os.walk`, sorted
absolute paths, SHA-256 hashed bytes, assigned one `sha256:<digest>` duplicate
key per content group, selected representatives in this order (root,
`research-probes`, image2299 candidate, `research-probe-infras`, other direct
research, nested provenance), and wrote `manifest.tsv`. It also grouped same
research-relative paths (suffix after the first `/research/`) to expose byte
divergence before deduplication. A dirty candidate unit changed during the
capture window; its manifest row was refreshed to the final observed SHA-256
and then rechecked. This is why the candidate status and content identity stay
visible rather than being treated as a clean checkout.

## Worktree capture

The repository root was `main` at `b274d596ba3ce98b4f5bb64b0247a773637f5d9c`
with 54 pre-existing dirty entries. The 31 registered `.worktrees` were
captured with path, HEAD, branch/detached state, and porcelain-entry count:

| worktree | HEAD | branch | dirty entries |
|---|---|---|---:|
| `CoordExp-swift` | `22f2fc9e0` | `coordexp-swift` | 0 |
| `codex-rtk-correctness-first` | `38b30ebc1` | `codex/rtk-correctness-first` | 0 |
| `codex-wake-me-up-event-monitor` | `8dfb8102a` | `codex/wake-me-up-event-monitor` | 0 |
| `coverage-ledger-mechanistic-probing` | `77acee47c` | `codex/coverage-ledger-mechanistic-probing` | 0 |
| `geometry-aware-denoising-sft` | `b3919b49f` | `codex/prefix-denoising-sft` | 0 |
| `human13-analyzer` | `68fafe2fb` | `codex/human13-analyzer` | 0 |
| `human13-discovery-adapter` | `4d3d01800` | `codex/human13-discovery-adapter` | 0 |
| `human13-live-model` | `a6bfb1c94` | `codex/human13-live-model` | 0 |
| `human13-loss-census` | `a4cb7e89e` | `codex/human13-loss-census` | 0 |
| `human13-manifest-collector` | `10c37852e` | `codex/human13-manifest-collector` | 0 |
| `human13-materializer-launcher` | `fd2a7542b` | `codex/human13-materializer-launcher` | 0 |
| `human13-nk-factorial-probe` | `a904e3ae3` | `codex/human13-nk-factorial-probe` | 0 |
| `human13-runner` | `832dd63f8` | `codex/human13-runner` | 0 |
| `image2299-mechanism-microscope` | `463092bea` | `codex/image2299-mechanism-microscope` | 21 |
| `ledger-auxiliary-loss` | `91cddea1a` | `codex/ledger-auxiliary-loss` | 0 |
| `owner-commit-binding` | `210ad8c0c` | `codex/owner-commit-binding` | 0 |
| `permanent-owner-bridge` | `84fc6b318` | `codex/permanent-owner-bridge` | 0 |
| `permanent-owner-bridge-cache-validation` | `477b376a3` | detached | 1 |
| `permutation-bundle-coordinate-noise-pilot` | `9ddbe4e91` | `codex/permutation-bundle-coordinate-noise-pilot` | 0 |
| `regionlock-simplified-pointer` | `444cd7ba3` | `codex/regionlock-simplified-pointer` | 0 |
| `research-probe-infras` | `74609d2b1` | `codex/research-probe-infra-foundation` | 0 |
| `research-probes` | `f775ff2c3` | `research-probes` | 0 |
| `rp-crossover-analyzer` | `b796b5ebb` | `codex/rp-crossover-analyzer` | 0 |
| `rp-crossover-integration` | `efc57dc11` | `codex/rp-crossover-integration` | 0 |
| `rp-crossover-launcher` | `78d0069d0` | `codex/rp-crossover-launcher` | 0 |
| `rp-crossover-live-integration` | `7cb17a832` | `codex/rp-crossover-live-integration` | 0 |
| `rp-crossover-materializer` | `dbc36730c` | `codex/rp-crossover-materializer` | 0 |
| `rp-crossover-production` | `e7e373037` | `codex/rp-crossover-production` | 0 |
| `rp-crossover-runtime` | `2c5632e10` | `codex/rp-crossover-runtime` | 0 |
| `rp-crossover-wave5-correction` | `51d518f75` | `codex/rp-crossover-wave5-correction` | 0 |
| `vllm-mechanistic-round-gaussian-rps` | `cd9c92211` | `codex/vllm-mechanistic-round-gaussian-rps` | 0 |

## Verification and sampling

The manifest has exactly 9,746 data rows, all source paths exist, all recorded
SHA-256 values recompute, and its header exactly matches `CONTRACT.md`.
Content grouping yields 653 unique byte contents, 398 duplicate groups, and
9,093 duplicate path instances. Research-relative grouping yields 599 keys,
30 divergent same-path groups, and 979 path instances in those groups. The
relevance/lifecycle assignment is a triage field, not scientific authority;
the synthesis below is the bounded reading path.

Read samples included more than 10 high/medium rows and more than 10 skips,
including root routers, locked canonical decisions/results, image2299 current
candidate results, Human13/coverage/owner-bridge branch surfaces, and nested
history/worker snapshots. Current routers, all canonical decision nodes, the
latest image2299 results, and representative negative/invalid/partial records
were deep-read. Byte-identical copies were not reread as independent findings;
their paths remain in the manifest.

Sample receipt (all paths were present and hash-checked):

- High: `research-probes/.../qwen3-vl-dense-enumeration/compass.md`,
  `image2299-mechanism-microscope/.../2026-08-24-image2299-near-policy-prefix-dose-response/results.md`,
  `image2299-mechanism-microscope/.../2026-08-24-image2299-second-row-prefix-spatial-counterfactual/results.md`,
  `image2299-mechanism-microscope/.../2026-08-25-image2299-natural-and-gt-prefix-free-decode/results.md`,
  `image2299-mechanism-microscope/.../2026-08-25-image2299-xy-same-owner-serialization-sentinel/results.md`,
  `image2299-mechanism-microscope/.../2026-08-25-image2299-safe-successor-rectangle-guard-training-vertical/results.md`.
- Medium: `coverage-ledger-mechanistic-probing/.../ledger-auxiliary-loss/overview.md`,
  `owner-commit-binding/.../qwen3-vl-painted-gt-transcription-probe/index.md`,
  `permanent-owner-bridge/.../2026-08-11-permanent-owner-bridge-recovery-successor/results.md`,
  `research-probe-infras/.../2026-08-06-natural-boundary-routing-history-replication/results.md`,
  `vllm-mechanistic-round-gaussian-rps/.../gaussian-rps-mechanistic-round/2026-07-01_autoregressive_binding_prior_findings_synthesis.md`,
  `human13-live-model/.../research/investigations/qwen3-vl-dense-enumeration/index.md`.
- Skip: direct branch routers under `CoordExp-swift/research/`, and nested
  `docs/history/worktree-cleanup/**/snapshots/**/research/**` and
  `.pi-worker/**/source/research/**` copies. Ten-plus skip rows were checked;
  their reasons and duplicate keys are retained per row in the manifest.
