# `research-base-v1` pre-tag gate record

Captured on 2026-08-24 before the evidence-record commit. This is a timed
gate observation, not an evergreen claim about a moving branch. The final tag
candidate must receive its own exact-candidate review; the tag does not exist
and is not created by this record.

## Scope and source identity

- Revalidated post-infra anchor:
  `f337de5d0bd016b79aa012acfc491544e6313333` with recorded identity
  fingerprint `abe39a84025bc08e0a6249fe5415f5688d6a25e982dad58082bbbfddc028ded8`.
- Gate observation candidate:
  `c6ffc9bc9b45f76749890c48103070a9b0f426f9`; it was clean and descends from
  the anchor.
- `research-base-v1` was absent. `probe-final/*` had no refs, so there were no
  lifecycle tips to resolve.
- Independent concurrent worktrees are out of scope and were not consulted.

## Fixed-worktree capture

| Fixed path | Resolved Git admin directory | Resolved named ref | Gate-time commit | Native lock reason |
| --- | --- | --- | --- | --- |
| `/data/CoordExp/.worktrees/research-probes` | `/data/CoordExp/.git/worktrees/coordexp-infrastructure` | `refs/heads/research-probes` | `c6ffc9bc9b45f76749890c48103070a9b0f426f9` | `Canonical research authority; unlock only by explicit lifecycle decision` |
| `/data/CoordExp/.worktrees/research-probe-infras` | `/data/CoordExp/.git/worktrees/research-probe-infras` | `refs/heads/codex/research-probe-infra-foundation` | `baaa01e978a2a2b3c8cf7317564731d2b578c371` | `Bounded research infrastructure lane; unlock only by explicit lifecycle decision` |

Each row's named ref resolved to its recorded commit at capture time. The
separately held cache-validation checkout remains detached at
`477b376a3e31a5dbedf5a87ecafcb372e75a73a9`; no ref, prune, removal, or other
lifecycle action is authorized here.

## External-locator and verification record

This baseline change cites no external-artifact locator as an authority input:
its imported infra evidence is the Git anchor and identity fingerprint above.
Accordingly, the locator inventory is empty; no liveness or checksum result is
being omitted. CPU-only target-binding checks remain mechanics evidence only
and do not create a GPU claim.

The following commands succeeded at capture time:

```text
git status --short                              # empty
git merge-base --is-ancestor f337de5 HEAD       # exit 0
openspec validate harden-research-probe-target-binding --strict
openspec validate establish-research-probes-baseline-v1 --strict
pytest -q -p no:cacheprovider \
  tests/artifacts/test_research_probe_admission.py \
  tests/research/test_research_probe_admission_consumers.py  # 41 passed
```

The support and crossover target-tree captures both revalidated against the
clean `research-probes` gate candidate. Their immutable fingerprints were,
respectively,
`56e42c2d209014b291d7c8e7c68ce5ba09344b1382934137cf3b1075199dcfe8` (31
effective inputs) and
`5ecc59c021bdaaf2dbb26f4e999d38b2ad24ab205a99f8c8029a1b8edc0aaa78` (27
effective inputs).
