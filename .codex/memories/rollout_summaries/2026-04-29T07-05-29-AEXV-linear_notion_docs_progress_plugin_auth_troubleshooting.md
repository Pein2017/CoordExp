thread_id: 019dd80e-9b5b-7083-9204-a9bb8f334c23
updated_at: 2026-04-29T14:58:00+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/29/rollout-2026-04-29T07-05-29-019dd80e-9b5b-7083-9204-a9bb8f334c23.jsonl
cwd: /data/CoordExp
git_branch: main

# Linear/Notion vs repo docs/progress, plus Linear plugin auth troubleshooting

Rollout context: The user was exploring whether to use `Notion` or `Linear` alongside the repo’s existing `docs/` and `progress/` folders, then shifted to whether ChatGPT Web could access those systems, and finally asked to install/login to the `linear` plugin and create a toy Linear doc for a ChatGPT Web access test. The work happened in `/data/CoordExp` and used the repo-local `.codex` configuration.

## Task 1: Compare Notion vs `docs/` and `progress/`

Outcome: success

Preference signals:
- The user repeatedly asked some version of “How [is] `notion` different from my current `docs/` folder (doc-base) and `progress/` and what’d I expect to gain?” indicating they want the comparison grounded in their actual repo conventions, not generic tool marketing.
- The user framed the question as a practical decision about what they would “expect to gain,” which suggests future answers should emphasize concrete workflow benefits and trade-offs rather than abstract feature lists.

Key steps:
- The assistant checked repo docs and memory notes to ground the answer in CoordExp’s own routing model.
- `docs/AGENT_INDEX.md` and `docs/PROJECT_CONTEXT.md` confirmed the repo’s current split: `docs/` is the stable contract/workflow layer; `progress/` is the historical/evidence layer.
- `progress/README.md` confirmed `progress/` is router-first and explicitly non-normative, used for historical derivation, experiment evidence, audits/diagnostics, and benchmark context.

Reusable knowledge:
- In CoordExp, the canonical split is already explicit in repo docs: `docs/` = current/stable contracts and workflows; `progress/` = dated evidence, diagnostics, benchmarks, and research history.
- `progress/` should not be used as normative truth if `docs/` or `openspec/specs/` already answer the question.
- The repo’s own authoring rule is to promote a note from `progress/` to `docs/` only when it is no longer tied to one dated run, defines the current recommended workflow, and would reasonably be the first page someone opens.

Failures and how to do differently:
- No major failure here. The main useful pattern was to anchor the explanation in the repo’s own precedence/routing docs rather than giving generic Notion-vs-markdown advice.

References:
- `docs/AGENT_INDEX.md`: `progress/` only when current docs do not answer the historical or empirical question; `docs/` first for current behavior.
- `docs/PROJECT_CONTEXT.md`: precedence order `openspec/specs/` -> `docs/` -> `openspec/changes/<active-change>/` -> `progress/`.
- `progress/README.md`: “Current behavior belongs in `docs/`. Historical motivation and empirical evidence belong here.”

## Task 2: Compare Linear vs current repo workflow

Outcome: success

Preference signals:
- The user asked “How about the `Linear` tool? Would I expect gain from using/learning it?” which suggests they want a direct usefulness assessment tied to their research workflow.
- The user’s repeated focus on research progress and documentation implies they care about separating active tasks from durable evidence.

Key steps:
- The assistant framed `Linear` as an execution/task layer distinct from `docs/` and `progress/`.
- The answer emphasized a practical separation: `progress/` records what happened, `docs/` records what is now true, and `Linear` records what still needs to be done.
- A minimal starter setup was proposed: 4 states (`Backlog`, `In Progress`, `Blocked`, `Done`) and 4 labels (`experiment`, `eval`, `infra`, `docs`).

Reusable knowledge:
- `Linear` is most valuable when the pain is dropped follow-up work, parallel research threads, or operational task tracking; it is not a replacement for repo documentation or evidence logs.
- A lightweight Linear taxonomy is likely enough to start; avoid overbuilding process before it proves useful.

Failures and how to do differently:
- No concrete failure. The answer should continue to be framed in terms of the user’s actual pain points and the repo’s existing division of labor, not as a generic PM-tool recommendation.

References:
- The concise workflow mapping given in the answer: `progress/` = what happened, `docs/` = what is now true, `Linear` = what still needs to be done.
- Suggested minimal states/labels for an initial research queue.

## Task 3: Determine whether Web GPT can access Linear/Notion content

Outcome: success

Preference signals:
- The user asked twice, in slightly different words, whether `Linear` and `Notion` are “online document base[s] that sync to web so that my Web GPT have access to,” indicating they care specifically about what a web-connected ChatGPT session can or cannot see.
- The user later asked whether the Notion docs could be readable “so tha only same notion account can read the docs,” which shows they care about access control and not just publication.

Key steps:
- For Linear, the assistant checked current docs and explained that web access requires an explicit connector/integration; being online does not make data automatically visible to a generic browsing ChatGPT session.
- For Notion, the assistant explained the distinction between public pages, private workspaces, and connected app access.
- For the permission question, the assistant clarified that private/invite-only sharing is the safe way to ensure only specified Notion accounts can access the docs, while publishing to the web removes that restriction.

Reusable knowledge:
- Public web availability and connected-app availability are different access paths.
- For private research docs, use invite-only/private sharing; do not publish to web if account-restricted access is required.
- ChatGPT access to Linear/Notion content depends on having the relevant connector enabled and authorized in that specific ChatGPT environment.

Failures and how to do differently:
- The assistant initially reasoned from product behavior, then verified against current help docs/search results. Future similar answers should keep the distinction explicit: public URL access vs authenticated connector access vs repo-local access.

References:
- Linear connector explanation: access requires an authenticated Linear connector/integration in ChatGPT; public web presence alone is not enough.
- Notion sharing explanation: invite specific people/accounts for private access; publish-to-web makes it publicly link-readable.

## Task 4: Try to create a toy Linear doc and install/login the Linear plugin

Outcome: partial

Preference signals:
- The user asked for a toy Linear doc multiple times and then explicitly said, “Please help me install the `linear` plugin accordingly. I'll need your autonomy,” which is a strong signal they want the assistant to drive the setup rather than wait for step-by-step prompting.
- The user then shortened that to “$linear help me login,” reinforcing that they wanted an autonomous login/setup flow.
- The repeated “Please create a toy doc by Linear.” requests indicate the desired deliverable was the doc itself, not just an explanation.

Key steps:
- The assistant inspected the repo-local Codex config and plugin cache.
- `~/.codex`-equivalent repo-local config showed `apps = true` and explicit plugin enables for `github@openai-curated`, `notion@openai-curated`, and `linear@openai-curated`.
- The current Linear plugin bundle existed in the repo cache under `.codex/plugins/cache/openai-curated/linear/...`, with app id `asdk_app_69a089a326dc8191b32a3f2553f5be2c`.
- The assistant determined the blocker was not installation but authentication/session refresh: the live Linear endpoints were not exposed in the session, so it could not create or list Linear docs/issues from tools.
- Because the connection was not live, the assistant could not actually create the toy doc and instead gave the exact login steps and a ready-to-paste toy doc template.

Reusable knowledge:
- In this workspace, the repo-local Codex config is the effective plugin source of truth; the `Linear` plugin was already enabled in `/data/CoordExp/.codex/config.toml`.
- The presence of plugin files in `.codex/plugins/cache/openai-curated/linear/...` does not mean the app is authenticated; tool availability still depends on the live app connection.
- The current bundle path found during the later check was `/data/CoordExp/.codex/plugins/cache/openai-curated/linear/6807e4de/.app.json`, and the app id inside it was `asdk_app_69a089a326dc8191b32a3f2553f5be2c`.
- The expected next step after app login was to start a fresh Codex session/thread before retrying Linear actions.

Failures and how to do differently:
- Attempting to create the toy doc before the connector was authenticated failed because the Linear tool endpoints were not exposed in the session.
- The assistant correctly stopped at the auth boundary instead of pretending to create the doc; future agents should do the same when the plugin is enabled locally but the live connector is unavailable.
- A useful retry rule emerged: after Linear OAuth completes in the UI, open a fresh session and then re-attempt a safe read/create action.

References:
- `/data/CoordExp/.codex/config.toml:151` — `[plugins."linear@openai-curated"] enabled = true`
- `/data/CoordExp/.codex/plugins/cache/openai-curated/linear/6807e4de/.app.json:1` — `{"apps":{"linear":{"id":"asdk_app_69a089a326dc8191b32a3f2553f5be2c"}}}`
- `/data/CoordExp/.codex/plugins/cache/openai-curated/linear/6807e4de/.codex-plugin/plugin.json` — describes Linear as “Find and reference issues and projects” / “Manage issues, projects, and team workflows in Linear from Codex.”
- Recommended next-step wording given to the user: connect Linear in the Apps/Connectors panel, complete OAuth, then send `Linear connected` or retry the create action in a fresh session.
