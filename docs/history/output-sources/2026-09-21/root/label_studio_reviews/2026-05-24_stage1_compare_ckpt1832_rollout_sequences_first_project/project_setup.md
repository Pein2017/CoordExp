# Label Studio Project Setup: 2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project

This bundle is import-ready. A live Label Studio project was not created because `http://localhost:8080/api/version` returned 502 and no `LABEL_STUDIO_API_KEY` was available in the environment.

## Start Label Studio with local file serving

```bash
cd /data/CoordExp/label-studio
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/data/CoordExp
# Choose one of the normal Label Studio startup paths documented in README.md.
# Example for an installed package:
label-studio
```

## Create the project manually

1. Open Label Studio.
2. Create a new project named `CoordExp stage1 compare ckpt1832 rollout review`.
3. Paste the label config from:
   `/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/label_config.xml`
4. Import tasks from:
   `/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/import/tasks.json`

## Optional API shape

When `LABEL_STUDIO_API_KEY` and a healthy `LABEL_STUDIO_URL` are available, the project body is:

```bash
curl -X POST "$LABEL_STUDIO_URL/api/projects" \
  -H "Authorization: Token $LABEL_STUDIO_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/project_payload.json
```

Then import tasks to the returned project id:

```bash
curl -X POST "$LABEL_STUDIO_URL/api/projects/<PROJECT_ID>/import" \
  -H "Authorization: Token $LABEL_STUDIO_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/import/tasks.json
```
