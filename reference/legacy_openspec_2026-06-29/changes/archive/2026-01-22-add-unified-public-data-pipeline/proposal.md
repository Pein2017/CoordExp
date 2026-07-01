# Change: Add Unified Public-Data Dataset Prep Pipeline (Shell Runner)

## Why
Public datasets in `public_data/` (e.g., LVIS, Visual Genome) currently have per-dataset download/convert scripts and shared preprocessing scripts, but there is no unified, reproducible “one command” entrypoint. This increases setup friction and makes it harder to standardize outputs and validation across datasets.

## What Changes
- Add a **single shell entrypoint** (proposed `public_data/run.sh`) that orchestrates dataset preparation with a consistent interface across datasets.
- Introduce a lightweight **dataset plugin contract** in shell (`public_data/datasets/<dataset>.sh`) so each dataset can customize:
  - Internet download sources and options (proxy, mirrors, partial download, etc.).
  - Parsing/conversion from raw artifacts into the CoordExp JSONL contract.
- Reuse the existing shared preprocessing steps for all datasets:
  - `public_data/scripts/rescale_jsonl.py` (smart resize + relative image paths)
  - `public_data/scripts/convert_to_coord_tokens.py` (coord-token supervision)
  - Add or extend a contract validator to support both `bbox_2d` and `poly` plus coord-token values (current `public_data/scripts/validate_jsonl.py` is bbox-only), and keep minimal inspection via `scripts/inspect_chat_template.py`

## Non-Goals
- No change to the global data contract (`docs/data/JSONL_CONTRACT.md`).
- No refactor of existing dataset converters unless needed to fit the runner interface (wrapping is preferred).
- No new “fusion training” default; dataset mixing remains offline and explicit (e.g., `public_data/scripts/merge_jsonl.py`).

## Impact
- Affected area: `public_data/` dataset preparation workflows and documentation.
- Expected benefit: consistent dataset prep commands, standardized output locations, and repeatable preprocessing across datasets.
