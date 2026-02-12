## 0) Prereqs / quick sanity

- **Runner sanity** (no dataset required; validates the runner wiring):
  ```bash
  ./public_data/run.sh help
  ./public_data/run.sh vg help
  ```

* **Disk check**: VG raw images + resized images are large

  * Raw: `public_data/vg/raw/`
  * Resized: `public_data/vg/<preset>/`

---

## 1) Download raw VG artifacts

You run this; nothing is downloaded automatically.

**Recommended** (unified runner `public_data/run.sh:1` + plugin `public_data/datasets/vg.sh:1`):

```bash
./public_data/run.sh vg download -- --objects-version 1.2.0
```

**Notes:**

* If zip files already exist in:

  * `public_data/vg/raw/annotations/`
  * `public_data/vg/raw/`

  the step uses `wget --continue` so it can resume partial downloads. If a file is
  already complete, `wget` may print `416 Requested Range Not Satisfiable` which
  means "nothing left to download".

* For annotation-only (not trainable), add `--skip-images`.
  For real training, download images.

**Expected raw layout after download:**

* `public_data/vg/raw/annotations/image_data.json` (+ objects json)
* `public_data/vg/raw/images/VG_100K/...`
* `public_data/vg/raw/images/VG_100K_2/...`

---

## 2) Convert raw VG annotations → contract JSONL

(pixel coords, relative images)

```bash
./public_data/run.sh vg convert -- --objects-version 1.2.0 --val-mod 5
```

**Outputs:**

* `public_data/vg/raw/train.jsonl`
* `public_data/vg/raw/val.jsonl`

**Important defaults** (from `public_data/scripts/prepare_visual_genome.py:1`):

* **Split**: `image_id % val_mod == 0` → val
* **Boxes**: converted to `xyxy` and clipped to `[0,W-1] / [0,H-1]`
* **Paths**: images are referenced as relative paths under `public_data/vg/raw/`
  (e.g. `images/VG_100K/1.jpg`) to match `docs/data/JSONL_CONTRACT.md`

**Optional: write stats**

```bash
./public_data/run.sh vg convert -- --objects-version 1.2.0 \
  --stats-json public_data/vg/raw/convert_stats.json
```

---

## 3) Rescale images + rewrite geometry (offline)

Produces resized images under the preset and JSONLs that still reference
`images/...` relative to the preset directory
(implemented by `public_data/scripts/rescale_jsonl.py:1`).

**Repro command (explicit params; recommended):**

```bash
./public_data/run.sh vg rescale --preset rescale_32_768_bbox -- \
  --image-factor 32 \
  --max-pixels $((32*32*768)) \
  --min-pixels $((32*32*4)) \
  --num-workers 8
```

**Outputs:**

* `public_data/vg/rescale_32_768_bbox/train.jsonl`
* `public_data/vg/rescale_32_768_bbox/val.jsonl`
* `public_data/vg/rescale_32_768_bbox/images/...`

---

## 4) Export training-ready JSONL variants

Two common training modes:

### A) Coord-token mode

(used by many `CoordExp` configs)

```bash
./public_data/run.sh vg coord --preset rescale_32_768_bbox
```

**Outputs:**

* `public_data/vg/rescale_32_768_bbox/train.coord.jsonl`
* `public_data/vg/rescale_32_768_bbox/val.coord.jsonl`

### B) Numeric norm1000 ints (LVIS-style)

Not emitted automatically; run directly:

```bash
conda run -n ms python public_data/scripts/convert_to_coord_tokens.py \
  --input public_data/vg/rescale_32_768_bbox/train.jsonl \
  --output-norm public_data/vg/rescale_32_768_bbox/train.norm.jsonl

conda run -n ms python public_data/scripts/convert_to_coord_tokens.py \
  --input public_data/vg/rescale_32_768_bbox/val.jsonl \
  --output-norm public_data/vg/rescale_32_768_bbox/val.norm.jsonl
```

Converter: `public_data/scripts/convert_to_coord_tokens.py:1`

---

## 5) Validate (fail-fast before training)

```bash
./public_data/run.sh vg validate --preset rescale_32_768_bbox
```

Runs `public_data/scripts/validate_jsonl.py:1` on:

* Raw train / val
* Preset train / val
* Coord-token train / val
  …and best-effort `scripts/tools/inspect_chat_template.py` if a cached model exists.

**Troubleshooting: `*.coord.jsonl` invalid bbox**

If you see errors like `x2 <= x1` / `y2 <= y1` in `train.coord.jsonl`, it's usually
because very thin (1px) boxes can collapse under norm1000 quantization when
converted into coord tokens.

Current pipeline behavior is designed to be robust:

* `public_data/scripts/convert_to_coord_tokens.py` uses floor/ceil rules for
  `bbox_2d` in norm1000 space to preserve extents.
* As a safety net, it drops objects whose `bbox_2d` is still invalid after
  normalization (should be extremely rare).

If you updated the code and want to regenerate coord files:

```bash
./public_data/run.sh vg coord --preset rescale_32_768_bbox
./public_data/run.sh vg validate --preset rescale_32_768_bbox
```

---

## 6) Train: minimal config changes

### Coord-token training (recommended default)

```yaml
extends: base.yaml
custom:
  train_jsonl: public_data/vg/rescale_32_768_bbox/train.coord.jsonl
  val_jsonl: public_data/vg/rescale_32_768_bbox/val.coord.jsonl
  emit_norm: none
  coord_tokens:
    enabled: true
    skip_bbox_norm: true
```

### Numeric (pre-normalized ints) training

```yaml
extends: base.yaml
custom:
  train_jsonl: public_data/vg/rescale_32_768_bbox/train.norm.jsonl
  val_jsonl: public_data/vg/rescale_32_768_bbox/val.norm.jsonl
  emit_norm: none
  coord_tokens:
    enabled: false
```

---

## 7) Visual QA (optional but useful)

* **One-record overlay + legend** (counts or list):

  * Script: `public_data/scripts/visualize_jsonl_bbox_poly.py:1`
  * Example:

    ```bash
    conda run -n ms python public_data/scripts/visualize_jsonl_bbox_poly.py \
      --jsonl public_data/vg/raw/val.jsonl \
      --line 1 \
      --out temp/vg_vis/val1.png \
      --draw-poly 0 \
      --legend \
      --legend-mode counts
    ```
