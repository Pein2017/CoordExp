# Named model profiles

This is a small support owner for specific model compositions already consumed
by multiple research families. It is not an experiment runner, registry, factory
DSL, scheduler or a place for arbitrary shared helpers. General loading, adapters
and native operations remain in `src`.

| Profile | Meaning and guard |
|---|---|
| `source256.py` | FP32/SDPA with explicit DoRA and embedding payloads; the selected language training surface still checks588 tensors/18,006,016 scalars |
| `mature_tied_untied.py` | Fixed mature checkpoints and their1004 selected-token composition; tied versus untied delta sharing and selected input/output hashes remain distinct |
| `mature_source.py` | Adapter for the original-policy panel/raw/trace/receipt format; preserves companion order, literal IDs and executed media identity |
| `dora_composition.py` | Loaded DoRA/embedding identity checks with their existing receipt semantics |

`configs/source256.yaml` and the minimal `configs/source-gate-root` evidence were
moved byte-for-byte from their prior package. The source-gate note and roundtrip
receipt are required profile inputs, not a restored migration-time source archive.
Their schema, payloads and scientific identity were not rewritten.

Changing a named profile requires checking its actual consumers. It must not
silently choose a new checkpoint, precision, attention implementation, adapter
surface or native batch policy. The mature source adapter's old image-plan
enrichment remains explicit; it records a producer path as provenance rather
than importing or executing that producer.

`src.config.inference.replace_adapter_path` changes only the adapter path in an
immutable inference config. It is not a profile selector. Family tests validate
these seams; a CPU pass does not certify new real-model composition or generation.
