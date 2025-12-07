# Packed Dataset Implementation Analysis

## 1. Goal
Implement a packed dataset mechanism in `CoordExp` to improve training efficiency for object detection tasks, mimicking `ms-swift`'s implementation. The goal is to maximize token usage per batch by packing multiple short samples into a single sequence context (e.g., 2048 or 4096 tokens), reducing padding waste.

## 2. Current State Analysis

### CoordExp Dataset Handling
- **Entry Point**: `src/sft.py` manually instantiates datasets (`BaseCaptionDataset` or `FusionCaptionDataset`) based on config.
- **Dataset Class**: `BaseCaptionDataset` (`src/datasets/dense_caption.py`) is a map-style `torch.utils.data.Dataset`.
- **Lazy Encoding**: It loads raw JSONL records but encodes them (via `template.encode`) lazily in `__getitem__`.
- **Length Information**: Token length is **not known** until `__getitem__` is called. This prevents global sorting/bin-packing without a full pre-pass.

### ms-swift Implementation
- **Classes**: `PackingDataset` (map-style) and `IterablePackingDataset` (streaming) in `swift/llm/dataset/utils.py`.
- **Logic**:
  - **Grouping**: Uses `binpacking.to_constant_volume` to group samples into chunks that fit `max_length`.
  - **Collation**: `template.data_collator` checks `template.packing=True` and uses `packing_row` to concatenate `input_ids`, `labels`, etc.
  - **Position IDs**: Crucially, the collator resets `position_ids` for each sub-sequence in a packed sample (Attention Sink / Packed Sequence approach).
- **Difference**: `ms-swift`'s `PackingDataset` expects an **already tokenized** dataset (or one with a `length` column) to perform global bin-packing. `CoordExp`'s dataset is lazy.

## 3. Feasibility & Gap Analysis

Directly using `ms-swift.llm.dataset.PackingDataset` is **not feasible** because:
1.  It requires `dataset['length']` (column access), which `BaseCaptionDataset` does not support.
2.  It assumes global access to lengths for optimal packing, which requires iterating the entire lazy dataset first (slow for "infinite" datasets).

**Recommendation**: Implement a custom `IterablePackedWrapper` in `CoordExp` that mimics `ms-swift`'s `IterablePackingDataset`.

## 4. Proposed Implementation Plan

### A. Create `src/datasets/packing.py`
Create a wrapper class `PackedDatasetWrapper` (inheriting from `IterableDataset`) that:
1.  **Wraps**: Takes an initialized `BaseCaptionDataset` (or `FusionCaptionDataset`).
2.  **Buffers**: Iterates over the source dataset, buffering `N` (e.g., 1000) encoded samples.
3.  **Packs**: Uses `binpacking` (or a simple greedy algorithm) to group buffered samples into `max_length` chunks.
4.  **Yields**: Yields list of samples `[sample1, sample2, ...]` as a single item. The `ms-swift` collator will handle the flattening.
5.  **Template Config**: Sets `template.packing = True` and `template.padding_free = True` upon initialization to trigger the correct collator behavior.

### B. Modify `src/sft.py`
Update the dataset loading section to wrap the dataset if packing is enabled in config.

```python
# src/sft.py

# ... existing dataset loading ...
dataset = BaseCaptionDataset.from_jsonl(...)

# New Logic
if custom_config.packing.enabled:
    from .datasets.packing import PackedDatasetWrapper
    logger.info(f"Enabling packed dataset (length={custom_config.packing.max_length})")
    dataset = PackedDatasetWrapper(
        dataset=dataset,
        template=sft.template,
        packing_length=custom_config.packing.max_length,
        buffer_size=custom_config.packing.buffer_size or 1000
    )
```

### C. Configuration
Add `packing` section to `src/config/schema.py` (or `custom_config`) to control this behavior via YAML.

## 5. Implementation Details (Draft)

```python
# src/datasets/packing.py (Draft)
import torch
from torch.utils.data import IterableDataset
from typing import List, Dict, Any
import binpacking

class PackedDatasetWrapper(IterableDataset):
    def __init__(self, dataset, template, packing_length, buffer_size=1000):
        self.dataset = dataset
        self.template = template
        self.packing_length = packing_length
        self.buffer_size = buffer_size
        
        # Side-effect: Enable packing on template for the collator
        self.template.packing = True
        self.template.padding_free = True # ms-swift specific flag for packing logic

    def __iter__(self):
        buffer = []
        iterator = iter(self.dataset)
        
        while True:
            try:
                # Fill buffer
                while len(buffer) < self.buffer_size:
                    item = next(iterator)
                    # Item is already encoded dict with 'length' from BaseCaptionDataset
                    buffer.append((item, item['length']))
            except StopIteration:
                is_finished = True
            else:
                is_finished = False

            if not buffer:
                break

            # Pack buffer using binpacking
            # binpacking expects (item, weight) tuples or similar. 
            # ms-swift uses: binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
            # where sequences is list of (index, length). 
            
            # We have full items, so we might need to separate them for binpacking
            items_with_len = [(i, x[1]) for i, x in enumerate(buffer)]
            
            packed_indices = binpacking.to_constant_volume(
                items_with_len, 
                self.packing_length, 
                weight_pos=1
            )
            
            # Yield packed groups
            for group in packed_indices:
                # group is list of (index, length)
                packed_sample = [buffer[idx][0] for idx, _ in group]
                yield packed_sample # Yields List[Dict]
                
            # Handle remaining/unused items if any, or clear buffer
            # Ideally, keep unplugged items for next batch if strictly implementing constant volume,
            # but for simplicity in streaming, we might just clear or rely on binpacking to return all.
            
            buffer = [] # Clear buffer for next chunk
            
            if is_finished:
                break
```

## 6. Dependencies
- `binpacking` library (needs to be installed/available in `ms` env).
- `ms-swift` template logic (already present).

## 7. Verification
- **Loss**: Since `ms-swift` collator resets `position_ids`, the model effectively sees independent samples in one batch. Loss calculation remains valid for each token.
- **Efficiency**: Monitor GPU utilization and throughput (samples/sec). Should see reduced padding (tokens per batch closer to max_length).
