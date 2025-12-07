# Packed Dataset for CoordExp: Implementation Guide

## Executive Summary

This document outlines how to implement packed datasets in CoordExp for efficient training on large-scale public object detection datasets (LVIS, COCO, Objects365). Packing multiple samples into a single sequence maximizes GPU utilization by eliminating padding waste.

**Key Design Decision**: Since we have nearly infinite image-text pairs for object detection and truncation is acceptable, we need:
- Dataset fusion (mixing LVIS, COCO, Objects365) without per-dataset performance tracking
- Overall aggregated `loss` and `token_acc` metrics only
- Packing to maximize throughput

---

## ms-swift Packing Architecture Overview

### Core Components

ms-swift implements packing through these key modules:

| Component | Location | Purpose |
|-----------|----------|---------|
| `PackingDataset` | `swift/llm/dataset/utils.py:130` | Non-streaming packing dataset wrapper |
| `IterablePackingDataset` | `swift/llm/dataset/utils.py:188` | Streaming packing dataset wrapper |
| `calculate_matched_group` | `swift/llm/dataset/utils.py:117` | Bin-packing algorithm using `binpacking` library |
| `packing_row` | `swift/llm/template/base.py:553` | Concatenates multiple samples into one |
| `_data_collator` | `swift/llm/template/base.py:1596` | Handles padding-free collation |

### Arguments

```python
# From swift/llm/argument/base_args/base_args.py:84-85
packing: bool = False
packing_length: Optional[int] = None  # defaults to max_length

# From swift/llm/argument/base_args/template_args.py:41
padding_free: bool = False  # automatically set True when packing=True
```

### Required Dependencies

```
binpacking  # For efficient bin-packing algorithm
flash_attn  # Required for padding-free attention
```

---

## How Packing Works in ms-swift

### 1. Pre-computation Phase (`create_packed_idx`)

```python
# swift/llm/dataset/utils.py:158-177
def create_packed_idx(self):
    lengths = self.dataset['length']  # Pre-computed token lengths
    data = [(i, length) for i, length in enumerate(lengths)]
    
    # Process in batches with bin-packing
    sequences = binpacking.to_constant_volume(data, packing_length, weight_pos=1)
    
    # packed_idx: [[sample_indices...], ...]  - groups of samples to pack
    # packed_length: [total_tokens, ...]       - total length per group
    return packed_idx, packed_length
```

### 2. Bin-Packing Algorithm

The algorithm (from `binpacking` library, see [paper](https://arxiv.org/pdf/2404.10830)) groups samples to maximize packing efficiency:

```python
# swift/llm/dataset/utils.py:117-127
def calculate_matched_group(template, sequences, packing_length, is_finished=True):
    import binpacking
    # Groups (index, length) tuples into bins of max size packing_length
    sequences = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
    return sequences, ret_sequences
```

### 3. Row Packing (`packing_row`)

```python
# swift/llm/template/base.py:553-571
def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
    packed = {}
    length = []
    for r in row:
        length.append(r['length'])
    
    # Concatenate input_ids, labels, loss_scale
    for key in ['input_ids', 'labels', 'loss_scale']:
        packed[key] = sum((x.get(key) or [] for x in row), start=[])
    
    packed['length'] = sum((x['length'] for x in row))
    
    # Create position_ids for each segment (reset to 0 for each sample)
    packed['position_ids'] = sum((list(range(x)) for x in length), start=[])
    
    # Merge multimodal data
    packed.update(self._data_collator_mm_data(row))
    return packed
```

### 4. Qwen VL-specific Packing

For Qwen2.5-VL/Qwen3-VL, special position ID handling is required:

```python
# swift/llm/template/template/qwen.py:414-422
def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
    position_ids = []
    for r in row:
        r = r.copy()
        r['input_ids'] = torch.tensor(r['input_ids'])[None]
        position_ids.append(self._get_position_ids(r))  # Uses model's get_rope_index
    packed = super().packing_row(row)
    packed['position_ids'] = torch.concat(position_ids, dim=-1)  # 3D for mRoPE
    return packed
```

### 5. Data Collation (Padding-Free Mode)

```python
# swift/llm/template/base.py:1606-1619
def _data_collator(self, batch, padding_to=None):
    if self.padding_free:
        batch[:] = [self.packing_row(batch)]  # Pack all samples in batch
        assert 'position_ids' in batch[0]
    
    # In padding_free mode, batch is already a single packed sample
    if self.padding_free:
        assert len(batch) == 1
        for k in ['input_ids', 'labels', 'position_ids', 'loss_scale', 'channel']:
            v = batch[0].get(k)
            if v is not None:
                res[k] = v if k == 'channel' else [v]
```

---

## Implementation Plan for CoordExp

### Phase 1: Core Packing Dataset

Create `src/datasets/packing.py`:

```python
"""Packed dataset wrapper for efficient training."""

from typing import Optional, List, Dict, Any
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.distributed as dist
from tqdm import tqdm

from swift.utils import get_logger, is_dist, is_master

logger = get_logger()


def calculate_matched_group(sequences, packing_length: int, is_finished: bool = True):
    """Bin-pack samples into groups that fit within packing_length."""
    if len(sequences) == 0:
        return [], []
    import binpacking
    # sequences: List[(index, length)]
    sequences = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
    if sequences and not is_finished:
        sequences, ret_sequences = sequences[:-1], sequences[-1]
    else:
        ret_sequences = []
    return sequences, ret_sequences


class PackedDataset(Dataset):
    """Non-streaming packed dataset for CoordExp.
    
    Wraps a pre-tokenized dataset and groups samples using bin-packing
    to maximize GPU utilization.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        template,
        packing_length: Optional[int] = None,
        num_proc: int = 1,
        strict: bool = False,
    ):
        self.dataset = dataset
        self.template = template
        self.packing_length = packing_length or template.max_length
        self.num_proc = num_proc
        self.strict = strict
        
        # Enable packing mode on template
        template.packing = True
        template.padding_free = True
        
        # Compute packing indices (only on master, then broadcast)
        self.packed_idx, self.packed_length = (
            self._create_packed_idx() if is_master() else (None, None)
        )
        if dist.is_initialized() and is_dist():
            obj_list = [(self.packed_idx, self.packed_length)]
            dist.broadcast_object_list(obj_list)
            self.packed_idx, self.packed_length = obj_list[0]
    
    def _create_packed_idx(self):
        """Create packing indices using bin-packing algorithm."""
        # Requires dataset to have 'length' field pre-computed
        lengths = self.dataset['length'] if hasattr(self.dataset, '__getitem__') else [
            d['length'] for d in self.dataset
        ]
        data = [(i, length) for i, length in enumerate(lengths)]
        
        PACKING_BATCH_SIZE = 1000
        input_data, packed_idx, packed_length = [], [], []
        i = 0
        
        with tqdm(total=len(data), dynamic_ncols=True, desc='Packing: ') as prog_bar:
            while True:
                new_data = data[i:i + PACKING_BATCH_SIZE]
                input_data += new_data
                prog_bar.update(len(new_data))
                if not input_data:
                    break
                i += PACKING_BATCH_SIZE
                is_finished = i >= len(data)
                sequences, input_data = calculate_matched_group(
                    input_data, self.packing_length, is_finished=is_finished
                )
                packed_idx += [[x[0] for x in seq] for seq in sequences]
                packed_length += [sum(x[1] for x in seq) for seq in sequences]
        
        return packed_idx, packed_length
    
    def __getitem__(self, index) -> List[Dict[str, Any]]:
        """Return list of samples to be packed together."""
        sequence = self.packed_idx[index]
        row = [self.dataset[i] for i in sequence]
        return row
    
    def __len__(self):
        return len(self.packed_idx)


class IterablePackedDataset(IterableDataset):
    """Streaming packed dataset for large-scale data."""
    
    def __init__(
        self,
        dataset: IterableDataset,
        template,
        packing_length: Optional[int] = None,
        packing_interval: int = 128,
        strict: bool = False,
        cyclic: bool = False,
    ):
        self.dataset = dataset
        self.template = template
        self.packing_length = packing_length or template.max_length
        self.packing_interval = packing_interval
        self.strict = strict
        self.cyclic = cyclic
        
        template.packing = True
        template.padding_free = True
    
    def __iter__(self):
        if self.cyclic:
            iterator = self._cyclic_iter(self.dataset)
        else:
            iterator = iter(self.dataset)
        
        buffer = []
        for sample in iterator:
            # Encode sample and get length
            encoded = self.template.encode(sample, return_length=True)
            if not encoded:
                continue
            buffer.append((encoded, len(encoded['input_ids'])))
            
            if len(buffer) >= self.packing_interval:
                sequences, buffer = calculate_matched_group(
                    buffer, self.packing_length, is_finished=False
                )
                for row in sequences:
                    yield [r[0] for r in row]
        
        # Flush remaining
        if buffer:
            sequences, _ = calculate_matched_group(
                buffer, self.packing_length, is_finished=True
            )
            for row in sequences:
                yield [r[0] for r in row]
    
    @staticmethod
    def _cyclic_iter(iterable):
        while True:
            for x in iterable:
                yield x
```

### Phase 2: Template Adapter for Packing

Add to `src/datasets/packing.py` or `src/coord_tokens/template_adapter.py`:

```python
def pack_samples(template, row: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pack multiple encoded samples into one.
    
    Compatible with Qwen3-VL's mRoPE position encoding.
    """
    packed = {}
    keys = set()
    length = []
    
    for r in row:
        keys.update(r.keys())
        length.append(r['length'])
    
    # Concatenate sequences
    for key in keys:
        if key in {'input_ids', 'labels', 'loss_scale'}:
            packed[key] = sum((x.get(key) or [] for x in row), start=[])
        elif key == 'length':
            packed[key] = sum(x[key] for x in row)
    
    # Handle position_ids for Qwen VL models (mRoPE)
    if hasattr(template, '_get_position_ids'):
        # Qwen VL: use model's get_rope_index for 3D position_ids
        position_ids = []
        for r in row:
            r_copy = r.copy()
            r_copy['input_ids'] = torch.tensor(r['input_ids'])[None]
            position_ids.append(template._get_position_ids(r_copy))
        packed['position_ids'] = torch.concat(position_ids, dim=-1)
    else:
        # Standard: simple incremental position_ids per segment
        packed['position_ids'] = sum((list(range(x)) for x in length), start=[])
    
    # Merge multimodal data (pixel_values, image_grid_thw, etc.)
    packed.update(_merge_multimodal_data(row))
    
    return packed


def _merge_multimodal_data(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge pixel_values, image_grid_thw, etc. from multiple samples."""
    res = {}
    
    pixel_values = [b['pixel_values'] for b in batch if b.get('pixel_values') is not None]
    if pixel_values:
        res['pixel_values'] = torch.concat(pixel_values)
    
    for media_type in ['image', 'video']:
        grid_key = f'{media_type}_grid_thw'
        grids = [b[grid_key] for b in batch if b.get(grid_key) is not None]
        if grids:
            res[grid_key] = torch.concat(grids, dim=0)
    
    return res
```

### Phase 3: Integration with Existing Dataset

Modify `src/sft.py` to support packing:

```python
# Add to CustomConfig or yaml schema
packing: bool = False
packing_length: Optional[int] = None

# In main(), after dataset creation:
if custom_config.packing:
    from .datasets.packing import PackedDataset
    
    # Ensure dataset has 'length' field
    # (BaseCaptionDataset should return this from encode)
    
    dataset = PackedDataset(
        dataset=dataset,
        template=sft.template,
        packing_length=custom_config.packing_length,
    )
    logger.info(f"Packed dataset: {len(dataset)} packed sequences")
```

### Phase 4: Configuration

Add to `configs/base.yaml` or experiment configs:

```yaml
custom:
  packing: true
  packing_length: 8192  # or null to use max_length

# Required for packing:
template:
  padding_free: true

training:
  attn_impl: flash_attn  # Required for padding-free attention
  per_device_train_batch_size: 1  # Each "batch" is one packed sequence
  gradient_accumulation_steps: 16  # Adjust for effective batch size
```

---

## Fusion Without Per-Dataset Tracking

For your use case (LVIS + COCO + Objects365 without per-dataset metrics), simplify the existing fusion:

### Current FusionCaptionDataset Structure

The current `FusionCaptionDataset` in `src/datasets/unified_fusion_dataset.py` supports:
- Multiple sources with separate tracking
- Per-epoch resampling
- Source-level statistics

### Simplified Aggregated Approach

For "infinite" detection data where you only care about overall metrics:

```python
# In FusionCaptionDataset or a new SimpleFusionDataset:

class SimpleFusionDataset(Dataset):
    """Fusion dataset that only tracks aggregate metrics."""
    
    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None,
        seed: int = 42,
    ):
        self.datasets = datasets
        self.weights = weights or [len(d) for d in datasets]
        
        # Build unified index
        self._build_index()
    
    def _build_index(self):
        """Create shuffled index across all datasets."""
        indices = []
        for ds_idx, ds in enumerate(self.datasets):
            for sample_idx in range(len(ds)):
                indices.append((ds_idx, sample_idx))
        
        # Shuffle
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)
        self.indices = indices
    
    def __getitem__(self, idx):
        ds_idx, sample_idx = self.indices[idx]
        return self.datasets[ds_idx][sample_idx]
    
    def __len__(self):
        return len(self.indices)
```

---

## Metrics: Aggregate Only

Since you don't need per-dataset performance tracking:

### Simplified Metrics Collection

The trainer already computes aggregate `loss` and `token_acc`:

```python
# In swift/trainers/mixin.py:896-967
def _compute_acc(self, outputs, labels):
    # Computes accuracy across all samples regardless of source
    preds = logits.argmax(dim=-1)
    metrics = compute_acc(preds, labels, acc_strategy=args.acc_strategy, ...)
```

### Remove Source-Level Tracking (if needed)

If `FusionCaptionDataset` currently tracks per-source metrics via callbacks:
1. Keep the fusion for data mixing
2. Remove or disable the `FusionEpochCallback` source-level logging
3. Rely on trainer's built-in aggregate metrics

---

## Flash Attention Requirement

Packing requires flash attention for efficient variable-length attention:

```python
# swift/llm/argument/train_args.py:132-141
def _check_padding_free(self):
    if self.padding_free or self.packing:
        if self.packing:
            feature = 'packing'
            self.padding_free = True
        else:
            feature = 'padding_free'
        if self.attn_impl not in {'flash_attn', 'flash_attention_2', 'flash_attention_3'}:
            raise ValueError(f'The "{feature}" feature requires flash attention.')
```

---

## Performance Expectations

Based on ms-swift examples:

| Scenario | Without Packing | With Packing | Speedup |
|----------|-----------------|--------------|---------|
| Qwen2.5-VL-7B on LaTeX_OCR | ≥1 hour | 10 minutes | ~6x |
| LLM SFT | baseline | significant | ~2-4x |

The speedup depends on:
- Variance in sample lengths (higher variance = more padding waste = bigger gains)
- `max_length` setting (longer = more packing opportunity)
- Dataset size

---

## Summary: Implementation Checklist

1. **Dependencies**:
   - [x] `binpacking` library (add to requirements)
   - [x] `flash_attn` (already likely present)

2. **Core Code**:
   - [ ] `src/datasets/packing.py` - `PackedDataset` and `IterablePackedDataset`
   - [ ] Template adapter for `packing_row` (Qwen3-VL mRoPE compatible)
   - [ ] Data collator update for padding-free mode

3. **Configuration**:
   - [ ] Add `packing`, `packing_length` to `CustomConfig` schema
   - [ ] Update example configs

4. **Integration**:
   - [ ] Modify `src/sft.py` to wrap dataset with packing
   - [ ] Ensure `BaseCaptionDataset` returns `length` in encoded samples

5. **Testing**:
   - [ ] Verify token counts match before/after packing
   - [ ] Check loss computation is correct across packed samples
   - [ ] Benchmark throughput improvement

---

## References

- ms-swift packing implementation: `swift/llm/dataset/utils.py`
- Bin-packing paper: https://arxiv.org/pdf/2404.10830
- ms-swift packing examples: `examples/train/packing/`
- Qwen VL template: `swift/llm/template/template/qwen.py`

