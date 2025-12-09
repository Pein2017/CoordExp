# GPU Memory Usage Estimates

**Note**: These are rough estimates. Actual usage depends on model architecture, optimization settings, and DeepSpeed configuration.

## Memory Components

- **Model parameters**: ~8.0 GB (8B model in bf16)
- **Optimizer states**: ~8.0 GB (Adam with DeepSpeed Zero-2)
- **Activations**: Varies with sequence length and batch size
- **KV cache**: Varies with sequence length

## Activation Memory Comparison

| max_length | Padding (bs=2) | Packing (bs=1) | Difference | Notes |
|------------|----------------|----------------|------------|-------|
| 8,000 | 7 MB | 20 MB | +13 MB (+200.1%) | |
| 10,000 | 7 MB | 24 MB | +18 MB (+275.1%) | |
| 12,000 | 7 MB | 29 MB | +23 MB (+350.1%) | |
| 14,000 | 7 MB | 34 MB | +28 MB (+425.1%) | |
| 16,000 | 7 MB | 39 MB | +33 MB (+500.2%) | |
| 18,000 | 7 MB | 44 MB | +37 MB (+575.2%) | |
| 20,000 | 7 MB | 49 MB | +42 MB (+650.2%) | **Recommended** |
| 24,000 | 7 MB | 59 MB | +52 MB (+800.2%) | |

## Total Estimated GPU Memory (per device)

**Padding mode** (per_device_batch_size=2):
- Model + optimizer: ~16.0 GB
- Activations + KV: ~7 MB
- **Total**: ~16.0 GB

**Packing mode** (per_device_batch_size=1, max_length=20000):
- Model + optimizer: ~16.0 GB
- Activations + KV: ~49 MB
- **Total**: ~16.0 GB

## Conclusion

Memory usage is **similar** between padding and packing modes:
- Packing uses longer sequences but fewer per batch
- Total memory footprint is comparable
- DeepSpeed Zero-2 helps distribute optimizer states

**Recommendation**: Start with `global_max_length=20000` and monitor actual usage. Reduce if OOM occurs.
