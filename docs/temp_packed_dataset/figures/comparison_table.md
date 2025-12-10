# Padding vs Packing: Configuration Comparison

## Training Configuration Comparison

| Parameter | Padding Mode | Packing Mode | Notes |
|-----------|--------------|--------------|-------|
| **Dataset** | | | |
| Total samples | 99,388 | 99,388 | Same dataset |
| Samples analyzed | 99,388 | 99,388 | Full analysis |
| **Batch Configuration** | | | |
| per_device_train_batch_size | 2 | 1 | Reduced for packing |
| effective_batch_size | 128 | 12 | New default (w=4 → grad_accum≈3) |
| num_gpus | 4 | 4 | Same hardware |
| gradient_accumulation_steps | 16 | ≈3 | Auto-computed |
| **Sequence Configuration** | | | |
| global_max_length | ~66930 (dynamic) | 16,000 | Fixed for packing |
| avg_sequence_length | 1334 | 1334 | Same input data (post-merge) |
| padding_overhead | ~40-50% | ~0% | **Efficiency gain** |
| **Training Dynamics** | | | |
| num_train_epochs | 4 | 4 | Same |
| steps_per_epoch | 776.5 | ~852 | Packing uses fewer micro steps but similar-quality opt steps |
| total_optimizer_steps | 3106 | ~3408 | Slightly higher; safer gradient scale |
| samples_per_step | 128 | ~117 | Close to baseline |
| total_samples_seen | 397,552 | 397,552 | Same (4 epochs) |
| **Efficiency Metrics** | | | |
| packing_efficiency | ~50-60% | ~100% | **~40pp improvement** |
| samples_per_pack | 1 | ~9.7 | Packing advantage |
| data_loss | 0% | ~0% | Negligible |
| **Performance Estimates** | | | |
| relative_training_speed | 1.0× | ~1.2-1.3× | **Faster with headroom** |
| gpu_memory_usage | Baseline | Similar / safer | Slightly longer seqs, fewer per batch |

## Key Takeaways

1. **Efficiency**: Packing reduces padding waste from ~40-50% to ~9%, improving GPU utilization
2. **Speed**: Expected ~30-50% training speedup due to reduced wasted computation
3. **Dynamics**: Different notion of "step" - packing processes ~13.5× more samples per optimizer step
4. **Equivalence**: Same total samples seen (4 epochs), but in ~14.0× fewer steps
5. **Memory**: Similar GPU memory usage (longer sequences but fewer per batch)

## Recommendation

**Use packing mode** with:
- `global_max_length: 16000`
- `per_device_train_batch_size: 1`
- `effective_batch_size: 12` (w=4 ⇒ grad_accum≈3)
- Adjust `eval_steps` and `save_delay_steps` to ~80/200 for 4 epochs (≈852 steps/epoch)

Monitor convergence closely; packing may achieve similar performance in fewer epochs due to more samples per step.
