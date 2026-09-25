# Current upstream boundary

The maintained source uses Transformers/Qwen, PyTorch, Accelerate, PEFT and the
selected attention backend. Inspect the installed versions and actual call
signatures when changing their behavior; old pinned notes are provenance, not an
automatic compatibility promise for a newer environment.

Local interface notes: [Qwen-VL](upstream/QWEN_VL.md),
[FlashAttention](upstream/FLASH_ATTENTION.md),
[training ecosystem](upstream/TRAINING_ECOSYSTEM.md).
These notes support the retained core rather than a legacy MS-Swift runner.

Before an upstream change, verify token/processor/image-grid identity,
logits-to-keep and causal alignment, packed attention isolation, dtype/backend
requirements, adapter target/payload composition and loss reduction. Run the
actual affected CPU contracts; real-model numerical qualification is separate.
Do not infer runtime compatibility from a source URL, package name or old result.
