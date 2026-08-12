# Wave 2 CPU BF16-derived gradient-provenance diagnostic

Status: post-failure mechanism diagnostic only; no GPU/model/cache use and no
change to the frozen v2 acceptance result or tolerance.

Runtime: repository `ms` environment, PyTorch 2.9.1, CPU autocast BF16,
`torch.manual_seed(17)`. One FP32 trainable matrix was used by either one
concatenated matmul or two separate matmuls. Both arms formed the same global
sum-of-squares objective and called backward once. Gradients were compared in
FP32 at both pre-existing frozen bands.

| Shapes `(n1,n2,d)` | Loss abs diff | Gradient max abs diff | FP32 band `1e-4/1e-5` | BF16 band `5e-3/5e-3` |
|---|---:|---:|---|---|
| `(17,19,64)` | `0` | `0.0078125` | fail | pass |
| `(127,131,256)` | `0.000030517578125` | `0.00390625` | fail | pass |
| `(1436,1386,128)` | `0` | `0.00390625` | fail | pass |

The number of nonzero gradient-difference elements was respectively `1430`,
`22008`, and `4409`. Large maximum relative differences were concentrated near
zero-valued reference elements and are not used as a promotion signal.

A second CPU-only comparison changed the separate arm from one backward over
the summed loss to the production-shaped two streaming backward calls. For the
same three shapes, combined-versus-streaming maximum absolute gradient
differences were `0.005859375`, `0.0029296875`, and `0.0029296875`.
Separate-summed versus separate-streaming differences were `0.00390625`,
`0.001953125`, and `0.001953125`. Every comparison still failed the FP32 band
and passed the BF16 band. Thus matching production cadence remains required for
fidelity, but cadence alone does not repair storage-dtype tolerance selection
in this diagnostic.

This diagnostic establishes a narrow mechanism fact: FP32 parameter storage
does not imply FP32-equivalent gradients when the forward/backward computation
is BF16-derived and the packed and separate arms use different matmul/reduction
shapes. Parameter-storage-dtype-only tolerance selection can therefore reject
mathematically equivalent objectives even when their scalar losses agree.

It does **not** establish which real Qwen parameter failed, the magnitude of the
discarded real gradient mismatch, that the real v2 gradients would pass the
BF16 band, or that v2 should be retrospectively accepted. The final GPU receipt
remains terminal failed and immutable. Any change from storage-dtype to
compute-provenance tolerance is a new post-result contract/version decision
owned by the user and requires fresh pre-execution review before any new GPU
evidence.
