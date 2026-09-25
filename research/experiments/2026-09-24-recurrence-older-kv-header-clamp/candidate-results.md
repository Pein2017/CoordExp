# Older K/V header clamp candidate

Status: **candidate cold readback passed; lead acceptance pending.**

Protocol SHA `3d2ad946c4fe5560987fa14c4f9de364c0d7e8c7e0abe2c2a0a3f5b3418c1cbe`; admission SHA `4cbaa75a831698324550c4b3d2dbc9a9e5636fcaff5c9c595cf97e205ff5cb77`; producer SHA `f895be9d785d1e9c5b8be3d41cb5c402ae06f4f9b3aa2ee55fc829a87aad24ef`.
Raw receipt SHA `32bf6a9f9be6bb884ffdb08530461fd45b223f5b4b6ee6cca2d69b3c61da89bb`; cold readback SHA `6a7089b35776856aa602448cfc8ca41518d8755c3faa7765c9c692a5487e4036`. Original refined-03 four requests, target2 train351017; AF/FF common header replayed, zero generated tokens.

## Frozen FP64 full-vocabulary decision

Shared outcome **bidirectional_displacement_collapse**. Bidirectional endpoint retention **False**; displacement-collapse comparator **True**; secondary component pattern **not_retained**.

| Base | Adaptive D | Clamped D | e to adaptive joint | t from clamped native | Retention | Collapse | Component |
|---|---:|---:|---:|---:|---|---|---|
| AF | 0.78242855438 | 0.030320392024 | 0.975708551201 | 0.0387516430149 | False | True | False |
| FF | 0.530136028192 | 0.0543449783899 | 0.999269873624 | 0.102511384814 | False | True | False |

The clamped component ratios are descriptive; the secondary retained-pattern gate also requires the shared primary, which failed.

| Base | r_V to clamped joint | r_K to clamped joint | n_V to clamped native | n_K to clamped native |
|---|---:|---:|---:|---:|
| AF | 0.304169472 | 0.692677515 | 1.121406399 | 0.408115075 |
| FF | 0.787245633 | 1.738090598 | 0.507922307 | 0.769146636 |

| Clamped cell | TV to native | TV to adaptive joint | TV to adaptive counterpart | Global winner / runner | Gap | z(151671) − z(151670) |
|---|---:|---:|---:|---|---:|---:|
| clamp_native_AF | 0.000000000 | 0.782428554 | 0.000000000 | 151670 / 151673 | 1.272991180 | -1.332893372 |
| clamp_joint_AF_from_FF | 0.030320392 | 0.763422231 | 0.763422231 | 151670 / 151673 | 1.310722351 | -1.376060486 |
| clamp_V_AF_from_FF | 0.034001482 | 0.764842792 | 0.726758771 | 151670 / 151673 | 1.296457291 | -1.363149643 |
| clamp_K_AF_from_FF | 0.012374209 | 0.776450577 | 0.704535615 | 151670 / 151673 | 1.275209427 | -1.346817017 |
| clamp_native_FF | 0.000000000 | 0.530136028 | 0.000000000 | 151671 / 151670 | 0.302612305 | 0.302612305 |
| clamp_joint_FF_from_AF | 0.054344978 | 0.529748962 | 0.529748962 | 151671 / 151670 | 0.437984467 | 0.437984467 |
| clamp_V_FF_from_AF | 0.027603027 | 0.529670337 | 0.033862020 | 151671 / 151670 | 0.426275253 | 0.426275253 |
| clamp_K_FF_from_AF | 0.041799257 | 0.533592733 | 0.104404835 | 151671 / 151670 | 0.266731262 | 0.266731262 |

For each clamped cell, both fixed tokens’ full-vocabulary probabilities and ranks are preserved in the cold readback. Neither a mixed result nor scientific nonpass skipped a fixed component.

## Technical qualification and cost

Calls1–14 reproduced accepted all-four vectors; native AF/FF header Q/K/V was freshly captured at all28 layers and four positions. Both separately executed native clamps matched their anchors before six clamped treatments. All22 raw vectors/inputs and selected older K/V donors, latest/prehistory/companion complements, masks, rotary identities, actual Q/K/V outputs, appended K/V, scale-aware FP64 pre-o_proj reconstruction and finally restoration were checked. Cold readback recomputed saved-headout error and all scientific TVs in a separate CPU process; it did not perform a second SDPA forward.

Independent NumPy FP64 recomputation of both bases' D, e and t agrees with the cold reader within 1.67e-15. All 504 layer headout records passed; the largest reconstruction error was 0.102 of its frozen scale-aware bound, and maximum appended-current-K error was zero.

Actual 22 model / 4 vision / 0 generated. Parent outer 186.280167371 s; internal 179.413203910 s. Prior sequence 0.592427020230 GPUh; cumulative 0.644171511166 GPUh.
Peak RSS 11076928 KiB; GPU allocated/reserved 12476571136/13266583552 B; artifact bytes before receipt 119679542; terminal child PID 2180643, exit 0.

Raw attempt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-older-kv-header-clamp/attempt-001`. Technical gate tolerances are from the frozen unit; no scientific threshold was changed. F/F2 physical identity remains HOLD. The experiment does not identify a natural mediation fraction, literal address-content circuit or free-row outcome. No self-acceptance or successor.
