# Head direction/norm attempt 002 — worker candidate

**Technical candidate complete; lead acceptance pending.** Exact six-cell order ran at the original refined-03 four-request source, target2 train351017, A history raw[:9], replayed six-token current prefix and native attention at physical y1 query1376. The first four fresh N/R anchors and identities qualified before both G and D ran. No generation, new history, cached split, phase, or QKV clamp occurred. Every head and layer used its own incoming Q/K/V. The historical removal reference was accepted `remove_H`, not the current-mask endpoint.

## Frozen outcome

FP64 full-vocabulary TV baseline **B=0.964094437011828**. Normalized distances: d_N_G=0.929776197476041, d_R_G=0.937110707594177, d_N_D=0.960140751481809, d_R_D=0.390278974408115. TV(G,D)=0.853774636416870. **Paired direction primary NONPASS; paired norm comparator NONPASS; category `mixed_or_other`.** G is far from both anchors; D is closer to R but does not rescue either paired criterion. Winners below are descriptive only, not a criterion or physical-owner finding.

| Arm | Target winner | Accepted full-vector max error | Raw SHA-256 | Input SHA-256 |
| --- | ---: | ---: | --- | --- |
| native_anchor | 151703 | 0.0 | 04baaadbdb7f50d161a77752b3fc6f87ab3fd1acf877dae2079302af7388ad0f | fcf4a094187ef351cc8ff8d93a180471c1ec6ad44d1f48325add7fb65e0be4b5 |
| remove_anchor | 152203 | 0.0 | af91a19a770bd6addd28a4c0514cc51bca5c91053cfc6c919b8a0f05723154f4 | fcf4a094187ef351cc8ff8d93a180471c1ec6ad44d1f48325add7fb65e0be4b5 |
| native_identity | 151703 | 5.412101745605469e-05 | e6892737aac5813cd44adedd90a90814657baf7957e4eb7e2ae9dce65ba99f50 | fcf4a094187ef351cc8ff8d93a180471c1ec6ad44d1f48325add7fb65e0be4b5 |
| remove_identity | 152203 | 0.0 | 408e9505229892880e751c17ee43cca2588b623f2999cd83552487571c1e203b | fcf4a094187ef351cc8ff8d93a180471c1ec6ad44d1f48325add7fb65e0be4b5 |
| gain_control | 151952 | new | 69ebfd5aeb1c75b41775aa74dc50efe01afeeae486e7e5eff2de10d51e551128 | fcf4a094187ef351cc8ff8d93a180471c1ec6ad44d1f48325add7fb65e0be4b5 |
| direction_control | 152206 | new | 850a5ce2249fd13a69708b6954f83da36029090376b05465140a8dfcf4412d0b | fcf4a094187ef351cc8ff8d93a180471c1ec6ad44d1f48325add7fb65e0be4b5 |

## Qualification and evidence

- Protocol `60ccbb639cb1a86b0c663edc0454d7806089a703a9c2d51f72fef130d3d979c0`; admission `db2ead15457ee36ba2d7d1d732aa9aa0e451ae6ee0fd9157491f69914cfca317`; parent repair `24e6a7d8883f2bade152204e6b326fee25150c2384cc204f18e03261a0cd4a25`; Bash command ruling `224be97a06caeb64c6d18dfff91c5810101947522b70d0d8b09e885b60c2b3c0`. Frozen preflight `62aa0144edb1153d4a6c6c4f004100a2fa7a177056645eefbae9055b4ffb102b` captured 34 maintained source/import pairs and passed 50 CPU caller checks plus all six actual cold fixture paths. The 14 computational ASTs remained frozen.
- Exactly one qualified nonmodel CUDA fixture: 6 cold, 6 mixed-device actual reader and 20 negative checks; CUDA reconstructed inputs/mask and CPU serialized Q/K/V/input were observed. Fixture receipt `0ba26a10c21080af889fc9d8a148ab82154a0c0acd37e8f2631e3419f23c81ad` and child `281aa84c1bf23247b3878222867fefc5a9daad880cb2f404287e6d543808ebe5`. Original failed qualification-v1 and shell logs remain unchanged.
- Exactly 6 model and 6 vision forwards, 0 generated tokens and 0 reused calls. All 28 actual native masks and post-o_proj selected/complement consumers were checked for each call. Separate CPU readback `bd98f0cb78e4b0be281b881aa15d0a1c401ea4ac925f36c5fc2547244ac3b309` reconstructed source, Q/K/V, rotary, partition formulas, raw vectors, identities, companion traces and finite counts. Maximum fresh reference error 5.41210175e-05 <=2e-4. Maximum recorded prewrite headout absolute error 0.000107225608 passed its scale-aware5e-5 gate. Maximum per-head FP32 relative norm error 4.29000611e-08 and unit-direction max error 2.56504558e-08, each below2e-6.
- Model raw artifacts: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-head-direction-norm/attempt-002`; 1996777128 bytes, below the 3 GiB planning envelope. Model receipt `ef6dcdc43e4e564f5d94cfa2dae8d1b2db1e64c0214ebc69c6b79deb8057ea52`; inner launcher receipt `c43961da871c9c9bac249ffff7275c72d479bc5f09dd1319fdad3c56e55f3847`. Both terminal child PIDs are absent, exit0. No new arm or retry was run.

## Cost and boundary

Full enclosing Bash intervals, including Python startup/imports/checks: fixture **24.082 s**, model **59.535 s**. Prior sequence charge **0.9370399939039638 GPUh** includes the two preserved shell setup failures (0.000061877 s). Corrected cumulative charge **0.9602669383484082444444444444 GPUh**. The immutable cold readback's embedded cumulative omits that tiny shell overhead; the versioned [final ledger](/data/CoordExp/.worktrees/research-probes/research/experiments/2026-09-24-recurrence-head-direction-norm/supporting/attempt-002-final-ledger-v1.json) SHA `250a488433373d0a9980ac7d013ae1d00406b78962c82045bc7999288a61ce95` is authoritative for final accounting. Inner child/launcher intervals remain separate in that ledger.

This is a fixed-endpoint current-query intervention at one A prefix. It does not identify a physical F/F2 owner, a natural recurrence loop, or a general mediation fraction. The worker stops at this candidate; lead owns independent acceptance and any successor decision.
