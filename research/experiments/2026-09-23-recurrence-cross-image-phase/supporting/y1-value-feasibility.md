# Latest-y1 value swap: CPU feasibility candidate

Status: **candidate CPU feasibility only**. No model load, CUDA operation, model/vision forward, generation, or GPU charge was made. This report answers the [lead brief](y1-value-feasibility-brief.md); it is not an experiment admission or a result about attention or copying.

## Bound source and arithmetic

The three fixed D cases are `mature:1584:2`, `mature:2299:2`, and `mature:4134:7` in [manifest.json](../manifest.json) (SHA-256 `1815e9e9cb5640876955d1f6b7b4bf2eb0d7d4ca4d292c60c2be374cef6201e0`). All use original four-request `refined-00`, with target batch indices 0, 1, and 3 respectively. The original raw, runtime receipt, and shared input identity have SHA-256 `cf49a34ded3df998b3f8ddce7cfbe687d65d09b6f09a49b48dcc816554f7f061`, `819fbc93e377b9b94c40b66c67bc244024ec743e972698125020cf1d26a4395b`, and `2856b08333a0d9082c4cfeafba868c0c1ca644c4b3f474add443ee48c2198a01`. The source image SHA-256 values for 1584/2299/4134 are `06b9d29a50b896f1bec14a267a57016723e54e205a1d1a40088237d95ce91206`, `cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3`, and `60dfd1369e0efa83dfa6c7d4035f4d9d66ca6ba9a0ec6b760349d0a0e30d7b34`, respectively.

All indices below are zero-based. Raw means index within the target completion; unpadded and cache mean physical sequence indices before and after left padding. Rotary phase lists all three axes. `y1` is offset 5 in each 9-token historical row. Current `x1` is the last token of the five-token S prefix; the next distribution predicts current `y1`.

| Case / batch | Role, coordinate token | Raw | Unpadded | Cache slot | Rotary phase (3 axes) |
| --- | --- | ---: | ---: | ---: | --- |
| 1584 / 0 | earlier y1, 152285 | 5 | 1377 | 1377 | (385,385,385) |
| 1584 / 0 | latest y1, 152311 | 14 | 1386 | 1386 | (394,394,394) |
| 1584 / 0 | current x1, 151847 | 22 | 1394 | 1394 | (402,402,402) |
| 2299 / 1 | earlier y1, 151968 | 5 | 1227 | 1377 | (391,391,391) |
| 2299 / 1 | latest y1, 151923 | 14 | 1236 | 1386 | (400,400,400) |
| 2299 / 1 | current x1, 151790 | 22 | 1244 | 1394 | (408,408,408) |
| 4134 / 3 | earlier y1, 151770 | 55 | 1417 | 1427 | (442,442,442) |
| 4134 / 3 | latest y1, 151977 | 64 | 1426 | 1436 | (451,451,451) |
| 4134 / 3 | current x1, 151965 | 72 | 1434 | 1444 | (459,459,459) |

The native query-minus-latest-y1 phase is `(8,8,8)` in every case. Shifting the latest row's **K phase** by +9 gives `(-1,-1,-1)`; shifting only S phase by -9 gives the same relative offset. Physical cache slots and causal order do not move. Earlier/latest/current y1 tokens are distinct in each case: 1584 `152285/152311/152293`, 2299 `151968/151923/151778`, 4134 `151770/151977/151964`. This arithmetic is a relative-position fact, not evidence that the query attends the slot.

## Saved V and reference bindings

The accepted raw roots are:

- 1584: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d-continuation-v1/d1-1584-2`
- 2299: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d-continuation-v1/d2-2299-2`
- 4134: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d4-only-v1/d4-4134-7`

The filenames below are relative to each root. SHA-256 values were rechecked against the actual files. The latest vector is the accepted whole-latest-row K+9 treatment, not a single-key treatment.

| Case | `prefill-blocks.pt` SHA-256 | `cells/01-full-native/vocabulary.pt` SHA-256 | `cells/03-native/vocabulary.pt` SHA-256 | `cells/06-latest/vocabulary.pt` SHA-256 |
| --- | --- | --- | --- | --- |
| 1584 | `0ab97c956cc4134d6d12c463491bb9953f35dbc0b2cee47e30cdf947418e90df` | `d1e1c5ce0c47d6f990cf28e7f8ea1ed651018ab546fdecaa68c56fc8fa83fbbc` | `60c50fd60f40d4cabb0cfd762b4cdd72fab526f9a1e407e4418f603948b278fb` | `da078c333c035415277224e465d97fa4988a11e4c0f4e574c2cc03dce8629d14` |
| 2299 | `2c1d3debb64cc33857829ccac0699bd76087d83a8424bfb84b03bfd391daf0e0` | `2fff6d7868756fa4543e06357dcadb2c35c5f1bf70aab0ea7dc004e7fcd265b3` | `d277f89c955fc7188dc234450e87df34789fe5cdfc2c89dc24b7a715d2308041` | `767c9b1a7255bb15601f1828ca699eb15724b1b78d9d7f2d1b2ffc9ebd065e99` |
| 4134 | `f0fd926ebccfb6ba1f4ddfded9550ce51487a280f8337c9272a71c7ed82c5ec1` | `b8bb9d0292ed76cc29fcd7a4819780aa04e125ddce37e3c414679d7dfa0374b9` | `0c3846dee48d8e63a15df0d2736b5c73f2e6d8f19c62deb7357bcafcc52e07c2` | `95d31e801f3328664c3fd1b4f2baff487f505add42a2a366fdaa7d9d72ae7f4f` |

Each prefill file is 6,280,857 bytes. Its `layers` array has 28 entries, each with native earlier/latest `pre_k`, `native_k`, and `native_v` tensors of shape `(8,9,128)`. CPU `torch.load(..., map_location="cpu", weights_only=True)` and all-layer finite/shape checks passed. For the proposed swap, donor is `layers[i]["earlier"]["native_v"][:,5,:]` and destination is `layers[i]["latest"]["native_v"][:,5,:]`, each `(8,128)` FP32 for every layer. No cross-case donor is needed.

| Case | Equal donor/destination layers | All-layer donor-minus-destination L2 range | Median L2 |
| --- | ---: | ---: | ---: |
| 1584 | 0/28 | 1.733–553.496 | 16.772 |
| 2299 | 0/28 | 2.122–1463.138 | 20.637 |
| 4134 | 0/28 | 3.415–1020.090 | 36.189 |

The donor V is a *contextual* native-prefill value, not a pure coordinate-token embedding. These magnitudes establish a nontrivial, finite intervention; they do not select layers or predict its output effect.

## Minimal maintained route and checks for a future admitted run

[scale.py](/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/recurrence_cross_image_phase/scale.py) is the frozen seven-call full-batch path (SHA-256 `cff77ec4b6ae3f129fa94c1710d7264810dc134cba4091668a2ab738b78238c4`). Its `source_inputs`/`verify_full` preserve original request order, companions, masks, EOS/pad and full-batch source step; `patched` wraps entry, suffix forward, `crop(width)`, and restoration in one `torch.inference_mode()` scope and compares full historical K/V digests. Its pre-attention observer verifies actual cache, causal mask and rotary inputs at all 28 layers; post-attention observer verifies historical K/V and companion suffix K/V. The accepted `prefill-raw.json`, `phase-qualification.json`, per-cell `consumer-raw.json`, and cold readback provide a direct pattern. [run.py](/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/recurrence_cross_image_phase/run.py) computes and independently checks the full +9 phase; [recurrence_native_x1_phase/run.py](/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/recurrence_native_x1_phase/run.py) supplies checked pre-K rotary reconstruction. None was edited here.

Smallest adaptation: keep the native prefill and its full-batch cache. For +9 conditions reuse the qualified latest-row K candidate for all 28 layers; native-phase conditions leave all K native. In the same checked context, optionally copy each saved earlier-row native V `[:,5,:]` into only `cache.layers[i].values[target,:,latest_start+5,:]`. Identity V-sham copies the native latest V back into that same slice while +9 K is active. Save the old K span and one V slice; `finally` crop S and restore both before requiring the original full-cache digest. The patch context must cover the actual suffix forward under inference mode, including body exceptions. A future CPU fixture should falsify wrong target/slot, wrong V donor, unrelated K/V changes, and failed restoration before any GPU launch.

At the actual attention consumer, record and compare per-layer hashes for: selected latest-y1 V against the saved donor/native value; the other eight latest-row V positions and all target V outside that slot against native; the selected latest K span against the qualified +9/native candidate; all target K outside that span and every companion historical K/V against native. Retain full-cache historical digest, mask and rotary checks; post-forward compare all companion suffix K/V against the native anchor and enforce final cache restoration. A full-source native vector and cached-native all-four-row parity are required before treatment. The +9 anchor must reproduce its accepted full-vocabulary vector, and the identity V-sham must agree with that anchor, before either donor cell. These are future qualification requirements, not checks claimed complete here.

The frozen prospective comparison would use earlier-y1 minus latest-y1 **logit margin** under native K and latest-row K+9, with and without the V swap. The lead still must freeze a numerical threshold and admission. Even a positive interaction would implicate use of the edited contextual V under this whole-row K treatment; it would not isolate the latest-y1 key alone, prove direct attention/copying, or establish physical recurrence. A negative would be limited to this all-layer single-V-slot intervention.

## Finite proposed cost; no launch authority

Per case: (1) original full native source, (2) native historical prefill, (3) cached native anchor, (4) latest-row K+9 anchor, (5) K+9 identity-V sham, (6) native-K donor-V, (7) K+9 donor-V. That is **7 model / 2 vision calls per case, 21 / 6 total**, zero free generation, with anchors and sham before donor conditions. Source/input shape remains four requests, 23,396,352 full-batch pixel elements; source widths are 1395, 1395, and 1445. All four source rows remain active at the respective source steps (raw offsets 23, 23, 73); keep companion identity checks rather than assuming they transfer.

The previous accepted seven-call runs measured 49.807, 49.521, and 50.670 allocated GPU seconds, including setup. Since the proposed route has the same per-case full-batch shape and number of full/vision/suffix calls, a 2× planning allowance is 99.614, 99.043, and 101.341 seconds: **299.997 seconds / 0.083333 allocated GPU-hour** total. Previous artifact bytes were 16,266,786; 16,266,812; and 16,303,077, totaling 48,836,675; 2× is 97,673,350 bytes. This is a forecast, not a runtime or artifact ceiling; new V observer data and failures must be charged. Prior closed sequence cost was 0.189686339 GPU-hour, yielding 0.273018885 GPU-hour if the forecast were fully charged. Prior observed GPU peak allocated/reserved was at most 11,291,790,336 / 12,081,692,672 bytes across these cases. A new hard cap and source-capture/command preflight belong to a lead-admitted contract.

**Feasibility conclusion:** geometry, saved native V, direct cache/consumer route, and measured call-shape cost support a finite qualification without a new donor or slot search. No CPU/source conflict was found. The material interpretation limit is that +9 shifts the entire latest K row while only one V slot changes; the proposed interaction is therefore conditional on that whole-row key context. No scientific or GPU qualification is claimed.
