# Human-13 image-1584 K16 no-update parity witness

Date: 2026-08-16 (UTC)

This is the raw, value-level record of a second production-shaped no-update
run performed after the first report-only observation was challenged for lack
of a durable handle.  The run was executed from
`/data/CoordExp/.worktrees/research-probes` with `CUDA_VISIBLE_DEVICES=0` and
the working tree at `HEAD=9e0b4ed7118ebb8539998d3a3ffc56009dd7d9e8`, plus the
uncommitted source/test changes recorded by
`worktree_diff_sha256=0f357277ec0a16858d1c27bb431ecdc6d0287e774482e40e0aa98d2727865850`.

The inline command called, in order:

1. `build_human13_all_hf_vertical_source_plan()`;
2. `assemble_human13_live_model(..., pack_count=1, repo_root=...)`;
3. `build_human13_parity_skeleton(image_id=1584, ...)`;
4. `open_hf_shared_surface(plan_image1584_k16(), assembly, skeleton)`;
5. for each of the four frozen seed groups, `sample_group` followed by
   `replay_group`; and
6. the session context-manager close and `resource_receipt` readback.

No Task-4 entry service was used.  The run did not call backward, an
optimizer step, checkpoint writing, HF-fp32 audit loading/forwarding, network
actions, or output creation.  It did not use the configured one-image output
root as a write target.

## Structured terminal observation

```json
{
  "schema": "human13_all_hf_shared_surface_no_update_witness.v1",
  "cwd": "/data/CoordExp/.worktrees/research-probes",
  "cuda_visible_devices": "0",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "attn_implementation": "flash_attention_2",
  "model_object_id": 139815703338672,
  "actions": {
    "audit_forwards": 0,
    "audit_model_loads": 0,
    "backwards": 0,
    "checkpoint_writes": 0,
    "network_actions": 0,
    "optimizer_steps": 0,
    "output_creations": 0
  },
  "elapsed_s": 475.637,
  "groups": [
    {
      "group_index": 0,
      "seeds": [35001, 35002, 35003, 35004],
      "token_counts": [100, 100, 109, 109],
      "sample_steps": 109,
      "replay_steps": 418,
      "sampled_group_sha256": "affe86343e90b91137855fdfb92d8e952200f2898ee63fefa256bfada00cff02",
      "replay_group_sha256": "79e4df364f0471fe1ae114af6019a5d47829bae632950978975a7328f869ddce",
      "max_abs_error": 0.0,
      "mean_abs_error": 0.0
    },
    {
      "group_index": 1,
      "seeds": [35005, 35006, 35007, 35008],
      "token_counts": [91, 100, 118, 109],
      "sample_steps": 118,
      "replay_steps": 418,
      "sampled_group_sha256": "39f078dfee6f69ab7cc56ac25bf4c6da59de56c27916d8366f6f8ea12df99009",
      "replay_group_sha256": "ecb5133d98674bd35e773873ace97f3ea4565b7b4f38df147ab48f17f025076a",
      "max_abs_error": 0.0,
      "mean_abs_error": 0.0
    },
    {
      "group_index": 2,
      "seeds": [35009, 35010, 35011, 35012],
      "token_counts": [109, 109, 109, 109],
      "sample_steps": 109,
      "replay_steps": 436,
      "sampled_group_sha256": "e9e87e03b7ffb9bb1b381bcc8a01fc7b4e568a6e3c907063b3d4c4124a912917",
      "replay_group_sha256": "f0c7069c40b8b2ed1b33dfc2b6df54967193d7d4f9e14515727093fcb7f954df",
      "max_abs_error": 0.0,
      "mean_abs_error": 0.0
    },
    {
      "group_index": 3,
      "seeds": [35013, 35014, 35015, 35016],
      "token_counts": [100, 100, 100, 127],
      "sample_steps": 127,
      "replay_steps": 427,
      "sampled_group_sha256": "a52a9d2f8ecaecfa40880e67ca227e1567c65490a6bb1c8d6ea3a04fde52e029",
      "replay_group_sha256": "fc33d806e1a6a6f3bb364f83c3fc449882a53ccafce8e983657e18a270c318db",
      "max_abs_error": 0.0,
      "mean_abs_error": 0.0
    }
  ],
  "resource_receipt": {
    "sample_forward_count": 463,
    "replay_forward_count": 463,
    "total_forward_count": 926,
    "no_cache_forward_count": 926,
    "sampled_group_sha256s": [
      "affe86343e90b91137855fdfb92d8e952200f2898ee63fefa256bfada00cff02",
      "39f078dfee6f69ab7cc56ac25bf4c6da59de56c27916d8366f6f8ea12df99009",
      "e9e87e03b7ffb9bb1b381bcc8a01fc7b4e568a6e3c907063b3d4c4124a912917",
      "a52a9d2f8ecaecfa40880e67ca227e1567c65490a6bb1c8d6ea3a04fde52e029"
    ],
    "replay_group_sha256s": [
      "79e4df364f0471fe1ae114af6019a5d47829bae632950978975a7328f869ddce",
      "ecb5133d98674bd35e773873ace97f3ea4565b7b4f38df147ab48f17f025076a",
      "f0c7069c40b8b2ed1b33dfc2b6df54967193d7d4f9e14515727093fcb7f954df",
      "fc33d806e1a6a6f3bb364f83c3fc449882a53ccafce8e983657e18a270c318db"
    ],
    "latest_replay_group_sha256": "fc33d806e1a6a6f3bb364f83c3fc449882a53ccafce8e983657e18a270c318db",
    "cleanup_state": "closed",
    "cleanup_reason": "completed",
    "cleanup_call_count": 1,
    "retained_graph_count": 0,
    "session_held_reference_count": 0,
    "assembly_ownership": "borrowed_external",
    "caller_release_claim": "not_claimed"
  }
}
```

The `replay_steps` field is the number of causal gathers in the value receipt;
the resource receipt's `replay_forward_count` is the number of model forwards
and is the exact 463-step parity count.  The assembly model is returned in its
normal training state before the shared-surface session enters; the session
itself enforces the admitted eval/no-cache identity while sampling and replay.

## Artifact boundary

The configured one-image root was inspected after the run.  It still contains
only the pre-existing stale reservation:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/one-image/run-reservation.json`

Its `run_id` is `20260816T032422Z-pid377949`, `pid=377949` is no longer
running, and its mtime is `2026-08-16 03:24:22 UTC`.  That reservation records
zero actions but has no terminal/resource/parity fields.  It is not claimed as
the run above, and it was not deleted or overwritten.  The research file
above is therefore the durable evidence handle for this no-update observation;
the stale reservation remains an unresolved Task-5 ownership/recovery issue.

Source/config/manifest hashes at capture:

```text
scripts/research/human13_hf_shared_surface_live.py  2229f580e103cd102ef780780c581b041e9e748e47beea5cc6a719c612281585
scripts/research/human13_live_model.py               ce36d163dedb9e832fba1b6b1759ebe9f310c227b6ed119d9670fb30c0307fa3
tests/research/test_human13_hf_shared_surface_live.py e80a6a9bec76d885da7037ac6568910b5a3db729d97fc9ef5d7d95ec62b59f89
tests/research/test_human13_live_model.py             0cd029f43ceb5528ae9b5707ced1b339c642921b95db50f14db22cfaba5bf20f
configs/coordexp_infras/research/human13_all_hf_shared_surface_vertical/01_image1584.yaml f93010394efb2d7cf7fcbe5061f6f8c89037087b5a1b6b590fabb0e9823aa01b
human13-k-union-manifest.json                         a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb
```

Claim boundary: this is real Qwen/CUDA/BF16/FA2 sampling and exact sampler-step
replay parity/resource evidence only.  It does not establish a private update,
HF-fp32 audit, rollback receipt, checkpoint/output publication, or downstream
continuation gate.

## Current-tree recapture after terminal metadata cleanup

Because the first capture preceded the cleanup-only list clearing, the same
no-update command was run again after that patch.  This second capture is the
current-tree binding used for the final gate.  It again performed no update,
audit, checkpoint, network, or output action.

```json
{
  "schema": "human13_all_hf_shared_surface_no_update_witness.v2",
  "head": "9e0b4ed7118ebb8539998d3a3ffc56009dd7d9e8",
  "cuda_visible_devices": "0",
  "source_sha256": "f70a22867f161728a4f48525ac1ec6391b8dcce2df275756ef59211a4e028e56",
  "test_sha256": "ff87c46f7fdc38269148311a70dfa41042d8864e6be3f7f1f2dc4ce4f7a28a53",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "attn_implementation": "flash_attention_2",
  "assembly_model_training_before_session": true,
  "elapsed_s": 476.66,
  "groups": [
    {
      "group_index": 0,
      "seeds": [35001, 35002, 35003, 35004],
      "token_counts": [100, 100, 109, 109],
      "sample_steps": 109,
      "replay_steps": 418,
      "sampled_group_sha256": "3e1a1c290c0a9c787475d0abe6b4d5bf854c0426203f855225292f71d4d45ef6",
      "replay_group_sha256": "15b56f678799a333c0497dd29705da70dfe5412eea5a33ee293a1740013e1634",
      "max_abs_error": 0.0,
      "mean_abs_error": 0.0
    },
    {
      "group_index": 1,
      "seeds": [35005, 35006, 35007, 35008],
      "token_counts": [91, 100, 118, 109],
      "sample_steps": 118,
      "replay_steps": 418,
      "sampled_group_sha256": "726a0b22796fb32dea75b58707aaa9b347b764762bbc0f96a6830c8c023ee975",
      "replay_group_sha256": "af81fd89ab50b0d20c075cce21a2af84098c14a8d30ab47650e662f993841694",
      "max_abs_error": 0.0,
      "mean_abs_error": 0.0
    },
    {
      "group_index": 2,
      "seeds": [35009, 35010, 35011, 35012],
      "token_counts": [109, 109, 109, 109],
      "sample_steps": 109,
      "replay_steps": 436,
      "sampled_group_sha256": "06c86c6276e97a6638a0ce84c59d736a3c60afb38581abb2a04bbfac8232192c",
      "replay_group_sha256": "80636f793f83e10c79988bdcf0a534f734cb73ee5a1cc1fdc561410897e6d6b5",
      "max_abs_error": 0.0,
      "mean_abs_error": 0.0
    },
    {
      "group_index": 3,
      "seeds": [35013, 35014, 35015, 35016],
      "token_counts": [100, 100, 100, 127],
      "sample_steps": 127,
      "replay_steps": 427,
      "sampled_group_sha256": "76de0aa781488203e9097b71ecc311e3006b0bb95b2c15fa9ebe27223e8c7773",
      "replay_group_sha256": "24b411e8395e7b6270d9b499b4a5e60147f06f7f5b4746f5092d932673868b3e",
      "max_abs_error": 0.0,
      "mean_abs_error": 0.0
    }
  ],
  "resource_receipt": {
    "sample_forward_count": 463,
    "replay_forward_count": 463,
    "total_forward_count": 926,
    "no_cache_forward_count": 926,
    "sampled_group_sha256s": [
      "3e1a1c290c0a9c787475d0abe6b4d5bf854c0426203f855225292f71d4d45ef6",
      "726a0b22796fb32dea75b58707aaa9b347b764762bbc0f96a6830c8c023ee975",
      "06c86c6276e97a6638a0ce84c59d736a3c60afb38581abb2a04bbfac8232192c",
      "76de0aa781488203e9097b71ecc311e3006b0bb95b2c15fa9ebe27223e8c7773"
    ],
    "replay_group_sha256s": [
      "15b56f678799a333c0497dd29705da70dfe5412eea5a33ee293a1740013e1634",
      "af81fd89ab50b0d20c075cce21a2af84098c14a8d30ab47650e662f993841694",
      "80636f793f83e10c79988bdcf0a534f734cb73ee5a1cc1fdc561410897e6d6b5",
      "24b411e8395e7b6270d9b499b4a5e60147f06f7f5b4746f5092d932673868b3e"
    ],
    "latest_replay_group_sha256": "24b411e8395e7b6270d9b499b4a5e60147f06f7f5b4746f5092d932673868b3e",
    "cleanup_state": "closed",
    "cleanup_reason": "completed",
    "cleanup_call_count": 1,
    "retained_graph_count": 0,
    "session_held_reference_count": 0,
    "assembly_ownership": "borrowed_external",
    "caller_release_claim": "not_claimed"
  },
  "post_close_session_lists": {
    "sampled_groups": 0,
    "replayed_groups": 0,
    "live_replay_tensors": 0
  }
}
```

The configured output root still contains only the pre-existing stale
`run-reservation.json` described above; the current-tree recapture did not
create or modify it.
