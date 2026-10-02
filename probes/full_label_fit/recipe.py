"""Frozen recipe and optimizer profile helpers for full-label fitting."""

FULL_LABEL_MODE = 'full-label-region-v1'
IDENTITY_SELECTION = 'first_eligible_literal_duplicate_per_identity_round'
IDENTITY_NORMALIZATION = 'complete_positive_plus_margin_mean_1_over_K_per_image'
RESTORED_M_WEIGHTING = 'restored_M_mass'
RESTORED_M_COMPLETION = 'CHAIN_ALL_M_relocated_1_over_m_plus_B_1_over_m_plus_k'
OWNER_REGION = dict(kind='owner_region_max_hinge_v1', tau=.5, margin=.2, coefficient=1,
    prefix='actual_first_violation', coordinate_ce='replaced', coordinate_redirect_margin='included_in_row_region',
    auxiliary='existing_type_and_actual_prefix_order_geometry')
FULL_LABEL_BOUNDS = dict(vocabulary=152670, trace=[4456, 3084], bridge=[5032, 628], redirect=[4467, 11])
FULL_LABEL_DECODER = dict(backend='vllm', version='0.29.0+cu129', generation_config='vllm', logprobs_mode='raw_logprobs',
    temperature=0, top_p=1, top_k=-1, repetition_penalty=1, min_tokens=0, custom_processor=False,
    score_order='neutral_processing_greedy_argmax', loss_scores='HF_replay', outcomes='native_vllm_greedy')


def validate_lr_profile(profile):
    import math
    assert type(profile) is dict and set(profile) == {'lr_scale', 'warmup_updates'}, 'unknown LR profile fields'
    assert type(profile['lr_scale']) in (int, float) and math.isfinite(profile['lr_scale']) and profile['lr_scale'] > 0, 'invalid LR scale'
    assert type(profile['warmup_updates']) is int and 0 <= profile['warmup_updates'] <= 16, 'invalid warmup horizon'


def full_label_learning_rates(profile, global_update):
    assert type(global_update) is int and 1 <= global_update <= 16, 'invalid global LR update'
    if profile is None:
        return [1e-5, 5e-6]
    validate_lr_profile(profile)
    warmup = profile['warmup_updates']
    factor = profile['lr_scale'] * (min(global_update / warmup, 1) if warmup else 1)
    return [1e-5 * factor, 5e-6 * factor]


def full_label_recipe(checkpoint, manifest_sha256, training_path, training_sha256,
                      lr_profile=None, *, rollout_policy=None):
    if lr_profile is not None:
        validate_lr_profile(lr_profile)
    if rollout_policy is not None:
        assert type(rollout_policy) is str and rollout_policy == 'previous_rollout_tokens_lpt_v1', 'unknown full-label rollout policy'
    recipe = dict(mode=FULL_LABEL_MODE, checkpoint=str(checkpoint), manifest_sha256=manifest_sha256,
        training_path=str(training_path), training_sha256=training_sha256, owner_region=dict(OWNER_REGION),
        arms={'treatment': 1}, updates=[2, 16], completion={'treatment': RESTORED_M_COMPLETION},
        completion_weighting=RESTORED_M_WEIGHTING, event=IDENTITY_SELECTION,
        event_normalization=IDENTITY_NORMALIZATION,
        objective='positive_row_region_plus_description_divergence_softplus_1', execution_bounds=dict(FULL_LABEL_BOUNDS),
        decoder=dict(FULL_LABEL_DECODER),
        optimizer=dict(kind='fresh_continuous_AdamW', language_lr=1e-5, delta_lr=5e-6, betas=[.9, .999],
            eps=1e-8, weight_decay=0, clip=1, seed=92711))
    if lr_profile is not None:
        recipe['optimizer']['lr_profile'] = dict(lr_profile)
    if rollout_policy is not None:
        recipe['rollout_policy'] = rollout_policy
    return recipe
