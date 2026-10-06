# DetectionScene Clean-Break Config Archive

This directory quarantines retired config roots removed from current launch
routing during the DetectionScene clean-break cleanup. These files are preserved
for historical inspection only. Do not use this archive as a current config
authoring surface.

Routes selected at the historical clean-break boundary:

- Stage-1 compact detection teacher forcing: `configs/stage1/detection_teacher_forcing/`
- Stage-2 rollout correction: `configs/stage2/rollout_correction/`

Archive checkpoint for pre-cleanup compatibility: `5ac77b5a9131d27fd5681b8e4c6a8d233ec72844`.

Current main routing is owned by [the config README](../../README.md); these
historical routes are not a new authoring or launch instruction.
