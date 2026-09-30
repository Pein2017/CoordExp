# Saved detection evaluation

The maintained entries are `python -m scripts.evaluate_detection --help` and
`python -m scripts.visualize_detection --help`. They delegate to `src.eval` and
`src.visualization` rather than historical launchers.

Current inference, evaluation, visualization commands, and output-root selection
are in the `coordexp-infer-eval-workflow` Skill.

[Contract](CONTRACT.md) specifies artifact binding and metric eligibility.
[Physical evaluation](../../research/questions/physical-evaluation.md) owns the
scientific distinction between annotation matching and physical-owner truth.
`src.eval.saved_rows` retains bounded readback accounting and golden fixtures.
No saved-row reader authorizes model continuation or turns a structural score
into a population precision/recall conclusion.
