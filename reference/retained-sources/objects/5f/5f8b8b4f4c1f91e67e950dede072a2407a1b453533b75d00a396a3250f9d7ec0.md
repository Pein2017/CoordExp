# Root pre-acceptance checkpoint

The first monitor was consumed exactly once. It fired on the execution child's
PACKAGE_CANDIDATE while the persistent worker was still integrating results.
This was runtime settlement, not an integrated research candidate or acceptance.
Root sent two steering messages; the worker owns completion and initial reduction.

Independent checks already completed:
- All1135 bindings from the runtime settlement/scaleout match; no recorded producer PID remains.
- `python -m probes.training_set_completion.numerical_feedback.reduce --output .../coordination/lead-reduction.json` completed; JSON-exact to worker reduction (277 cells including one no-op,6048 contrasts, one deduplicated pilot).
- Historical missing producer snapshots were recovered by the worker. Root independently hashes all four entries in `verification.json:source_resolution`; all match. No source gap remains at this checkpoint.
- One Luna-max read-only falsification pass (`/root/feedback_replay_audit`) found no decision-changing fixed-suffix or target-release blocker. 116/276 releases have shorter padded companions, but every analyzed target is unpadded. Only target outputs are interpretable. All25 source captures meet trace parity, all276 paired jobs pass. The explicit one native/no-op release reproduces288 saved tokens; this is not blanket companion qualification.

Bounded interpretation pending stable integrated candidate: role-selective local
coordinate sensitivity exists, but substitution usually fails to control the next
emitted value. Failure episodes copy the substitute at the first same-role field
in1/66 untied and2/55 tied replacements. Same-role responses also occur at the
nonrecurrent/pre-onset proxy boundaries; these are not necessarily globally
healthy trajectories. No failure-specific maintenance circuit, physical recovery,
or complete absence of historical feedback is established.

Worker must finish results/state and publish integrated-terminal.json plus
INTEGRATED_CANDIDATE, preserving the child settlement. Root has not accepted or
released another experiment yet. Current likely next discriminator is the
normalization common-versus-centered readout intervention suggested by Pro,
conditioned on accepting this evidence; no layer sweep or training is needed.

Transport note: `codex_app__wait_threads` with timeoutMs0 did not return within
several minutes and its exec cell was terminated. The existing socket read/steer
route worked and confirmed the same active worker turn. Do not start a duplicate.
