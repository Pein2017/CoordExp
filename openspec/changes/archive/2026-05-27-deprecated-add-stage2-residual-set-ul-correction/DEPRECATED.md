# Deprecated

This incomplete change is archived as historical context.

Reason: its remaining `stage2_trie_ce` / Stage-2 AB alias cleanup conflicts with
the hard-cut unified `stage2_rollout_correction` contract, which rejects
`stage2_trie_ce`, `stage2_ab`, and Channel-A/B public semantics.

Do not finish this change in place. If residual-set UL work is revived, open a
new change against `stage2_rollout_correction` only.
