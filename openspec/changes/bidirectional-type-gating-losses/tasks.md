## 1. Governance Delta

- [x] 1.1 Create the repo-local OpenSpec change
  `bidirectional-type-gating-losses`.
- [x] 1.2 Add proposal and delta specs limited to the promoted
  `objective.terms.token_type_mass` contract.
- [x] 1.3 Keep coverage ledger, continuation, `bbox_positive_area`,
  smoke/train128 gates, and diagnostic salvage out of stable specs.
- [x] 1.4 Validate the change with
  `openspec validate bidirectional-type-gating-losses --strict`.

## 2. Implementation Follow-Up

- [ ] 2.1 Implement the hard-SFT token-type mass config/schema and loss behavior
  against the approved delta.
- [ ] 2.2 Add focused config, loss, and metric tests for the promoted
  token-type mass contract.
- [ ] 2.3 Do not mark this change ready for archive until implementation
  evidence exists and the stable specs can be synced safely.
