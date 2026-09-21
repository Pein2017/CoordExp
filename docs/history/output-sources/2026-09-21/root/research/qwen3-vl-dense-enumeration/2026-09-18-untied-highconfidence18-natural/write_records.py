from pathlib import Path
import json
A=Path(__file__).parent;ROOT=A.parent;repo=Path('/data/CoordExp/.worktrees/research-probes');B=ROOT/'2026-09-18-untied-active-readout-geometry';C=ROOT/'2026-09-18-untied-gradient-path-accounting';D=ROOT/'2026-09-18-untied-recurrence-feedback'
d=json.loads((A/'reduction.json').read_text())
def save(root,text):(repo/'research/experiments'/root.name/'results.md').write_text(text)
s='# Natural inference — candidate, awaiting root acceptance\n\nAll 148 fixed native groups completed: 145 unique images × four conditions = 580 outputs. The independent saved-output reduction is JSON-exact. No further model calls were made during closeout. Compare complete tied versus untied+axis training packages; this is not an untie-only causal effect.\n\n## Full frozen strata\n\nPrimary class-agnostic one-to-one IoU≥0.5. UNKNOWN is annotation-unmatched, not physical FP. Repeat is the frozen valid-box near-repeat proxy, not automatically an owner recurrence.\n\n| Stratum / positives | Condition | Match / FN | Strict repeat | Invalid / malformed | UNKNOWN | EOS / cap |\n|---|---|---:|---:|---:|---:|---:|\n'
for stratum,total in [('human13',392),('refined5',178),('sentinel',919)]:
 for cond,z in d['summaries'][stratum].items():
  s+=f"| {stratum} / {total} | {cond.replace('untied','untied+axis')} | {z['matches']} / {z['FN']} | {z['strict_valid_repeats']} | {z['invalid']} / {z['malformed']} | {z['unknown']} | {z['eos']} / {z['cap']} |\n"
s+='\n## Turnover, damage and uncertainty\n\n| Stratum | Within-model normalization | New / lost known IDs | Match delta | Sentinel paired 95% interval |\n|---|---|---:|---:|---|\n'
for stratum in ['human13','refined5','sentinel']:
 for model in ['tied','untied']:
  c=d['comparisons'][stratum][model];p=c['pairs'];g=sum(len(x['gain']) for x in p);l=sum(len(x['loss']) for x in p)
  s+=f"| {stratum} | {model.replace('untied','untied+axis')} | {g} / {l} | {g-l:+d} | {c.get('paired_total_delta_95percentile_interval','not estimated')} |\n"
s+='''
Intervals use the frozen paired-image bootstrap (seed19, 10,000 resamples), not token independence. Both sentinel within-model intervals include zero. The sentinel is the previously used128-image panel, not fresh held-out evidence. Shared bird309264 is generated once but retains refined14 and sentinel10 annotation memberships; do not add the two strata as disjoint images.

Untied+axis normalization substantially reduces the selected refined5 debt and increases matches33→79; it also improves Human13 by7 known matches, but introduces3 invalid rows. Tied normalization loses8 Human13 matches and increases sentinel invalid rows174→284 despite lower strict valid repeats. Therefore no universal normalization repair or deployment promotion follows.

Sentinel healthy-baseline strata (EOS, no invalid/malformed/strict repeats): tied121 images contribute −3 match delta, seven unhealthy images +7; untied+axis122 healthy contribute +3, six unhealthy +15. Legal-border incumbent losses are2/102 tied and1/103 untied+axis on sentinel; Human13 loses1/16 tied and0/16 untied+axis; refined5 loses0/4 and0/6. Legal endpoints are not themselves errors.

Cross-package original policy has equal sentinel580 matches but23 new and23 lost IDs. Human13 is175→167 and refined5 is12→33. This is a package comparison confounded by axis loss, optimization and training history. Complete G/L/retention identities, IoU0.8 and same-category scores remain in `reduction.json`; aggregate equality is not preserved owner identity.

## Bounded physical review

The frozen first-proxy / first-loss-else-gain selection yielded30 events across the18 eligible images, below the72 ceiling. One bounded pass was reused, with full-image and local-context plots. Six events indicate retained owner with changed extent; one retained extent/class change; three credible local new-owner witnesses; one credible local omission; one confirmed same-chair recurrence with extent debt. Sixteen remain HOLD and two further events retain a region but not resolved owner identity. These are selected-event observations, not population precision or exhaustive disappearance findings. In particular changed IoU often reflects a different box around the same object. No new bird event is selected by the frozen rule. No labels were changed.

## Shadow decisions and technical evidence

Normalized trajectories have2392 tied and1963 untied+axis active winner disagreements with original logits at the SAME treated histories; all are coordinate-to-coordinate, with zero family switches or EOS-choice changes. These are treated-path observations, not proof that EOS is irrelevant on every native failing history.

Shared real-entry admission verified independent untied input/output deltas, effective base-plus-delta rows, no-op tokens, policy coefficients, non-coordinate invariance and parameter restoration. The source snapshots resolve historical producer paths whose current file was later changed; `verification.json` records the exact hash mapping. All148 group receipts and raw/trace/source bindings were checked; all six main exits are0 and all owned PIDs ended. Technical admission is separate from scientific improvement.

## Reproduction, cost and boundary

'''
s+=f"Authoritative [result]({A}/result.json), [reduction]({A}/reduction.json), [verification]({A}/verification.json), [physical sidecar]({A}/physical-review/final-review.json), [artifact map]({A}/ARTIFACTS.md), [terminal]({A}/terminal.json). CPU replay: `PYTHONPATH=. python {A}/reduce.py --output /tmp/untied-natural-recheck.json`.\n\n"
s+=f"A used {d['cost']['forwards']} model forwards, {d['cost']['vision_forwards']} vision passes, {d['cost']['active_tokens']} active generated tokens and {d['cost']['padded_token_work']} padded token work; completed-worker allocated time {d['cost']['allocated_gpu_seconds_completed_workers']:.3f} GPU-seconds, including load/qualification. Peak memory and per-group clocks remain in receipts. No dropped/failed scientific groups, no quality retries. No automatic follow-up. Root owns acceptance and next decisions.\n"
save(A,s)
b=json.loads((B/'summary.json').read_text());s='# Active readout geometry — candidate, awaiting root acceptance\n\nAll145 source images were inspected in frozen order;11 qualify for the geometric recurrence proxy. The bounded native panel has442 selected slots, not442 independent trials:230 tied and212 untied+axis. Both original models are retained on selected images, with their own literal native histories. Cross-model states are not matched-history causal contrasts.\n\n| Model package | Coordinate slots | Endpoint wins | Same endpoint after equal norms | Endpoint→interior | Any coordinate winner change |\n|---|---:|---:|---:|---:|---:|\n'
for m,z in b['models'].items():s+=f"| {m} | {z['coordinate_decisions']} | {z['endpoint_coordinate_wins']} | {z['same_endpoint_after_equal_norm']} | {z['endpoint_to_interior']} | {z['all_coordinate_winner_changes']} |\n"
s+='''
Equalizing actual effective output-row norms changes a subset of endpoint preferences at the SAME captured head input, while most endpoint wins survive through alignment. Output magnitude alone is insufficient to explain all these native decisions. This is an immediate coordinate readout sensitivity, not a free-continuation rescue or proof that upstream input embeddings are irrelevant. All1000 rows participate; non-coordinate logits remain fixed but full-vocabulary competition can change.

The affine logit fit has median R²0.515 tied /0.522 untied+axis, maximum0.897/0.855. Slope and residual terms reconstruct endpoint-versus-best-interior margins exactly; the slope actually opposes the winning endpoint in1/44 tied and3/46 untied+axis endpoint wins. Thus a simple affine trend is not a complete explanation. These are descriptive curve fits, not a semantic circuit or effective-rank proof.

Effective output coord999/0 retain norm ranks1/2 in both packages; untied input ranks28/234. Separate input/output payloads were verified. This shows output peaks coexist with untied inputs in this trained package, not that untying causally creates/removes peaks. Final RMS normalization includes the learned per-dimension gain; hidden norm alone does not select coordinate argmax when no bias exists.

Native replay parity passed442/442, with maximum top-two errors8.39e−5 tied /5.15e−5 untied+axis; all native margins exceed twice their discrepancy. The saved equal-norm coordinate winner margins also exceed twice those per-event discrepancies. Effective W·h reconstruction maxima are5.90e−6/5.59e−6 and final-norm reconstruction1.53e−5. Independent CPU re-reduction reproduced all saved tensor results bitwise and scalar results within1e−12. Stored full-vocabulary logits preserve EOS/row-boundary competition separately.

Selected residual/attention-output/MLP vectors from28 layers are retained only at these slots; `layer-summary.json` reports their norm distributions. They are observational and no causal layer responsibility is assigned. No KV archive, full attention matrices, layer patching or new rollout was performed. Physical recurrence stays HOLD unless the separately bounded natural review supports a same-owner witness.

'''
s+=f"[Result]({B}/result.json), [event selection]({B}/events.json), [per-event summary]({B}/summary.json), [artifact map]({B}/ARTIFACTS.md), [terminal]({B}/terminal.json). CPU reproduction: `PYTHONPATH=. python {B}/summarize.py`.\n\nCost436 model/vision forwards,1,077.887 allocated GPU-seconds,626,747,858 runtime tensor bytes reported; actual serialized .pt files total1,942,968,903 bytes, including static-weights.pt1,316,221,045 bytes. The static receipt reports73,736,192 logical tensor bytes, which undercounts serialized backing storage; original receipt is preserved and actual disk accounting supersedes that cost field. Both exits0; no live jobs. Completed within2,000-forward/6-GPU-hour/16-GiB bounds. The strongest surviving account is context-dependent alignment with a measurable readout-scale contribution; no causal feedback experiment is admitted by these observations alone.\n"
save(B,s)
save(D,f'''# Conditional feedback — admission HOLD

CPU-only closeout; zero model calls. No feedback experiment was released.

The existing untied+axis417044 native first strict repeat proxy is row9/P9. The repeated partial donut region remains physical identity/extent HOLD in the bounded review. A covered physical A, credible unvisited B and admissibly matched healthy native boundary are not jointly established. High IoU alone cannot fill this gap. No old tied/R16 prefix is substituted as natural untied reachability.

[Admission receipt]({D}/admission.json) binds the raw native trajectory, whole-token hash, proxy row and relevant review. Exact causal prefix/slots, score contrast, direction controls, epsilon and model budget are deliberately unfrozen because physical admission failed. This is not a feedback null. The shortest resolution is a root decision on a concrete physically grounded A/B/control contrast from saved evidence; there is no worker authority to search more images or launch derivatives.

[Result]({D}/result.json), [terminal]({D}/terminal.json). No live jobs; no labels, checkpoints or outputs altered. Root owns any future release.
''')
# Preserve original C failure and root correction text; add only a current result pointer.
cp=repo/'research/experiments'/C.name/'results.md';text=cp.read_text();marker='\n## Integrated package closeout\n';text=text.split(marker)[0]+marker+f'\nC remains technical-invalid and scientifically unanswered (7 forwards,2 backwards,0/18 population scenes). The CPU axis-context correction alone was independently verified by root in [lead-verification.json]({C}/cpu-axis-correction/lead-verification.json). It does not resolve the separate CE finite-difference discrepancy or validate the old axis derivative passes. No model rerun occurred. [Result]({C}/result.json), [terminal]({C}/terminal.json).\n';cp.write_text(text)
for root in [A,B,C,D]:
 files={'A':None}
 text=f'# Artifact index — {root.name}\n\n- `result.json`: bounded result and status; `terminal.json`: cost and ended-job witness. Candidate is not root acceptance.\n'
 if root==A:text+='- `panel.json`, `sources/`, `shared-gate.json`, `input-audit.json`: model/data/runtime identity and real-entry gate.\n- `runtime/{condition}/{group}/raw.json`, `trace.json`, `receipt.json`: all580 outputs, tokens/text/EOS/cap, per-step shadow winners/top2/logsumexp and raw chosen scores, exact source/companion identities.\n- `reduction.json`: parsed boxes, owner G/L/retention, all annotation views, errors/endpoints, secondary scores, bootstrap and shadow accounting. `reduce.py` reproduces it; `reduction-recheck.json` is JSON-exact.\n- `physical-review/events.json`, `final-review.json`, image/context PNGs:30 bounded events; UNKNOWN and HOLD preserved.\n- `verification.json`: exact checked bindings, source snapshot resolutions, exits/PIDs. `integrated-terminal.json`: four-unit candidate.\n- `weights/`: effective E/U and norm factors, saved once per producer; detailed selected states/logits are owned by B. No comprehensive KV/attention archive.\n- `closeout.py`, `write_records.py`: CPU integration, no GPU calls. `main-*.log/.exit`, qualification logs and original runtime-handoff.json retain execution history; terminal supersedes the old live handoff.\n'
 elif root==B:text+='- `static/static.json`, `static/static-weights.pt`: effective input/output rows, deltas, normalization gain, static spectra and identities.\n- `events.json`: frozen source-order panel, exact prefixes/roles/tokens and source bindings.\n- `native-captures/{tied,untied}-original/receipt.json`, `weights.pt`, per-event `.pt`/`.pt.reduced.pt`: full-vocabulary logits, pre/postnorm head input,28-layer current-position residual/attention/MLP vectors and exact replay checks.\n- `summarize.py`, `summary.json`, `layer-summary.json`: CPU reducer and compact per-event/layer summaries. `reconstruction-recheck.json`: earlier independent replay.\n- `runtime.json`, `native-captures/*.exit`: cost/exit evidence. No full KV or attention-matrix archive.\n'
 elif root==C:text+='- `job-closure.json`, `runtime/receipt.json`, `finite-difference.json` and original tensors/logs: preserved technical-invalid attempt.\n- `cpu-axis-correction/`: before/after producer snapshots, diff, actual-term CPU RED/GREEN checks, correction receipt and root CPU-only verification. No corrected model measurement exists.\n'
 else:text+='- `admission.json`: exact saved native candidate and missing physical/control admission; model calls0. No perturbation coefficients or model evidence are fabricated.\n'
 (root/'ARTIFACTS.md').write_text(text)
print('four unit records and artifact indexes written')
