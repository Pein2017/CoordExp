# Held row1: a prospective extent prediction

Lead note frozen before any row1 completion call, 2026-09-24. This adds a secondary readout of the already proposed trajectory, not another arm or launch authority. The physical-owner/validity outcome in `unit.md` remains primary.

Native history is A0 [0,157,277,504], then A1 [5,197,217,407], then B [157,0,415,241]. The admitted row2 mask removes access to A1 while leaving A0 readable; its complete output is [1,197,217,407], sharing A1's three remaining coordinates. Because all current-row queries are blocked from A1 keys at all layers, that agreement does not demonstrate literal copying from those blocked values. It is compatible with regenerating a similar localization from the image and remaining context.

A working hypothesis is that recent output serves as a reference for successive localization/progression: without the latest reference, the decoder can return to an earlier stage of the native sequence. A stronger alternative is generic perturbation of the region/coordinate competition, which need not recover any earlier extent.

The held row1 removes A0 from current-row attention and leaves no earlier completed detection row. Its masked x1=0 is already known from the accepted fixed-state readout and receives no new predictive credit. Freeze a strict prediction for the unobserved remainder: y1=157, x2=277, y2=504, followed by the native row terminator. Equivalently, the predicted five-token tail is [151670,151827,151947,152174,151649]. Judge exact equality of the three newly generated coordinates and terminator; no nearby-bin tolerance, fitted geometry threshold or substitution is allowed.

An exact A0 realization would support this one-step extent-reset prediction at the held completion endpoint. A different box rejects the exact prediction, even if physical owner A remains correct. Invalid/malformed/UNKNOWN results stay explicit. This does not establish an iterative algorithm, memory erasure, visited-owner semantics, or a common cause of natural bursts. Positions and sequence length remain native and differ between the original row0 and current row1.
