# User visual adjudication

Date: 2026-09-22. Source: direct user review in the lead task, after the
[accepted CPU readback](../results.md). These are user-supplied semantic
judgments, not another model run or a revision of the frozen acceptance receipt.

The user's observations:

> - `反复输出“person”的水面位置`,是幻觉,bad pred;
> - 单独的kite, 有些person, good/valid pred
> - bowl, 确实有个碗,但是 bbox 偏小了
> - 再沙滩上,直接一个很长的长方形的框的,bad pred,覆盖了多个 owners 的

Interpretation for continuation:

- The repeated water-region person predictions are hallucinations. Their
  numerical recurrence is not repeated coverage of a real person.
- The long beach rectangle is a bad prediction spanning multiple owners.
  Its appearance ends the exact numerical run but does not establish valid
  localization or owner recovery at that immediate transition.
- The isolated kite and some later person predictions are valid. This supports
  later return to valid predictions, without asserting that all later person
  rows are valid or that any is the first-ever coverage of its owner.
- The bowl exists, with an undersized predicted box. This resolves the earlier
  existence/category uncertainty in favor of a real bowl; it does not establish
  precise box quality, unique-owner matching, or a valid post-repeat transition.

For the next probe, distinguish numerical branch exit, return to valid
localization, and first-ever owner coverage. These are separate endpoints.
The original trace, renderings, measurements and acceptance receipt remain
unchanged. This note supersedes the earlier visual uncertainty only where the
user's statements explicitly resolve it.
