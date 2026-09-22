# Four-owner free-h clarification

Status: **CPU clarification complete** from the accepted raw-token/native-parser artifacts; no new model or visual call.

| Image / owner | GT class | N16 selected row / IoU | A best same-class row / IoU | A best any-class row / IoU |
|---|---|---:|---:|---:|
| 351017 / 2094819 | bottle | 4 / 0.532423 | 3 bottle / 0.379083 | 3 bottle / 0.379083 |
| 351017 / 96050 | bottle | 5 / 0.547788 | 3 bottle / 0.000000 | 4 person / 0.009485 |
| 417044 / 1079910 | donut | 22 / 0.863176 | 20 donut / 0.089882 | 20 donut / 0.089882 |
| 417044 / 1083295 | donut | 19 / 0.828431 | 15 donut / 0.071938 | 15 donut / 0.071938 |

All four A maxima are below IoU 0.5, so none is a global-assignment-only loss with an otherwise eligible owner edge. All four do have same-class predictions somewhere: the evidence localizes changed box/row support, not disappearance of the class and not physical missing-owner status.

## Literal row alignment

| Image | N16/A complete free rows | Earliest differing row (one-based) |
|---:|---:|---:|
| 477415 | 19/20 | 4 |
| 351017 | 13/6 | 3 |
| 417044 | 26/22 | 4 |
| 388795 | 5/5 | 3 |

For 351017, N16's selected owner rows are 4 and 5 while A has 6 rows; for 417044 they are 19 and 22 while A has 22 rows. Thus A's earlier EOS does not truncate before the corresponding N16 positions. It remains a correlated route-length change, not an identified cause.

On 388795, N16 emits exact `c` only at free row1/token offset0. A emits it at row1/offset0 and repeats it at row4/offset29 (zero-based offset; row token offsets29–37).

Remaining unknowns: physical presence/quality without visual review, class/extent ambiguity, and the cause of changed row order, geometry, and EOS. These four GT50 losses must not be promoted to physical missing-owner claims.
