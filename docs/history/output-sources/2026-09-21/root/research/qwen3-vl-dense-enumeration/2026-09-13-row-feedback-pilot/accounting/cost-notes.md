# Cost through endpoint v2

This note is governed by [cost-through-endpoint-v2.json](cost-through-endpoint-v2.json), which supersedes v1 only for the cap-compliance claim. Known receipt allocation through the completed endpoint is **12.62981675953086808677777777 GPUh lower bound**. The five disjoint buckets and the pending, excluded content retry are unchanged.

The six missing early outer intervals mean actual allocation is at least the known value, with no inferred upper bound for the missing launcher overhead. Therefore `27.37018324046913191322222223` and `35.37018324046913191322222223` GPUh are **upper bounds on remaining budget**, not verified remaining budget. Actual-spend compliance with the 40 GPUh declared stage envelope or 48 GPUh round ceiling is not fully verifiable from these receipts.

The declared stage ceilings remain a planning envelope: technical 2, teacher 2, fit 20, endpoint/content 16 GPUh, totaling 40 below the 48 GPUh round ceiling. Content-v1 uses its receipt-declared allocation; its roughly 3e-10 GPUh wall conversion difference remains recorded, and content retry stays pending, uninspected, and uncounted.

Final round accounting: `final-cost-v1.json` adds the successful content retry exactly once to v2. Known allocated GPU-hour lower bound: 12.85851471021150478677777777. The six missing early outer intervals remain unresolved; no exact actual-total compliance is inferred. All model jobs in this bounded round have settled.
