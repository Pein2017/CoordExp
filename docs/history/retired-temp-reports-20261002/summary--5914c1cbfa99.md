# A4 rp1.10 Teacher-Forced Probe Summary

Scope: high-density subset of first val200, 32 images with GT count >= 10; prefix modes: GT prefixes K=0,1,3,5,10,N plus generated-prefix boundary from each run's own rp=1.10 rollout artifact.

## Core Margins

| run | gen free boundary n | gen sep-vs-eos mean | gen sep <=0 | gen entry mean | gt sep mean | gt sep <=0 | gt entry mean | true-end eos prob |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_random_sft | 9 | 6.215 | 0.556 | 17.340 | 9.622 | 0.000 | 20.143 | 0.435 |
| A2_support_balance | 27 | -1.306 | 1.000 | 15.162 | 8.525 | 0.056 | 19.137 | 0.880 |
| A3_prefix_rollin | 22 | -1.000 | 1.000 | 15.099 | 7.087 | 0.048 | 18.334 | 0.858 |
| A4_eos_trust | 18 | -0.667 | 1.000 | 16.066 | 7.387 | 0.032 | 19.362 | 0.692 |

## GT Separator By K

| run | K | n | sep-vs-eos mean | <=0 | valid mass mean | min margin |
|---|---:|---:|---:|---:|---:|---:|
| A0_random_sft | 1 | 32 | 10.178 | 0.000 | 0.999 | 4.375 |
| A0_random_sft | 3 | 32 | 10.527 | 0.000 | 1.000 | 5.875 |
| A0_random_sft | 5 | 32 | 10.121 | 0.000 | 0.998 | 3.375 |
| A0_random_sft | 10 | 28 | 7.384 | 0.000 | 0.953 | 0.500 |
| A2_support_balance | 1 | 32 | 9.625 | 0.000 | 0.999 | 3.750 |
| A2_support_balance | 3 | 32 | 9.715 | 0.000 | 0.999 | 4.125 |
| A2_support_balance | 5 | 32 | 8.891 | 0.000 | 0.993 | 2.000 |
| A2_support_balance | 10 | 28 | 5.491 | 0.250 | 0.806 | -2.625 |
| A3_prefix_rollin | 1 | 32 | 7.273 | 0.000 | 0.989 | 0.750 |
| A3_prefix_rollin | 3 | 32 | 8.023 | 0.000 | 0.996 | 2.875 |
| A3_prefix_rollin | 5 | 32 | 7.770 | 0.000 | 0.982 | 0.500 |
| A3_prefix_rollin | 10 | 28 | 5.022 | 0.214 | 0.808 | -2.375 |
| A4_eos_trust | 1 | 32 | 7.402 | 0.000 | 0.987 | 0.500 |
| A4_eos_trust | 3 | 32 | 8.105 | 0.000 | 0.997 | 3.250 |
| A4_eos_trust | 5 | 32 | 8.078 | 0.000 | 0.989 | 1.125 |
| A4_eos_trust | 10 | 28 | 5.759 | 0.143 | 0.868 | -1.125 |

## Generated Free Boundary By Generated Depth

| run | bucket | n | sep-vs-eos mean | <=0 | valid mass mean | min margin |
|---|---|---:|---:|---:|---:|---:|
| A0_random_sft | gen_depth_0_5 | 4 | 14.703 | 0.250 | 0.830 | -0.750 |
| A0_random_sft | gen_depth_6_10 | 2 | -0.688 | 0.500 | 0.357 | -1.500 |
| A0_random_sft | gen_depth_11p | 3 | -0.500 | 1.000 | 0.382 | -1.000 |
| A2_support_balance | gen_depth_0_5 | 1 | -1.625 | 1.000 | 0.165 | -1.625 |
| A2_support_balance | gen_depth_6_10 | 14 | -1.312 | 1.000 | 0.260 | -4.125 |
| A2_support_balance | gen_depth_11p | 12 | -1.271 | 1.000 | 0.246 | -2.625 |
| A3_prefix_rollin | gen_depth_0_5 | 3 | -1.250 | 1.000 | 0.237 | -1.875 |
| A3_prefix_rollin | gen_depth_6_10 | 9 | -1.125 | 1.000 | 0.260 | -2.000 |
| A3_prefix_rollin | gen_depth_11p | 10 | -0.812 | 1.000 | 0.314 | -1.625 |
| A4_eos_trust | gen_depth_0_5 | 1 | -1.125 | 1.000 | 0.245 | -1.125 |
| A4_eos_trust | gen_depth_6_10 | 10 | -0.512 | 1.000 | 0.378 | -1.375 |
| A4_eos_trust | gen_depth_11p | 7 | -0.821 | 1.000 | 0.311 | -1.375 |
