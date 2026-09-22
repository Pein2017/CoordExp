# Lead source finding for Lane D

The actual base coord_init.json was read after dispatch:
/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent/coord_init.json

It records natural_adjacent, 1000 numeric tokens, 8 frequencies, scale0.02, seed0.
Source: scripts/tools/expand_coord_vocab.py, build_coord_positional_features and initialize_natural_adjacent_coord_rows.

The positional feature is u=k/999 plus sin/cos(2*pi*2^i*u), i=0..7, followed by a common seeded random projection and digit-embedding mean. This is smooth as a continuous function but not necessarily slowly varying between integer bins. The highest-band phase change per adjacent bin is2*pi*128/999=0.8050527720910781 radians (46.1261261261 degrees).

Before projection/scaling/training, feature L2 distances for bin differences1,2,4,8,999 are respectively0.90970130269,1.70473402560,2.62674593725,2.63132703521,1.0. The sinusoidal features coincide at0 and999 up to floating-point rounding; only the linear feature distinguishes the endpoints. A common digit-mean cancels from pairwise differences and can make cosine similarity less informative.

These are source-derived feature facts, NOT measured mature embedding distances or a proven cause of recurrence. Add this precise hypothesis to Lane D's already-authorized audit: does the projected base and the actual base+trained input delta preserve, erase or amplify this frequency/endpoint geometry, and does it predict fixed-prefix sensitivity beyond native margin? Reuse actual rows, do not launch new training, encoding changes, parameter sweeps or a fifth lane. Output normalization and input-geometry hypotheses remain distinct.
