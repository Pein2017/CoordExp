import pytest
import json
from probes.training_set_completion import repair as r

def test_forced_row_uses_exact_coordinates_and_no_eos():
    row=r.forced_row_tokens([7,8],[1,2,4,5],list(range(1000)),{"object_ref_start":10,"object_ref_end":11,"box_start":12,"box_end":13})
    assert row==[10,7,8,11,12,1,2,4,5,13] and r.EOS not in row

def test_forced_row_rejects_invalid_axes():
    with pytest.raises(ValueError,match="axes"):r.forced_row_tokens([7],[4,2,1,5],list(range(1000)),{"object_ref_start":10,"object_ref_end":11,"box_start":12,"box_end":13})

def test_forced_prefix_cap_excludes_eos():
    with pytest.raises(ValueError,match="terminal"):
        r._checked_terminal([1,r.EOS,2],"im_end")

def test_prefix_manifest_keys_are_normalized_before_json_hashing():
    prefixes = {str(25274): {"forced_prefix": [1]}, str(210457): {"forced_prefix": [2]}}
    assert json.loads(json.dumps(prefixes, sort_keys=True)) == prefixes
