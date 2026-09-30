import importlib
import json
import sys
from pathlib import Path
import pytest


def test_import_does_not_load_model_stack():
    before = set(sys.modules)
    importlib.import_module('probes.runtime_compat')
    assert not ({'torch','transformers','peft','flash_attn'} & (set(sys.modules)-before))


def args_for(root):
    from probes.runtime_compat import parse_args
    (root/'checkpoint').mkdir()
    argv=['--checkpoint',str(root/'checkpoint'),'--output',str(root/'result.json')]
    for key in ('inputs','retained','encodings','policy'):
        p=root/(key+'.json');p.write_text(json.dumps({} if key=='policy' else [{'image_id':7116}]))
        argv+=['--'+key,str(p)]
    return parse_args(argv)


def test_inputs_only_no_output(tmp_path):
    from probes.runtime_compat import inspect_inputs
    args=args_for(tmp_path);result=inspect_inputs(args)
    assert set(result)=={'inputs','retained','encodings','policy'}
    assert not args.output.exists()


def test_existing_output_and_duplicate_image_fail(tmp_path):
    from probes.runtime_compat import inspect_inputs
    args=args_for(tmp_path)
    args.inputs.write_text(json.dumps([{'image_id':7116}]*2))
    with pytest.raises(ValueError):inspect_inputs(args)
    args.output.write_text('keep')
    with pytest.raises(FileExistsError):inspect_inputs(args)
    assert args.output.read_text()=='keep'
