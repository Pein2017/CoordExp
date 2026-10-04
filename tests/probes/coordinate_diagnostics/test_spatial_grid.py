"""Model-free actual entry, exact saved consumer and exhaustive gallery checks."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import test_cued_visual as fixtures
from test_cued_visual import bounded_threads
from probes.coordinate_diagnostics import spatial_grid as probe


class FixtureModel(fixtures.SyntheticModel):
    def generate(self,**kwargs):
        ids=kwargs['input_ids'].clone();raw=[];processed=[];cell=self.cell
        incoming=torch.get_num_threads()
        for pos in range(kwargs['max_new_tokens']):
            scores=torch.full((1,probe.COORD_START+1000),-12.,dtype=torch.float32)
            winner=self.anchor['reference_median_winners'][pos]
            j=cell['query_index']
            if j is not None and 14<=pos<=17:
                value=[0,30,min(999,cell['x1']+20),100][pos-14]
                if j==31 and pos==16:value=0  # Preserve original reversed box.
                winner=probe.COORD_START+value
                scores[0,probe.COORD_START+396]=-.125*(j+1)
            if self.break_anchor and j is None and pos==0:winner+=1
            if j==30 and pos==16:winner=kwargs['eos_token_id']  # Missing box stays a slot.
            scores[0,winner]=8
            raw.append(scores);decision=kwargs['logits_processor'](ids,scores);processed.append(decision)
            assert torch.get_num_threads()==incoming
            token=decision.argmax(-1).reshape(1,1);ids=torch.cat((ids,token),1)
            if int(token)==kwargs['eos_token_id']:break
        return SimpleNamespace(sequences=ids,logits=tuple(raw),scores=tuple(processed))


class FixtureSession(probe.NativeSession):
    q_frontend=None
    break_anchor=False
    loads=[]

    def __init__(self,checkpoint,packet):
        q=replace(self.q_frontend,model=FixtureModel());self.loads.append(checkpoint.name)
        with patch('probes.rule_stability.runner.native_components',return_value=(q,None,dict(compute='CPU_FIXTURE',research_model_loaded=False,research_forwards=0))):
            super().__init__(checkpoint,packet)

    def generate(self,cell):
        self.q.model.cell=cell;self.q.model.anchor=self.packet['conditions'][0];self.q.model.break_anchor=self.break_anchor
        autocast=torch.autocast
        with patch('torch.autocast',side_effect=lambda *args,**kwargs:autocast('cpu',enabled=False)):
            return super().generate(cell)

    def close(self):self.q=self.norm=self.batches=self.delta=None


@pytest.fixture(scope='module')
def prepared():
    from probes import rollout_row_credit as retained
    q=retained.frontend();assert q.model is None;FixtureSession.q_frontend=q
    directory=Path(os.environ['SPATIAL_GRID_CPU_OUTPUT'])/'prepared'
    if not directory.exists():assert probe.main(['prepare','--output',str(directory)])==0
    _,packet=probe.load_packet(directory/'native-proposal.json',cpu=True)
    return directory,packet


def test_actual_entry_publication_fresh_consumer_gallery(prepared):
    folder,packet=prepared;output=folder.parent/'cpu-positive';FixtureSession.loads=[]
    assert probe.main(['run','--config',str(folder/'native-proposal.json'),'--output',str(output)],cpu_factory=FixtureSession)==0
    report=probe.a.load(output/'readback.json');terminal=probe.a.load(output/'terminal.json')
    assert report['complete'] and report['completed_requests']==33 and report['requested_candidates']==32
    assert report['actual_forced_actions']==494 and report['actual_free_actions']==131 and report['generated_actions']==625
    assert sum(c['budget'] for c in packet['conditions'])==627
    assert FixtureSession.loads==['checkpoint-16'] and terminal['counts']['checkpoint_loads']==0
    assert all('target_box' not in c and 'owner' not in c for c in packet['conditions'])
    assert all('target_metrics' not in o for r in report['conditions'] for ch in ('raw','median') for o in r['record'][ch+'_observations'])
    assert report['conditions'][-1]['analysis']['outcome']=='complete_invalid_geometry'
    assert report['conditions'][-2]['analysis']['selected_row'] is None
    assert all(len(r['analysis']['selected_row']['same_category_overlaps'])==37 for r in report['conditions'] if r['analysis']['selected_row'] and r['analysis']['selected_row']['valid'])
    fresh=subprocess.run([sys.executable,'-m','probes.coordinate_diagnostics.spatial_grid','readback','--output',str(output)],cwd=probe.ROOT,capture_output=True,text=True)
    (folder.parent/'fresh-consumer.log').write_text(fresh.stdout+fresh.stderr)
    assert fresh.returncode==0,fresh.stderr
    semantic=copy.deepcopy(report);semantic.pop('compute');assert json.loads(fresh.stdout)==semantic
    incoming=probe.a.load(output/'invocation.json')['cpu_threads_incoming']
    other=int(next(x.split('=',1)[1] for x in fresh.stderr.splitlines() if x.startswith('spatial_grid_cpu_threads_incoming=')))
    assert incoming==2 and other!=incoming and not torch.cuda.is_initialized()
    probe.a.write(folder.parent/'semantic-report-equality.json',dict(exact=True,producer_incoming_threads=incoming,
        fresh_consumer_incoming_threads=other,only_removed_field='compute',research_forwards=0,CUDA_initialized=False))
    assert probe.main(['gallery','--output',str(output)])==0
    index=probe.a.load(output/'gallery/index.json');assert len(index['manifests'])==46
    assert index['candidate_slots']==32 and index['baseline_groups']==14 and index['baseline_occurrences']==307
    from PIL import Image
    with Image.open(output/'gallery/full-scene.png') as im:assert im.size==(1248,832)
    assert probe.a.load(output/'gallery/grid-30.json')['pred']==[]
    invalid=probe.a.load(output/'gallery/grid-31.json')['pred'][0]
    assert invalid['geometry_valid'] is False and invalid['source_bbox']==[984,30,0,100]
    assert invalid['bbox_pixel_xyxy']==[1228,25,0,83]
    assert all(len(probe.a.load(m['path'])['gt'])==len(packet['images']['351017']['objects']) for m in index['manifests'])
    with pytest.raises(FileExistsError):probe.run(folder/'native-proposal.json',output,cpu_factory=FixtureSession)


@pytest.mark.parametrize('mutation',['grid','history','boundary','order','request','frame','budget','role','pixels','baseline'])
def test_frozen_inputs_and_cue_boundaries_reject(prepared,mutation):
    _,original=prepared;p=copy.deepcopy(original);cell=p['conditions'][1]
    if mutation=='grid':cell['x1']+=1
    if mutation=='history':cell['forced_actions']['5']+=1
    if mutation=='boundary':cell['forced_actions']['15']=probe.COORD_START+30
    if mutation=='order':p['conditions'][1],p['conditions'][2]=p['conditions'][2],p['conditions'][1]
    if mutation=='request':p['requests']['351017']['clean']['media_sha256']='changed'
    if mutation=='frame':p['images']['351017']['width']+=1
    if mutation=='budget':p['bounds']['actions']+=1
    if mutation=='role':cell['row_start']+=1
    if mutation=='pixels':p['pixels']['array']['sha256']='changed'
    if mutation=='baseline':p['bindings']['analysis-351017']['sha256']='changed'
    with pytest.raises(ValueError):probe.verify_inputs(p)
    assert probe.GRID==[15,46,78,109,140,171,203,234,265,296,328,359,390,421,453,484,515,546,578,609,640,671,703,734,765,796,828,859,890,921,953,984]


def saved(prepared,name='grid-00'):
    folder,p=prepared;return probe.a.load(folder.parent/'cpu-positive'/(name+'.json')),next(c for c in p['conditions'] if c['condition']==name),p


@pytest.mark.parametrize('mutation',['summary','coordinate','position','prefix','request','action','vector','schema'])
def test_saved_vector_summary_action_identity_reject(prepared,mutation):
    r,c,p=saved(prepared);changed=copy.deepcopy(r)
    if mutation=='summary':changed['raw_observations'][0]['full_log_normalizer']+=1e-6
    if mutation=='coordinate':changed['median_observations'][1]['coordinate_scores'][396]+=1
    if mutation=='position':changed['median_observations'][1]['position']+=1
    if mutation=='prefix':changed['median_observations'][1]['prefix_token_ids'][-1]+=1
    if mutation=='request':changed['request_id']+='wrong'
    if mutation=='action':changed['raw_steps'][15]['requested_token_id']+=1
    if mutation=='vector':changed['raw_observations'][0]['vector_key']='median_14'
    if mutation=='schema':changed['schema']=probe.reused.SCHEMA
    with pytest.raises(ValueError):probe.validate_record(changed,c,p)


def test_vector_bytes_and_optional_gpu_diagnostic(prepared):
    from safetensors.torch import load_file,save_file
    r,c,p=saved(prepared);changed=copy.deepcopy(r)
    path=Path(r['score_tensors']['path']).parent/'corrupt.safetensors'
    vectors=load_file(r['score_tensors']['path']);vectors['raw_15'][probe.COORD_START+396]+=1
    save_file(vectors,str(path));changed['score_tensors']['path']=str(path)
    with pytest.raises(ValueError,match='vector bytes'):probe.validate_record(changed,c,p)
    changed=copy.deepcopy(r);changed['raw_observations'][0]['native_device_full_log_normalizer']+=10
    probe.validate_record(changed,c,p)
    with probe.cpu_diagnostics():
        assert probe.continuity.score_summary(changed['raw_observations'][0],changed['raw_steps'][14])==probe.continuity.score_summary(r['raw_observations'][0],r['raw_steps'][14])


def test_failed_anchor_holds_all32_before_execution(prepared,monkeypatch):
    folder,p=prepared;monkeypatch.setattr(FixtureSession,'break_anchor',True)
    output=folder.parent/'hold-anchor'
    assert probe.run(folder/'native-proposal.json',output,cpu_factory=FixtureSession)==2
    report=probe.a.load(output/'readback.json');counts=probe.a.load(output/'terminal.json')['counts']
    assert counts['attempted_requests']==counts['completed_requests']==1
    assert not report['anchor_qualified'] and len([r for r in report['conditions'] if r['status']=='skipped-HOLD'])==32
    assert report['requested_candidates']==32 and all(r['analysis'] is None for r in report['conditions'][1:])


@pytest.mark.parametrize('mutation',['omitted','duplicated'])
def test_slot_corruption_reject(prepared,mutation,monkeypatch):
    folder,p=prepared;output=folder.parent/'cpu-positive';entries=probe.a.load(output/'conditions.json')
    changed=copy.deepcopy(entries)
    if mutation=='omitted':changed.pop()
    else:changed[-1]=changed[-2]
    load=probe.a.load
    monkeypatch.setattr(probe.a,'load',lambda path:changed if Path(path)==output/'conditions.json' else load(path))
    with pytest.raises(ValueError,match='query slots'):probe.consume(output,p,FixtureSession.q_frontend.tokenizer)


def test_parser_original_geometry_roles_and_baseline_multiplicity(prepared):
    r,c,p=saved(prepared,'grid-31');analysis=probe.analyze(r,c,p,FixtureSession.q_frontend.tokenizer)
    assert analysis['selected_row']['box']==[984,30,0,100] and not analysis['selected_row']['valid']
    assert analysis['selected_row']['same_category_overlaps'] is None
    row=probe.visual_row(dict(condition=c['condition'],query_index=31,x1=984,analysis=analysis),p)
    assert row.pred[0].bbox_pixel_xyxy[0]>row.pred[0].bbox_pixel_xyxy[2]
    crop=probe.contextual_crop(row);assert crop[0]<crop[2] and row.pred[0].geometry_valid is False
    r,c,p=saved(prepared,'grid-30');analysis=probe.analyze(r,c,p,FixtureSession.q_frontend.tokenizer)
    assert analysis['selected_row'] is None and analysis['causal_roles']=={'14':'x1','15':'y1','16':'x2'}
    assert 'designated_annotation_id' not in analysis
    b=probe.baseline(p);assert sum(g['multiplicity'] for g in b['groups'])==307
    assert all(len(g['same_category_overlaps'])==37 for g in b['groups'])
    assert sorted(o['order'] for g in b['groups'] for o in g['occurrences'])==list(range(1,308))
    assert b['unavailable_tail']['box'] is None and b['unavailable_tail']['generated_order']==308
    assert b['unavailable_tail']['positions']==list(range(3079,3084))


def test_cpu_scope_restoration_and_unreleased_before_load(prepared):
    before=torch.get_num_threads()
    with probe.cpu_diagnostics():assert torch.get_num_threads()==1
    assert torch.get_num_threads()==before
    with pytest.raises(RuntimeError):
        with probe.cpu_diagnostics():
            assert torch.get_num_threads()==1;raise RuntimeError('intentional')
    assert torch.get_num_threads()==before
    folder,_=prepared
    with pytest.raises(ValueError,match='exact lead release'):probe.load_packet(folder/'native-proposal.json',cpu=False)
    assert not torch.cuda.is_initialized()
