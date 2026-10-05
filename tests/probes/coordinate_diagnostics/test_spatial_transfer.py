"""Model-free entry, literal selection and exhaustive transfer presentation."""
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
from probes.coordinate_diagnostics import spatial_transfer as probe


def encode(tokenizer,description,box):
    text='<|object_ref_start|>'+description+'<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{v}|>' for v in box)+'<|box_end|>'
    return tokenizer.encode(text,add_special_tokens=False)


class FixtureModel(fixtures.SyntheticModel):
    def generate(self,**kwargs):
        ids=kwargs['input_ids'].clone();raw=[];processed=[];cell=self.cell;incoming=torch.get_num_threads()
        if cell['condition']=='baseline':
            assert cell['forced_actions']=={} and kwargs['max_new_tokens']==3084
            emitted=self.baseline_ids
        elif cell['condition']=='anchor':emitted=self.anchor['reference_median_winners']
        else:emitted=cell['expected_ids']+[probe.COORD_START,probe.COORD_START+30,probe.COORD_START+min(999,cell['x1']+20),probe.COORD_START+100,self.close_id]
        for pos,winner in enumerate(emitted):
            scores=torch.full((1,probe.COORD_START+1000),-12.,dtype=torch.float32)
            if cell['condition']=='anchor' and self.break_anchor and pos==0:winner+=1
            if cell['query_index']==31 and pos==cell['position']+2:winner=probe.COORD_START
            if cell['query_index']==30 and pos==cell['position']+2:winner=kwargs['eos_token_id']
            scores[0,winner]=8
            raw.append(scores);decision=kwargs['logits_processor'](ids,scores);processed.append(decision)
            assert torch.get_num_threads()==incoming
            token=decision.argmax(-1).reshape(1,1);ids=torch.cat((ids,token),1)
            if int(token)==kwargs['eos_token_id']:break
        return SimpleNamespace(sequences=ids,logits=tuple(raw),scores=tuple(processed))


class FixtureSession(probe.NativeSession):
    q_frontend=None
    break_anchor=False
    baseline_mode='ordinary'
    loads=[]

    def __init__(self,checkpoint,packet):
        q=replace(self.q_frontend,model=FixtureModel());self.loads.append(checkpoint.name)
        with patch('probes.rule_stability.runner.native_components',return_value=(q,None,dict(compute='CPU_FIXTURE',research_model_loaded=False,research_forwards=0))):
            super().__init__(checkpoint,packet)

    def generate(self,cell):
        tokenizer=self.q.tokenizer;model=self.q.model
        model.cell=cell;model.anchor=self.packet['conditions'][0];model.break_anchor=self.break_anchor
        model.close_id=tokenizer.convert_tokens_to_ids('<|box_end|>')
        if self.baseline_mode=='absent':model.baseline_ids=encode(tokenizer,'person',[0,0,800,900])+[self.packet['eos_id']]
        else:
            # Two equal person occurrences are not suppressed; first bottle may be malformed later.
            model.baseline_ids=(encode(tokenizer,'person',[0,0,800,900])+encode(tokenizer,'bottle',[100,30,125,100])+
                encode(tokenizer,'person',[0,0,800,900])+encode(tokenizer,'cup',[800,50,750,100])+
                tokenizer.encode('<|object_ref_start|>cat<|object_ref_end|><|box_start|>',add_special_tokens=False)+[self.packet['eos_id']])
        autocast=torch.autocast
        with patch('torch.autocast',side_effect=lambda *args,**kwargs:autocast('cpu',enabled=False)):
            return super().generate(cell)

    def close(self):self.q=self.norm=self.delta=self.batches=None


@pytest.fixture(scope='module')
def prepared():
    from probes import rollout_row_credit as retained
    q=retained.frontend();assert q.model is None;FixtureSession.q_frontend=q
    folder=Path(os.environ['SPATIAL_TRANSFER_CPU_OUTPUT'])/'prepared'
    if not folder.exists():assert probe.main(['prepare','--output',str(folder)])==0
    _,packet=probe.load_packet(folder/'native-proposal.json',cpu=True)
    return folder,packet


def test_actual_entry_publication_fresh_consumer_gallery(prepared):
    folder,packet=prepared;output=folder.parent/'cpu-positive';FixtureSession.loads=[]
    assert probe.main(['run','--config',str(folder/'native-proposal.json'),'--output',str(output)],cpu_factory=FixtureSession)==0
    report=probe.a.load(output/'readback.json');terminal=probe.a.load(output/'terminal.json')
    assert report['complete'] and report['completed_requests']==34 and report['requested_candidates']==32
    assert report['requested_denominators']==dict(anchors=1,ordinary_baselines=1,grid_candidates=32)
    assert report['entrance']['position']==14 and report['entrance']['row_start']==9
    assert report['actual_forced_actions']==494
    assert FixtureSession.loads==['checkpoint-16'] and terminal['counts']['checkpoint_loads']==0
    baseline=probe.a.load(output/'baseline.json')
    assert baseline['forced_actions']=={} and 'score_tensors' not in baseline and 'raw_steps' not in baseline
    assert report['baseline']['categories']==dict(person=2,bottle=1,cup=1)
    assert len(report['baseline']['groups'])==3 and report['baseline']['groups'][0]['multiplicity']==2
    assert len(report['baseline']['malformed'])==1
    assert report['conditions'][-1]['analysis']['selected_row']['box']==[984,30,0,100]
    assert report['conditions'][-2]['analysis']['selected_row'] is None
    assert all(len(r['analysis']['selected_row']['bottle_annotation_overlaps'])==13 for r in report['conditions'][2:]
               if r['analysis']['selected_row'] and r['analysis']['selected_row']['valid'])
    assert all('target_box' not in r['cell'] for r in report['conditions'])
    fresh=subprocess.run([sys.executable,'-m','probes.coordinate_diagnostics.spatial_transfer','readback','--output',str(output)],
        cwd=probe.ROOT,capture_output=True,text=True)
    (folder.parent/'fresh-consumer.log').write_text(fresh.stdout+fresh.stderr)
    assert fresh.returncode==0,fresh.stderr
    semantic=copy.deepcopy(report);semantic.pop('compute');assert json.loads(fresh.stdout)==semantic
    other=int(next(x.split('=',1)[1] for x in fresh.stderr.splitlines() if x.startswith('spatial_transfer_cpu_threads_incoming=')))
    incoming=probe.a.load(output/'invocation.json')['cpu_threads_incoming']
    assert incoming==2 and other!=incoming and not torch.cuda.is_initialized()
    probe.a.write(folder.parent/'semantic-report-equality.json',dict(exact=True,producer_incoming_threads=incoming,
        fresh_consumer_incoming_threads=other,only_removed_field='compute',research_forwards=0,CUDA_initialized=False))
    assert probe.main(['gallery','--output',str(output)])==0
    index=probe.a.load(output/'gallery/index.json')
    assert index['candidate_slots']==32 and index['baseline_groups']==3 and index['baseline_occurrences']==4
    assert len(index['manifests'])==36 and index['malformed_entries']==1
    invalid=probe.a.load(output/'gallery/grid-31.json')['prediction'][0]
    assert invalid['source_bbox']==[984,30,0,100] and not invalid['geometry_valid']
    assert invalid['bbox_pixel_xyxy']==[850,35,0,115]
    assert probe.a.load(output/'gallery/grid-30.json')['prediction']==[]
    assert all('original_slot' not in probe.a.load(m['path']) and 'gt' not in probe.a.load(m['path']) for m in index['manifests'])
    with pytest.raises(FileExistsError):probe.run(folder/'native-proposal.json',output,cpu_factory=FixtureSession)


@pytest.mark.parametrize('case',['eos','malformed','absent','late','header-at-cap','crossed-boundary'])
def test_earliest_literal_header_causal_selection(prepared,case):
    _,p=prepared;t=FixtureSession.q_frontend.tokenizer;person=encode(t,'person',[0,0,10,20]);header=t.encode(probe.HEADER,add_special_tokens=False)
    prefix=person+header
    if case=='eos':ids=prefix+[p['eos_id']]
    if case=='malformed':ids=prefix+t.encode('broken',add_special_tokens=False)+encode(t,'bottle',[10,20,30,40])
    if case=='absent':ids=person+[p['eos_id']]
    if case=='late':ids=person*8+header+[p['eos_id']]
    if case=='header-at-cap':ids=prefix
    if case=='crossed-boundary':
        # Last action closes box_start and spells a following x1 piece: geometry rejects this frame.
        original_tokenizer=t
        class Crossing:
            def get_added_vocab(self):return original_tokenizer.get_added_vocab()
            def decode(self,ids,skip_special_tokens=False):
                return original_tokenizer.decode(ids[:-1],skip_special_tokens=False)+('<|box_start|><' if ids[-1]==1784 else original_tokenizer.decode(ids[-1:],skip_special_tokens=False)) if ids else ''
        ids=prefix[:-1]+[1784];t=Crossing()
    text=t.decode(ids,skip_special_tokens=False);record=dict(token_ids=ids,text=text)
    result=probe.first_header(record,t)
    if case in ('absent','late','crossed-boundary'):assert result['status']=='unavailable'
    else:
        assert result['position']==len(prefix) and result['row_start']==len(person)
        assert result['prefix_token_ids']==prefix
        if case=='eos':assert result['next_native_action']==p['eos_id']
        if case=='header-at-cap':assert result['next_native_action'] is None


@pytest.mark.parametrize('mutation',['grid','anchor','order','source-label','selection','frame','request','context','bounds','exposure'])
def test_frozen_input_identity_has_teeth(prepared,mutation):
    _,p=prepared;p=copy.deepcopy(p)
    if mutation=='grid':p['conditions'][2]['x1']+=1
    if mutation=='anchor':p['conditions'][0]['forced_actions']['0']+=1
    if mutation=='order':p['conditions'][2],p['conditions'][3]=p['conditions'][3],p['conditions'][2]
    if mutation=='source-label':p['images']['25394']['objects'][0]['bbox_2d'][0]+=1
    if mutation=='selection':p['selection']['eligible'].pop()
    if mutation=='frame':p['images']['25394']['width']+=1
    if mutation=='request':p['requests']['25394']['image_grid_thw']=[1,1,1]
    if mutation=='context':p['processor']['25394']['prompt_token_ids']*=10
    if mutation=='bounds':p['bounds']['requests']+=1
    if mutation=='exposure':next(iter(p['exposure'].values()))['text']+='changed'
    with pytest.raises(ValueError):probe.verify_inputs(p)


def saved(prepared,name='grid-00'):
    folder,p=prepared;r=probe.a.load(folder.parent/'cpu-positive'/(name+'.json'))
    entrance=probe.a.load(folder.parent/'cpu-positive/entrance.json')
    cell=probe.query_cell(next(c for c in p['conditions'] if c['condition']==name),entrance)
    return r,cell,p


@pytest.mark.parametrize('mutation',['summary','vector','prefix','action','role','schema','frame'])
def test_original_vector_action_summary_reject(prepared,mutation):
    r,c,p=saved(prepared);r=copy.deepcopy(r)
    if mutation=='summary':r['raw_observations'][0]['full_log_normalizer']+=1e-6
    if mutation=='vector':r['raw_observations'][1]['vector_key']='median_15'
    if mutation=='prefix':r['median_observations'][1]['prefix_token_ids'][-1]+=1
    if mutation=='action':r['raw_steps'][15]['requested_token_id']+=1
    if mutation=='role':r['median_observations'][1]['position']+=1
    if mutation=='schema':r['schema']=probe.grid.SCHEMA
    if mutation=='frame':r['height']+=1
    with pytest.raises(ValueError):probe.validate_record(r,c,p,FixtureSession.q_frontend.tokenizer)


def test_optional_gpu_reduction_and_scope_restore(prepared):
    r,c,p=saved(prepared);r=copy.deepcopy(r);before=torch.get_num_threads()
    r['raw_observations'][0]['native_device_full_log_normalizer']+=10
    probe.validate_record(r,c,p,FixtureSession.q_frontend.tokenizer)
    with probe.cpu_diagnostics():assert torch.get_num_threads()==1
    assert torch.get_num_threads()==before
    with pytest.raises(RuntimeError):
        with probe.cpu_diagnostics():
            assert torch.get_num_threads()==1;raise RuntimeError('intentional')
    assert torch.get_num_threads()==before
    folder,_=prepared
    with pytest.raises(ValueError,match='exact lead release'):probe.load_packet(folder/'native-proposal.json',cpu=False)
    assert not torch.cuda.is_initialized()


def test_failed_anchor_holds_baseline_and32_before_acquisition(prepared,monkeypatch):
    folder,p=prepared;monkeypatch.setattr(FixtureSession,'break_anchor',True)
    output=folder.parent/'failed-anchor'
    assert probe.run(folder/'native-proposal.json',output,cpu_factory=FixtureSession)==2
    report=probe.a.load(output/'readback.json')
    assert report['completed_requests']==1 and len(report['conditions'])==34
    assert not report['anchor_qualified'] and report['baseline'] is None
    assert all(r['status']=='skipped-HOLD' for r in report['conditions'][1:])


def test_absent_header_retains32_unavailable_nongating(prepared,monkeypatch):
    folder,p=prepared;monkeypatch.setattr(FixtureSession,'baseline_mode','absent')
    output=folder.parent/'absent-header'
    assert probe.run(folder/'native-proposal.json',output,cpu_factory=FixtureSession)==0
    report=probe.a.load(output/'readback.json')
    assert report['complete'] and report['completed_requests']==2 and report['requested_candidates']==32
    assert all(r['status']=='unavailable' for r in report['conditions'][2:])


def test_dynamic_budgets_and_slot_corruption(prepared,monkeypatch):
    folder,p=prepared;prefix=list(range(64));entrance=dict(status='eligible',position=64,row_start=59,prefix_token_ids=prefix)
    cells=[probe.query_cell(s,entrance) for s in p['conditions'][2:]]
    assert sum(c['budget'] for c in cells)+3084+19==5311
    assert sum(len(c['forced_actions']) for c in cells)+14==2094
    assert all(c['observations']==[64,65,66,67] and c['budget']==69 for c in cells)
    output=folder.parent/'cpu-positive';entries=probe.a.load(output/'conditions.json');load=probe.a.load
    monkeypatch.setattr(probe.a,'load',lambda path:entries[:-1] if Path(path)==output/'conditions.json' else load(path))
    with pytest.raises(ValueError,match='slots'):probe.consume(output,p,FixtureSession.q_frontend.tokenizer)


def test_malformed_and_censored_original_action_spans(prepared):
    _,p=prepared;t=FixtureSession.q_frontend.tokenizer
    prefix=encode(t,'cat',[0,0,10,20]);header=t.encode(probe.HEADER,add_special_tokens=False)
    for suffix,stop in [([p['eos_id']],'im_end'),(t.encode('broken',add_special_tokens=False),'length')]:
        ids=prefix+header+suffix
        record=dict(token_ids=ids,text=t.decode(ids,skip_special_tokens=False),stop_reason=stop,
                    request_id=p['requests']['25394']['request_id'],width=864,height=1152)
        with probe.cpu_diagnostics():analysis=probe.baseline_analysis(record,p,t)
        tail=analysis['malformed'][0]
        assert tail['positions']==list(range(len(prefix),len(ids))) and tail['original_token_ids']==ids[len(prefix):]
        assert tail['box'] is None and tail['censored']==(stop=='length')


def test_maximum_complete_rows_all_categories_and_gallery_budget(prepared):
    folder,p=prepared;t=FixtureSession.q_frontend.tokenizer;output=folder.parent/'maximum-row-gallery';output.mkdir()
    # Canonical minimum: four wrappers + one nonempty description action + four coordinates.
    assert len(encode(t,'cat',[0,0,10,20]))==9 and len(encode(t,'bottle',[0,0,10,20]))==10
    ids=[]
    for i in range(342):
        description=('cat','person','cup')[i%3]
        box=[i,20,i+1,100] if i!=341 else [900,100,100,20]
        row=encode(t,description,box);assert len(row)==9;ids+=row
    ids+=t.encode('<|object_ref_start|>bottle<|object_ref_end|><|box_start|><|coord_5|>',add_special_tokens=False)
    assert len(ids)==3084
    record=dict(token_ids=ids,text=t.decode(ids,skip_special_tokens=False),stop_reason='length',
                request_id=p['requests']['25394']['request_id'],width=864,height=1152)
    with probe.cpu_diagnostics():b=probe.baseline_analysis(record,p,t)
    assert b['complete_rows']==342 and len(b['groups'])==342 and b['categories']==dict(cat=114,person=114,cup=114)
    assert not b['groups'][-1]['valid'] and b['groups'][-1]['bottle_annotation_overlaps'] is None
    assert len(b['malformed'])==1 and b['malformed'][0]['censored']
    report=copy.deepcopy(probe.a.load(folder.parent/'cpu-positive/readback.json'));report['baseline']=b
    probe.a.write(output/'baseline.json',record);report['conditions'][1]['artifact']=probe.binding(output/'baseline.json')
    probe.a.write(output/'readback.json',report);probe.a.write(output/'terminal.json',dict(readback=probe.binding(output/'readback.json')))
    result=probe.galleries(output,p)
    assert len(result['manifests'])==342+32+1 and result['baseline_groups']==342 and result['baseline_occurrences']==342
    assert result['usage']['retained_bytes']<probe.BOUNDS['retained_bytes'] and result['seconds']<300
    from PIL import Image
    for overview in result['overviews']:
        with Image.open(overview['path']) as im:assert im.width==1800 and im.height<=1840
    for manifest in result['manifests']:
        m=probe.a.load(manifest['path']);assert m['full_label_pool']==result['full_label_pool']
        assert Path(m['image']['path']).exists()
    probe.a.write(folder.parent/'maximum-gallery-measurements.json',dict(minimum_row_actions=9,complete_rows=342,
        query_slots=32,malformed_slots=1,views=len(result['manifests']),seconds=result['seconds'],usage=result['usage']))
