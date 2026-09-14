from probes.owner_successor_scale import data as d


def test_pool_is_fixed_density_balanced_and_excludes_protected_ids():
    rows = [{'image_id': i, 'objects': [None]*n} for i,n in enumerate([1,5,10,20]*10)]
    selected,reused,counts=d.select_pool(rows,[0,1,2],{1,8},size=18)
    reverse=d.select_pool(list(reversed(rows)),[2,1,0],{8,1},size=18)
    assert (selected,reused,counts)==reverse
    assert reused==[0,2]
    assert 1 not in selected and 8 not in selected
    assert len(set(selected))==18
    assert set(counts.values())=={4}


def test_repeat_history_is_any_class_strict_once_per_later_row(monkeypatch):
    class Tokenizer:
        def decode(self,tokens,**kwargs):return str(tokens[1])
    boxes={1:[0,0,100,100],2:[0,0,100,100],3:[0,0,100,100],4:[0,0,100,100]}
    def parse(text,*args):return {'pred':[{'bbox':boxes[int(text)],'description':str(text)}],'dropped_predictions':[]}
    monkeypatch.setattr(d,'native_record',parse)
    rows=[[d.scale.ROW_START,i,0,0,0,0,0,d.scale.ROW_END] for i in [1,2,3,4]]
    ids=sum(rows,[])+[d.e.EOS]
    histories=d.repeat_histories(ids,{'case':{},'golden':{}},Tokenizer())
    assert len(histories)==2
    assert histories[0]['h_token_ids']==rows[0]
    assert histories[1]['h_token_ids']==rows[0]+rows[1]
    assert [h['first_owner']['row_ordinal'] for h in histories]==[0,0]
    boxes[2]=[0,0,95,100]  # Exactly0.95 is not a strict event.
    histories=d.repeat_histories(ids,{'case':{},'golden':{}},Tokenizer())
    assert histories[0]['later_row_ordinal']==2
    assert d.repeat_histories([d.e.EOS],{},Tokenizer())==[]
    assert d.repeat_histories([d.scale.ROW_START,d.scale.ROW_START,d.scale.ROW_END],{},Tokenizer())==[]


def test_nomination_preserves_two_candidates_and_total_budget(monkeypatch):
    class Tokenizer:
        def decode(self,tokens,**kwargs):return 'history'
        def encode(self,text,**kwargs):return [d.scale.ROW_START,1,0,0,0,0,0,d.scale.ROW_END]
    monkeypatch.setattr(d,'repeat_histories',lambda *args:[{'history_index':0,'h_token_ids':[9]*3070}])
    monkeypatch.setattr(d.scale,'_candidate_text',lambda gt:gt['description'])
    monkeypatch.setattr(d,'native_record',lambda text,*args:{'pred':[] if text=='history' else [{'bbox':[0,0,int(text),10]}],'owner':text})
    monkeypatch.setattr(d.scale,'score',lambda parsed,**kwargs:{'50':{'owners':[parsed['owner']]}})
    frozen={'image_id':1,'example_id':'x','case':{},'golden':{'gt':[{'object_id':str(i),'description':str(i)} for i in [1,3,2]]}}
    jobs,_=d.nominate(frozen,{'action_ids':[]},Tokenizer())
    assert [j['owner_id'] for j in jobs]==['3','2']
    assert all(j['remaining_budget']==6 for j in jobs)
