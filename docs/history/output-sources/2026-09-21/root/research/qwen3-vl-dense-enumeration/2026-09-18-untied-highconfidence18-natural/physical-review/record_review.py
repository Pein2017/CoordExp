from pathlib import Path
import json,collections
R=Path(__file__).parent;events=json.loads((R/'events.json').read_text())['events']
notes=[
('HOLD','Tiny bus-window person: trusted positive; normalized nearest overlap is the bus, not a person. No credible retained person witness, but visibility is too small for stronger physical-loss adjudication.'),
('HOLD','Same tiny bus-window person; nearest normalized bus is not owner retention. Physical omission remains unresolved at this view.'),
('retained_owner_extent_change','Both boxes designate the same visible tie; shifted narrow extent changes IoU matching.'),
('HOLD','Nearest normalized person is a different foreground girl from the rightmost taller target. Consistent with loss, not an exhaustive absence proof.'),
('credible_local_omission','Visible tie had a specific original row; normalized nearest overlap is a whole person, not a retained tie row.'),
('retained_owner_extent_change','Both boxes designate the same background person; increased height changes known matching.'),
('HOLD','Clipped rightmost person and broad normalized box mix body/context and neighboring person; extent versus identity unresolved.'),
('retained_owner_extent_change','Same central adult is present in the large original box and tighter normalized box; known gain is an extent correction.'),
('HOLD','Tiny distant people overlap; shifted boxes may designate adjacent people or changed extent.'),
('HOLD','Repeated thin shoreline strip has no defensible single-person attribution; exact/IoU repeat only.'),
('HOLD','Trusted tiny distant person is localized by normalized output; limited pixels and prior coverage prevent stronger physical-gain claim.'),
('retained_owner_extent_class_change','Same visible container is retained with a shorter box; both predictions say cup whereas trusted category is bottle. Not physical disappearance.'),
('credible_local_new_owner','Normalized row localizes foreground red chair; nearest original row localizes the distinct light chair behind. Bounded local gain witness, not a full absence census.'),
('HOLD','Repeated thin book-stack region; individual volume boundaries/partial extent remain ambiguous.'),
('HOLD','Book extent shrinks toward neighboring spine; identity versus extent change unresolved.'),
('HOLD','Repeated book-stack box may include adjacent volumes; no unique physical-owner count.'),
('retained_owner_extent_change','Both rows designate the same lower visible book; wider normalized extent changes match.'),
('credible_local_new_owner','Normalized box localizes the visible dark backpack; closest original row is the distinct red folding chair.'),
('retained_owner_extent_change','Same visible window passenger is retained; shifted/wider bbox crosses matching threshold.'),
('retained_owner_extent_change','Same blue-clothed person is retained, with changed horizontal extent/context.'),
('HOLD','Top-left bottle-shaped sliver is not sufficiently resolved for single-owner identity; repeat remains proxy.'),
('HOLD','Top-left repeated person box is a background sliver, not a defensibly identified person.'),
('credible_local_new_owner','Normalized row identifies a visible background bottle at trusted location. No overlapping original owner row in this local candidate selection.'),
('HOLD','Repeated partial donut region overlaps a crowded neighboring extent; preserve earlier ambiguity rather than force an identity.'),
('HOLD','Normalized nearest box moves to adjacent donut below; target omission versus partial/assignment route remains unresolved.'),
('HOLD','Repeated donut box crosses adjacent pastry/reflection extents; dominant single-owner attribution unresolved.'),
('retained_region_extent_change_HOLD_identity','Original thin strip touches target donut region; normalized box captures its full extent. Cannot call this confidently a new physical owner.'),
('confirmed_owner_recurrence_with_extent_debt','Same leftmost foreground chair back is visibly repeated; box also includes seated person/context, so extent remains imperfect.'),
('HOLD','Full-width lower audience region contains many chairs/people; literal valid-box recurrence is not one repeated owner.'),
('retained_region_extent_change_HOLD_identity','Chair behind speaker is partially occluded; both boxes localize overlapping chair/person region, with narrower normalized extent. No confident physical-loss claim.')]
assert len(events)==len(notes)==30
records=[]
for event,(status,note) in zip(events,notes):records.append(dict(key=f"{event['image_id']}:{event['model']}:{event['kind']}",image_id=event['image_id'],model=event['model'],kind=event['kind'],status=status,note=note,plot=event['plot'],context=event['context']))
(R/'review-phase1.json').write_text(json.dumps(dict(status='candidate_bounded_review',events=records,counts=dict(collections.Counter(x['status'] for x in records)),scope='17 ready refined images; bird309264 pending. Thirty selected events, full-image plus local context, no proposal census. Unknown remains unknown; physical claims do not replace frozen numerical scores.'),indent=2)+'\n')
