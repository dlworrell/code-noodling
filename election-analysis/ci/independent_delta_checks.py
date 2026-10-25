# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:540d32f9-782a-4a75-89c3-249e27ef15c9
"""Independent delta review; keeps the original failed-snapshot evidence intact."""
import argparse,hashlib,json,math,sys,tempfile
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--edt',type=Path,required=True)
a=p.parse_args();s=a.repo/'election-analysis';base=a.base/'election-analysis'
sys.path[:0]=[str(s),str(s/'tests'),str(a.edt)]
from election_reporting.analysis import analyze_election
from election_reporting.config import load_config
from election_reporting.publish import analysis_payload,facebook_summary
from election_reporting.replay import run as replay
from test_election_reporting import result_rows,counting_config
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
m=json.loads(a.manifest.read_text());candidate_files=m.get('files',m.get('source_and_publication_files'))
assert all(sha(a.repo/f)==h for f,h in candidate_files.items())
results=[]

def make(rows,rule='winner',mutate=None):
    with tempfile.TemporaryDirectory() as t:
        r=Path(t);(r/'inputs').mkdir();c=r/'election.toml'
        c.write_text('schema_version=1\n[election]\ntitle="independent delta oracle"\n[[sources]]\nid="local"\npath="inputs"\njurisdiction="Local"\nscope="county"\npriority=1\nexpected_final_ballots=3000\n')
        for i,text in enumerate(rows,5):(r/f'inputs/2026-08-{i:02}.csv').write_text(text)
        cfg=counting_config(c)
        cfg=replace(cfg,result_statuses=tuple({(x.source_id,x.snapshot_sha256,x.contest_id):x for x in cfg.result_statuses}.values()))
        cfg.audit_contract['rules'][0].update(rule=rule,positions=1)
        if mutate:cfg=mutate(cfg)
        return next(x for x in analyze_election(cfg).races if x.contest_id=='9')

def row(ballots,choices):return result_rows('Local','9',ballots,choices)
old=row(150,[('A',80),('B',60),('C',10)])
bad=[('third_candidate_negative',row(160,[('A',90),('B',65),('C',5)])),
     ('ballots_decrease_while_pair_increases',row(145,[('A',85),('B',60),('C',0)])),
     ('candidate_removed',row(155,[('A',90),('B',65)])),
     ('candidate_added',row(170,[('A',90),('B',65),('C',10),('D',5)])),
     ('observed_votes_exceed_ballot_increment',row(160,[('A',90),('B',70),('C',10)]))]
for rule in ['winner','wa_judicial']:
    for name,current in bad:
        race=make([old,current],rule)
        assert race.evidence['quarantined'] and not race.forecast_eligible
        assert not race.scientific_eligibility['checks']['batch_annotation_consistent']
        assert all(x.batch_evidence['latest_verified_pair_votes']==0 and x.risk.change_probability is None for x in race.decisions)
        results.append({'case':name,'rule':rule,'status':'PASS','latest_votes':[x.batch_evidence['latest_verified_pair_votes'] for x in race.decisions],
                        'quarantined':True,'probabilities_withheld':True,'reasons':race.evidence['batch_transition']['reasons']})

# Exact original decreasing-denominator counterexample, without a third choice.
race=make([row(150,[('A',80),('B',60)]),row(145,[('A',85),('B',60)])])
assert not race.forecast_eligible and race.decisions[0].batch_evidence['latest_verified_pair_votes']==0
results.append({'case':'exact_original_two_choice_denominator','status':'PASS','quarantined':True,'latest_votes':0})

older=row(80,[('A',50),('B',30)]);current=row(100,[('A',60),('B',40)])
for rule in ['winner','wa_judicial']:
    race=make([older,current,current],rule)
    assert race.forecast_eligible and not race.evidence['quarantined']
    assert all(x.batch_evidence['latest_verified_pair_votes']==0 for x in race.decisions)
    assert race.evidence['batch_transition']['previous_snapshot_sha256']==race.evidence['batch_transition']['current_snapshot_sha256']
    results.append({'case':'duplicate_latest_after_prior_increase','rule':rule,'status':'PASS','latest_votes':0,
                    'cumulative_scenario_retained':True,'transition_hashes':race.evidence['batch_transition']})

for rule in ['winner','wa_judicial']:
    middle=row(90,[('A',55),('B',35)])
    def changed_stage(cfg):
        target=cfg.audit_contract['source_versions'][1]['snapshot_sha256']
        return replace(cfg,result_statuses=tuple(replace(x,status=replace(x.status,result_version='other-intermediate-version')) if x.snapshot_sha256==target else x for x in cfg.result_statuses))
    for label,rows,change in [('missing_immediate',[older,result_rows('Other','99',90,[('A',55),('B',35)]),current],None),
                              ('incompatible_immediate',[older,middle,current],changed_stage)]:
        race=make(rows,rule,change)
        assert all(x.batch_evidence['latest_verified_pair_votes']==0 for x in race.decisions)
        assert race.evidence['batch_transition']['classification']!='new_ballots'
        results.append({'case':label,'rule':rule,'status':'PASS','latest_votes':0,'reasons':race.evidence['batch_transition']['reasons']})

def counters(text,over,under):
    lines=text.splitlines();return lines[0]+',"Contest Overvotes","Contest Undervotes"\n'+'\n'.join(x+f',"{over}","{under}"' for x in lines[1:])+'\n'
for label,current in [('known_under_counter_decreases',counters(row(170,[('A',90),('B',65),('C',10)]),0,5)),
                      ('known_counter_availability_changes',row(170,[('A',90),('B',65),('C',10)]))]:
    race=make([counters(old,0,10),current])
    assert race.evidence['quarantined'] and not race.forecast_eligible
    assert race.decisions[0].batch_evidence['latest_verified_pair_votes']==0
    results.append({'case':label,'status':'PASS','latest_votes':0,'quarantined':True})

for field,value in [('coverage_complete',False),('source_published_at',None),('source_published_at','2026-08-06T00:00:00Z')]:
    def change(cfg,field=field,value=value):cfg.audit_contract['source_versions'][0][field]=value;return cfg
    race=make([older,current],mutate=change)
    assert race.decisions[0].batch_evidence['latest_verified_pair_votes']==0
    results.append({'case':'insufficient_prior_evidence','field':field,'value':value,'status':'PASS','latest_votes':0})
for declared in ['unknown','correction','transfer','recount']:
    def change(cfg,declared=declared):cfg.audit_contract['source_versions'][-1]['delta_kind']=declared;return cfg
    race=make([older,current],mutate=change)
    assert race.decisions[0].batch_evidence['latest_verified_pair_votes']==0
    results.append({'case':'unverified_or_nonnew_annotation','declared':declared,'status':'PASS','latest_votes':0})

# Positive control: independently calculated fractional-power alpha/beta values.
old=row(1500,[('A',800),('B',600),('C',100)]);new=row(1700,[('A',900),('B',650),('C',150)])
race=make([old,new]);d=race.decisions[0]
assert race.forecast_eligible and d.batch_evidence['latest_verified_pair_votes']==150
assert math.isclose(d.batch_evidence['alpha'],1+600/1400*300+50,abs_tol=1e-12)
assert math.isclose(d.batch_evidence['beta'],1+800/1400*300+100,abs_tol=1e-12)
results.append({'case':'positive_complete_pair','status':'PASS','latest_votes':150,'alpha':d.batch_evidence['alpha'],'beta':d.batch_evidence['beta']})
race=make([old,new],'wa_judicial');d=next(x for x in race.decisions if x.kind=='majority_status')
assert race.forecast_eligible and d.batch_evidence['latest_verified_pair_votes']==200
assert math.isclose(d.batch_evidence['alpha'],241,abs_tol=1e-12) and math.isclose(d.batch_evidence['beta'],261,abs_tol=1e-12)
assert d.batch_evidence['overlap_votes_reused']==0
results.append({'case':'positive_complete_majority','status':'PASS','latest_votes':200,'alpha':241,'beta':261})

real=analyze_election(load_config(s/'data/elections/washington/2026-08-04/election.toml'))
actual=analysis_payload(real)['races'];original=json.loads((base/'reports/washington/2026-08-04/analysis.json').read_text())['races']
def strip_added(v,parent=None):
    if isinstance(v,dict):
        return {k:strip_added(x,k) for k,x in v.items() if not (k=='batch_transition' and parent=='evidence' or k=='transition' and parent=='batch_evidence' or k=='batch_annotation_consistent' and parent=='checks')}
    if isinstance(v,list):return [strip_added(x,parent) for x in v]
    return v
assert strip_added(actual)==original
assert len(real.races)==298 and all(d.risk.change_probability is None for r in real.races for d in r.decisions)
committed=json.loads((s/'reports/washington/2026-08-04/analysis.json').read_text())
original_full=json.loads((base/'reports/washington/2026-08-04/analysis.json').read_text())
assert committed['races']==actual
assert {k:strip_added(v,k) for k,v in committed.items() if k!='generated_at'}=={k:v for k,v in original_full.items() if k!='generated_at'}
report_url='https://github.com/dlworrell/code-noodling/blob/main/election-analysis/reports/washington/2026-08-04/election-analysis.md'
assert facebook_summary(real,report_url=report_url)==(s/'reports/washington/2026-08-04/facebook-summary.txt').read_text()
for filename in ['election-analysis.md','election-analysis.html','facebook-summary.txt']:
    text=(s/'reports/washington/2026-08-04'/filename).read_text()
    assert '298 source representations of contests' in text
    assert 'overlapping jurisdictions may appear separately' in text
rep=replay(s/'tests/fixtures');assert [len(x.races) for x in rep]==[204,73,262]
assert not any(d.risk.change_probability is not None for v in rep for r in v.races for d in r.decisions)
unchanged={}
for directory in ['tests/fixtures','data','docs/engineering']:
    for f in sorted((base/directory).rglob('*')):
        if f.is_file() and (directory!='docs/engineering' or any(x in f.name for x in ['king-retrospective','oc-official','oc-runtime'])):
            other=s/f.relative_to(base)
            assert other.is_file() and f.read_bytes()==other.read_bytes(),str(f)
            unchanged[str(f.relative_to(base))]=sha(f)
for name in ['numerics.py','rcv.py','retrospective.py','official_validation.py','validation.py','official_archives.py']:
    assert (s/'election_reporting'/name).read_bytes()==(base/'election_reporting'/name).read_bytes()
    unchanged['election_reporting/'+name]=sha(s/'election_reporting'/name)
out={'review_scope':'Hash-bound uncommitted repair candidate;initial snapshot review retained separately.',
     'generated_at':datetime.now(timezone.utc).isoformat(),'base_commit':'3e9ae3bee1c657c91629b133f379a5b73e41781a',
     'candidate_files':candidate_files,'independent_cases':results,'case_count':len(results),'original_five_counterexamples_resolved':True,
     'primary_representations':298,'primary_numeric_forecasts':0,'primary_original_semantics_match_after_only_additive_evidence':True,
     'entire_published_json_matches_base_after_only_additive_evidence_and_generated_clock':True,
     'regenerated_primary_races_and_social_text_match':True,'representation_labels_confirmed_all_human_formats':True,
     'only_removed_comparison_fields':['evidence.batch_transition','decisions.batch_evidence.transition','scientific_eligibility.checks.batch_annotation_consistent'],
     'descriptive_replay_records':[204,73,262],'immutable_files_verified':unchanged,
     'scientific_status':'Experimental and uncalibrated;no data,caps,targets,frozen predictions or scores changed.'}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='immutable_files_verified'},indent=2))
