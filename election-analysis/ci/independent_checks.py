# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:a624d0a1-de32-42b1-8bfe-c03f6929240c
"""Separate AI review checks. This file does not modify implementation files.

Uses independent stdlib CSV parsers, direct proper-score formulas and SciPy's
distribution formula as oracles. Production imports are the subjects checked.
"""
from __future__ import annotations
import csv, hashlib, io, json, math, re, sys, tempfile, unicodedata, zipfile
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--original-package',type=Path,required=True)
parser.add_argument('--edt',type=Path,required=True)
args=parser.parse_args()
REPO = args.repo.resolve()
SCOPE = REPO / 'election-analysis'
OUT = args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0] = [str(SCOPE), str(SCOPE/'tests'), str(args.edt.resolve())]
from election_reporting.numerics import beta_binomial_tail, threshold, posterior_details
from election_reporting.analysis import analyze_election
from election_reporting.config import load_config
from election_reporting.publish import analysis_payload, facebook_summary
from election_reporting.replay import run as replay
from test_election_reporting import result_rows, counting_config
import numpy as np
from scipy.stats import betabinom, binom
from scipy.special import logsumexp

def sha(raw): return hashlib.sha256(raw).hexdigest()
def integer(v): return int(v.replace(',', '').strip())
def compact(s): return ' '.join(unicodedata.normalize('NFC', s).split())
def data(name): return json.loads((SCOPE/'docs/engineering'/name).read_text())

result = {'reviewer':'Separate agentic AI statistical reviewer; not implementation agent, human reviewer, certification, or blind preregistration',
          'reviewed_commit':'3e9ae3bee1c657c91629b133f379a5b73e41781a',
          'reviewed_tree':'2ccd8e0a0a08d610e61e0e51938794d57dddecd8',
          'generated_at':datetime.now(timezone.utc).isoformat()}

# First container identity, original audit member identity, descriptive fixture lineage.
q=args.original_package
with zipfile.ZipFile(q) as z:
    assert z.testzip() is None and sha(q.read_bytes())=='06d150c885e9f95aa3235dc4ceaab09f34379bc27f8de66864c00e85af67f73e'
    fixture_lineage={}
    for state in ['washington','alaska','maine']:
        a=z.read(f'historical-2024/replay/{state}/input/2026-10-23-0000.csv')
        b=(SCOPE/f'tests/fixtures/replay-2024/{state}.csv').read_bytes()
        assert a==b
        fixture_lineage[state]=sha(b)
    result['original_package']={'bytes':q.stat().st_size,'sha256':sha(q.read_bytes()),'members':len(z.namelist()),'crc':'PASS',
       'methods_member_sha256':sha(z.read('Election_Analysis_Statistical_Audit_2026-10-23.md')),
       'replay_fixture_original_bytes':fixture_lineage}

# Exact finite beta-binomial distribution, independent direct final-margin inequalities.
checks=0;maximum=0.0
for n in range(1,19):
    for margin in range(6):
        for tie_changes in (False,True):
            for a,b in [(1,1),(1,2),(2,1),(2,3),(4,2)]:
                expected=Fraction(0)
                for k in range(n+1):
                    if (2*k-n>=margin if tie_changes else 2*k-n>margin):
                        expected+=Fraction(math.comb(n,k)*math.factorial(k+a-1)*math.factorial(n-k+b-1)*math.factorial(a+b-1), math.factorial(n+a+b-1)*math.factorial(a-1)*math.factorial(b-1))
                maximum=max(maximum,abs(beta_binomial_tail(n,a,b,threshold(n,margin,tie_changes))-float(expected)))
                checks+=1
assert checks==1080 and maximum<1e-12
result['exact_rational_oracle']={'checks':checks,'maximum_absolute_error':maximum}
result['edge_cases']={'skewed_5001':beta_binomial_tail(5001,1,2,2502),
     'impossible_threshold':beta_binomial_tail(20,1,1,21),'zero_pool':beta_binomial_tail(0,1,1,1),
     'budget_withheld':beta_binomial_tail(200001,1,2,100000),
     'disjoint_posterior_example':posterior_details(500,1500,.1,1000)}

# Parse archived King CSVs without using archives.py, retaining the source identity tuple.
kzip=zipfile.ZipFile(SCOPE/'tests/fixtures/king-archives.zip')
kcache={}
def king_file(name):
    if name in kcache: return kcache[name]
    raw=kzip.read(name)
    try: text=raw.decode('utf-8-sig')
    except UnicodeDecodeError: text=raw.decode('cp1252')
    groups=defaultdict(dict)
    for row in csv.DictReader(io.StringIO(text)):
        key=tuple(row[c] for c in ['GEMS Contest ID','District Type','District Type Subheading','District Name','Ballot Title'])
        groups[key][row['Ballot Response']]=integer(row['Votes'])
    kcache[name]=groups
    return groups

# Parse the independent final CSV and precinct TSV without official_archives.py.
ozip=zipfile.ZipFile(SCOPE/'tests/fixtures/oc-official-2024.zip')
stream=io.StringIO(ozip.read('gen2024-final-summary.csv').decode('utf-8-sig'))
next(stream);next(stream)
ofinal={}
for row in csv.DictReader(stream):
    if row['Total Votes'] is None:
        assert row['Contest Name']=='Total Precincts Reported'
        continue
    name=compact(row['Candidate Name']).lstrip('*')
    party=row['Party']
    if party and name.endswith(f' ({party})'): name=name[:-(len(party)+3)]
    ofinal[(compact(row['Contest Name']),name)]=integer(row['Total Votes'])
media=zipfile.ZipFile(io.BytesIO(ozip.read('gen2024-media.zip')))
precinct=defaultdict(int)
precinct_methods=defaultdict(lambda:[0,0,0])
rows=list(csv.DictReader(io.StringIO(media.read('results-final.txt').decode('utf-8-sig')),delimiter='\t'))
for row in rows:
    key=(compact(row['Contest Title']),compact(row['Choice Name1']))
    precinct[key]+=integer(row['Total Votes'])
    for i,c in enumerate(['Total Votes','Vote Center Votes','Vote by Mail Votes']):
        precinct_methods[key][i]+=integer(row[c])

reports={name:data(f'{name}-scores-'+('v2' if name=='king-retrospective' else 'v1')+'.json') for name in ['king-retrospective','oc-official']}
score_checks={};numerical=[]
for name,report in reports.items():
    all_rows=report['scored_rows']
    base=[r for r in all_rows if r['scenario']=='base']
    outcome_checks=0
    for row in all_rows:
        if name=='king-retrospective':
            identity=tuple(row['source_identity']); final=king_file(row['outcome_file'])[identity]
            asof=king_file(row['input_file'])[identity]
            assert asof[row['current_side']]==row['current_votes'] and asof[row['change_side']]==row['change_votes']
            expected=int(final[row['change_side']]>final[row['current_side']])
        else:
            key=(row['title'],row['current_side']);other=(row['title'],row['change_side'])
            assert ofinal[key]==precinct[key]==row['final_current_votes']
            assert ofinal[other]==precinct[other]==row['final_change_votes']
            expected=int(precinct[other]>precinct[key])
        assert expected==row['outcome'];outcome_checks+=1
    # Explicit three-level weighting, independent of implementation _metrics.
    elections=sorted({r['election_id'] for r in base})
    totals=defaultdict(lambda:defaultdict(list))
    for r in base: totals[r['election_id']][r['canonical_id']].append(r)
    weighted=[(r,1/len(totals)/len(cs)/len(rs)) for cs in totals.values() for rs in cs.values() for r in rs]
    assert abs(sum(w for _,w in weighted)-1)<1e-14
    recomputed={};max_score_error=0.0;max_bin_error=0.0
    for method,reported in report['scores'].items():
        if reported.get('status')=='NOT RUN': continue
        get=lambda r:r['probability'] if method=='model' else r['baselines'][method]
        brier=sum(w*(get(r)-r['outcome'])**2 for r,w in weighted)
        loss=sum(-w*(r['outcome']*math.log(min(1-1e-12,max(1e-12,get(r))))+(1-r['outcome'])*math.log1p(-min(1-1e-12,max(1e-12,get(r))))) for r,w in weighted)
        max_score_error=max(max_score_error,abs(brier-reported['brier']),abs(loss-reported['log_loss']))
        for k,b in enumerate(reported['calibration_bins']):
            selected=[(r,w) for r,w in weighted if (k/10<=get(r)<(k+1)/10) or (k==9 and get(r)==1)]
            assert len(selected)==b['rows']
            if selected:
                mass=sum(w for _,w in selected)
                max_bin_error=max(max_bin_error,abs(sum(w*get(r) for r,w in selected)/mass-b['predicted']),abs(sum(w*r['outcome'] for r,w in selected)/mass-b['observed']))
        recomputed[method]={'brier':brier,'log_loss':loss}
    assert max_score_error<1e-13 and max_bin_error<1e-13
    naive_brier=sum((r['probability']-r['outcome'])**2 for r in base)/len(base)
    per_election={e:{'rows':sum(r['election_id']==e for r in base),'contests':len(totals[e]),'row_reversals':sum(r['outcome'] for r in base if r['election_id']==e)} for e in elections}
    reversal_contests=len({r['canonical_id'] for r in base if r['outcome']})
    score_checks[name]={'base_rows':len(base),'all_rows':len(all_rows),'independent_geographic_clusters':1,'elections':per_election,
       'unique_contests':len({r['canonical_id'] for r in base}),'contests_with_any_reversal':reversal_contests,
       'raw_outcome_checks':outcome_checks,'proper_scores':recomputed,'maximum_score_error':max_score_error,'maximum_calibration_bin_error':max_bin_error,
       'unweighted_row_brier_for_comparison_only':naive_brier}
    # Full direct SciPy mass sums: independent of the production recurrence.
    numerical_count=0;errors=Counter();largest={}
    for r in all_rows:
        n=r['remaining_pair_votes']; first=(n+r.get('margin',r['current_votes']-r['change_votes']))//2+1
        a,b=r['posterior']['alpha'],r['posterior']['beta']
        params=[('model','bb',a,b,r['probability']),('untempered_beta_binomial','bb',1+r['change_votes'],1+r['current_votes'],r['baselines']['untempered_beta_binomial']),('cumulative_share','binomial',r['change_votes']/(r['change_votes']+r['current_votes']),None,r['baselines']['cumulative_share'])]
        if name=='oc-official' and r['conditional_increment_posterior']:
            cp=r['conditional_increment_posterior']
            params.append(('disjoint_increment_conditional','bb',cp['alpha'],cp['beta'],r['baselines']['disjoint_increment_conditional']))
        for method,kind,a,b,actual in params:
            ks=np.arange(first,n+1)
            oracle=0.0 if first>n else float(np.exp(logsumexp(betabinom.logpmf(ks,n,a,b) if kind=='bb' else binom.logpmf(ks,n,a))))
            error=abs(oracle-actual)
            if error>errors[method]: errors[method]=error;largest[method]={'canonical_id':r['canonical_id'],'scenario':r['scenario'],'n':n,'actual':actual,'oracle':oracle}
            assert error<=max(1e-8,abs(oracle)*1e-6),(name,method,error)
            numerical_count+=1
    numerical.append({'experiment':name,'checks':numerical_count,'max_abs_errors':dict(errors),'largest_error_cases':largest})
result['independent_raw_parsers_and_grouped_scores']=score_checks
result['scipy_direct_mass_oracles']=numerical

# Independently check inventory composite keys and units. The final inventory
# supplies no dates or candidate votes; no inference of unique voter ballots.
mraw=(SCOPE/'tests/fixtures/oc-ballot-manifest-2024.csv').read_bytes()
manifest=list(csv.DictReader(io.StringIO(mraw.decode('utf-8-sig'))))
keys=[tuple(r[c] for c in ['Container','Tabulator','Batch Name']) for r in manifest]
assert len(set(keys))==len(manifest)==10208
result['inventory']={'sha256':sha(mraw),'unique_keys':len(set(keys)),
  'source_count_sum':sum(integer(r['Number of Ballots']) for r in manifest),
  'counts_by_container':{c:sum(integer(r['Number of Ballots']) for r in manifest if r['Container']==c) for c in ['in-person','mail']},
  'is_dated_candidate_new_ballot_ledger':False}

# Current real publication: actual failures stay unestimated; preserved sources.
analysis=analyze_election(load_config(SCOPE/'data/elections/washington/2026-08-04/election.toml'))
payload=analysis_payload(analysis)
pub=json.loads((SCOPE/'reports/washington/2026-08-04/analysis.json').read_text())
assert payload['races']==pub['races']
assert facebook_summary(analysis, report_url='https://github.com/dlworrell/code-noodling/blob/main/election-analysis/reports/washington/2026-08-04/election-analysis.md')==(SCOPE/'reports/washington/2026-08-04/facebook-summary.txt').read_text()
assert all(d.risk.change_probability is None for r in analysis.races for d in r.decisions)
result['actual_publication']={'race_representations':len(analysis.races),'decisions':sum(len(r.decisions) for r in analysis.races),'numeric_forecasts':0,'races_semantically_match_committed':True,'social_text_matches':True,'source_counts':dict(Counter(r.source_id for r in analysis.races)),'data_status':dict(Counter(r.data_status for r in analysis.races))}
r=replay(SCOPE/'tests/fixtures')
result['descriptive_replay']={'records':[len(x.races) for x in r],'numeric_probabilities':sum(d.risk.change_probability is not None for a in r for c in a.races for d in c.decisions)}
assert result['descriptive_replay']['records']==[204,73,262]
assert not result['descriptive_replay']['numeric_probabilities']

# Adversarial new-batch cases outside existing two-candidate tests. A truthful
# whole-contest compatibility check should refuse observed negative corrections
# elsewhere in a tally even if its selected pair increased.
adversarial=[]
with tempfile.TemporaryDirectory(dir=OUT) as t:
    root=Path(t);(root/'inputs').mkdir();cfg=root/'election.toml'
    cfg.write_text('schema_version=1\n[election]\ntitle="independent-review-adversarial"\n[[sources]]\nid="local"\npath="inputs"\njurisdiction="Local"\nscope="county"\npriority=1\nexpected_final_ballots=300\n')
    cases=[('third_candidate_negative',150,[('A',80),('B',60),('C',10)],160,[('A',90),('B',65),('C',5)]),
           ('ballots_decrease_while_pair_increases',150,[('A',80),('B',60)],145,[('A',85),('B',60)]),
           ('candidate_removed',150,[('A',80),('B',60),('C',10)],155,[('A',90),('B',65)])]
    for label,old_ballots,old,new_ballots,new in cases:
        (root/'inputs/2026-08-05.csv').write_text(result_rows('Local','9',old_ballots,old))
        (root/'inputs/2026-08-06.csv').write_text(result_rows('Local','9',new_ballots,new))
        c=counting_config(cfg)
        race=analyze_election(c).races[0];d=race.decisions[0]
        adversarial.append({'case':label,'old_ballots':old_ballots,'old_choices':old,'new_ballots':new_ballots,'new_choices':new,
            'expected_latest_pair_votes':0,'actual_latest_pair_votes':d.batch_evidence['latest_verified_pair_votes'],
            'scenario_eligible':race.forecast_eligible,'conditional_probability':d.risk.change_probability,'classification':d.batch_evidence['classification']})
    # Repeated latest bytes after an earlier change: de-duplicating the display
    # history must not change which transition a new-ballot annotation describes.
    (root/'inputs/2026-08-05.csv').write_text(result_rows('Local','9',80,[('A',50),('B',30)]))
    repeated=result_rows('Local','9',100,[('A',60),('B',40)])
    (root/'inputs/2026-08-06.csv').write_text(repeated)
    (root/'inputs/2026-08-07.csv').write_text(repeated)
    c=counting_config(cfg)
    c=replace(c,result_statuses=tuple({(s.source_id,s.snapshot_sha256,s.contest_id):s for s in c.result_statuses}.values()))
    race=analyze_election(c).races[0];d=race.decisions[0]
    adversarial.append({'case':'duplicate_latest_after_prior_increase','expected_latest_pair_votes':0,
        'actual_latest_pair_votes':d.batch_evidence['latest_verified_pair_votes'],'scenario_eligible':race.forecast_eligible,
        'conditional_probability':d.risk.change_probability,'classification':d.batch_evidence['classification'],
        'latest_label':race.snapshot,'displayed_margin_history':d.margin_history})
    (root/'inputs/2026-08-07.csv').unlink()
    (root/'inputs/2026-08-05.csv').write_text(result_rows('Local','9',150,[('A',80),('B',60),('C',10)]))
    (root/'inputs/2026-08-06.csv').write_text(result_rows('Local','9',160,[('A',90),('B',65),('C',5)]))
    c=counting_config(cfg);c.audit_contract['rules'][0].update(rule='wa_judicial',positions=1)
    race=analyze_election(c).races[0]
    d=next(x for x in race.decisions if x.kind=='majority_status')
    adversarial.append({'case':'majority_negative_third_candidate','expected_latest_pair_votes':0,
       'actual_latest_pair_votes':d.batch_evidence['latest_verified_pair_votes'],'scenario_eligible':race.forecast_eligible,
       'conditional_probability':d.risk.change_probability,'classification':d.batch_evidence['classification']})
result['batch_adversarial_checks']=adversarial

(OUT/'independent-check-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'rational_checks':checks,'score_checks':score_checks,'scipy':numerical,'actual_publication':result['actual_publication'],'replay':result['descriptive_replay'],'batch_adversarial_checks':adversarial},indent=2))
