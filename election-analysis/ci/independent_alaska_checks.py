# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:fa4b6036-7d94-4b19-bc96-f6f70913c6df
"""Independent source-key/mode aggregation and full frozen Alaska comparison."""
import argparse,csv,io,json,zipfile,hashlib,tempfile,re,sys
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--original-package',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();s=a.repo/'election-analysis';sys.path.insert(0,str(s))
from election_reporting.archives import alaska_medsl
from election_reporting.watch.alaska import load_crosswalk,resolve_places
from election_reporting.ingest import read_snapshot
with zipfile.ZipFile(a.original_package) as z:raw=z.read('historical-2024/alaska-medsl.csv')
cross=load_crosswalk(s/'data');outputs,evidence=alaska_medsl(raw,cross,resolve_places(cross,(),True))
seen={};repeats=0;modes=Counter();conflicts=set()
for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
    k=tuple(row[c] for c in ('precinct','office','district','mode','candidate'));v=int(row['votes'])
    if k in seen: assert seen[k][1]==0 or v==0;repeats+=1
    if k not in seen or v:seen[k]=(row,v)
for row,v in seen.values():
    modes[row['mode']]+=v
    if row['mode']=='ELECTION DAY' and v and not re.fullmatch(r'\d{3}-\d{3}',row['precinct']):conflicts.add(row['precinct'])
assert dict(modes)==evidence['mode_contributions']['alaska']
assert repeats==evidence['zero_filled_repetitions']==2788
assert sorted(conflicts)==evidence['district_rows_mislabeled_election_day']
with tempfile.TemporaryDirectory() as t:
    q=Path(t)/'2026-10-25.csv';q.write_text(outputs['alaska']);actual=read_snapshot(q,'alaska')
    q2=Path(t)/'2026-10-23.csv';q2.write_bytes((s/'tests/fixtures/replay-2024/alaska.csv').read_bytes());expected=read_snapshot(q2,'alaska')
    def tallies(snap):return {c.name:(c.reported_ballots,{v.name:v.votes for v in c.choices},c.overvotes,c.undervotes) for c in snap.contests.values()}
    assert tallies(actual)==tallies(expected) and len(actual.contests)==73
result={'status':'PASS','source_sha256':hashlib.sha256(raw).hexdigest(),'source_rows':evidence['rows'],
    'statewide_contests':73,'all_frozen_contest_tallies_match':True,'outputs':len(outputs),'unique_source_keys':len(seen),
    'zero_filled_repetitions':repeats,'independent_mode_contributions':dict(modes),
    'district_rows_mislabeled_election_day':sorted(conflicts),'split_precincts':list(cross.split_precincts),'vintage':cross.vintage,
    'qualifications':['Mode contributions count selections across contests, not unique voter ballots.',
                      'Boroughs are approximate precinct-only subsets; not controlling full contests.',
                      'MEDSL derivative bytes matched, not independently authenticated original ENR.']}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
