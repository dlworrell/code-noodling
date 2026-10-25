# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:184eb897-070a-4aab-ab96-9d3f41175ee0
"""Independent openpyxl read and final-allocation oracle, not eligibility adjudication."""
import argparse, hashlib, io, json, zipfile
from collections import Counter
from pathlib import Path
import openpyxl

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--original-package',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();f=a.repo/'election-analysis/tests/fixtures'
m=json.loads((f/'maine-2024-cd2-manifest.json').read_text())
patterns=Counter();keys=set();account=[]
with zipfile.ZipFile(a.original_package) as z:
    for spec in m['files']:
        raw=z.read(spec['path']);assert hashlib.sha256(raw).hexdigest()==spec['sha256']
        wb=openpyxl.load_workbook(io.BytesIO(raw),read_only=True,data_only=True)
        n=0
        for ws in wb.worksheets:
            rows=ws.iter_rows(values_only=True);header=next(rows)
            assert header[0]=='Cast Vote Record'
            for r in rows:
                if not r[0]:
                    assert not any(r)
                    continue
                if spec['kind']=='machine':
                    key=tuple(r[:3]);assert key not in keys;keys.add(key)
                ranks=tuple(str(x).strip() if x is not None and str(x).strip() else 'undervote' for x in r[3:6])
                assert len(ranks)==3
                patterns[ranks]+=1;n+=1
        wb.close();assert n==spec['ballots'];account.append({'file':spec['path'],'ballots':n,'sha256':spec['sha256']})
totals=Counter();stops=Counter()
for ranks,n in patterns.items():
    assert set(ranks)<=set(m['finalists']+m['noncontinuing']+['undervote','overvote'])
    # Independently enumerate the stopping position, then seek the first supplied
    # continuing choice before it. No generic candidate eligibility is inferred.
    stop=len(ranks);reason='no_continuing_choice'
    for i,v in enumerate(ranks):
        if v=='overvote':stop=i;reason='overvote';break
        if i and ranks[i-1]==v=='undervote':stop=i;reason='skipped_ranks';break
    selected=next((v for v in ranks[:stop] if v in m['finalists']),None)
    if selected:totals[selected]+=n
    else:stops[reason]+=n
fixture=json.loads((f/'maine-2024-cd2-patterns.json').read_text())
assert patterns==Counter({tuple(r['ranks']):r['count'] for r in fixture['patterns']})
out={'reader':'openpyxl '+openpyxl.__version__,'source':'exact original frozen ZIP; each six-export hash verified',
     'files':account,'anonymous_patterns':len(patterns),'machine_duplicate_keys':0,'ballots':sum(patterns.values()),
     'continuing':sum(totals.values()),'finalist_totals':dict(totals),'exhaustion':dict(stops),
     'matches_all_committed_anonymous_patterns':True,
     'scope':'Final allocation to supplied official finalists; no independent eligibility or elimination adjudication'}
assert sum(patterns.values())==402936
assert sorted(totals.values())==[194445,197151]
assert dict(stops)=={'skipped_ranks':10673,'no_continuing_choice':388,'overvote':279}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
