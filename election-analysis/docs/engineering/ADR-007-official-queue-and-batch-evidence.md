<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:417aa847-25a6-455c-a5cb-0cf234c0a28c
-->

# ADR-007: Official queue evidence and conditional batch comparisons

Status: Proposed for review; experimental offline research only
Date: 2026-10-25 (Pacific)
Issues: EA-10 #55, EA-13 #58; parent #45
Extends: ADR-004/005/006; distinct offline artifact schema 3
Base: merged PR #64, `69db5f7e96bff0c4cd0ca8a987f3d284fb4a0802`

## Problem and decision

The King experiment could not run an official outstanding-count comparison.
The closest acquired public match is Orange County's November 2024 general
election: two cumulative PDFs, two filled unprocessed-ballot forms, a duplicate
press copy, an official final CSV, certified statement of votes and precinct
export. Preserve all eight original files in a hashed archive. Use bounded
readers recovered under Dropbox PRM-0026, acceptance/comparison under PRM-0030,
and static missing-evidence reconciliation under PRM-0068. The raw files are
not repaired, relicensed as project code or promoted into live forecast inputs.

`official_validation` is a separate two-phase schema-3 reconstruction. The
prospective evaluator, King experiment/seal, production model, 300/200 caps and
actual forecast-eligibility rules stay unchanged. Primary predictions use zero
verified latest votes. A separate conditional comparison asks what the disjoint
and latest-increment calculations would yield **if** the compatible tally
changes contain only newly counted votes. The source does not establish that
condition; operational delta classification stays `unknown`.

## Recovered source semantics

Both press-release count labels point to the November 8 PDF. A separately
retrieved canonical Run_13 PDF supplies November 9. The press and canonical
November 8 files have different bytes/metadata but identical results in all
212 contests. Independent Poppler numeric rows agree with the pypdf reader.

The November 8 pool form says AMENDED and has a later producer modification
clock than its face timestamp. Neither that face timestamp nor embedded PDF
metadata authenticates historical publication. Use one qualified reconstruction
time, November 9 at 19:05 Pacific, after the latest observed producer activity.
This is a research convention, not evidence of a forecast published then or an
unchanged historical archive. Actual acquisition was October 25, 2026 UTC; only
the acquisition date was recorded, so no exact retrieval clock is invented.

Processed ballots rise from 1,148,043 to 1,212,226: 64,183, matching the dated
registrar release. Form field 2 is 140,894/205,077. Despite its last-report label,
each equals additions since a common 1,007,149 election-night baseline. It is
never automatically the latest batch. Queue estimates are 275,420/212,152;
processed plus estimated queue grows by 915. The challenged counts 9,462/9,512
are separate fields with unknown overlap. Do not silently add them to the queue.

The final CSV and precinct TSV have 172 exact contest titles and 412 exact
candidate names whose total, mail and vote-center counts all agree. Forty
measure titles are abbreviated differently; four non-scored candidate names
have damaged accents in the TSV. These remain unmatched. Rejected/unassigned
write-in rows remain visible, not silently dropped or joined by choice ID zero.
The SOV's original 1,915-page PDF is retained as a reference; the 100-page count
reader deliberately rejects it. It was not fully parsed or certified by us.

## Frozen experiment and target

The protocol freezes all 36 package Python files plus the pinned PDF dependency
file, the archive manifest, whole county/city scope, grouping, probabilities,
assumptions, losses and exclusions before first real scoring. No fitting occurs;
training groups are empty. Both snapshots, all contests and all sensitivities
belong to one held-out Orange County election group. The older King study stays
a separate assumption stratum rather than becoming outcome-selected training.
Public final results were inspected for source/identity qualification before
freeze; a regression generated as-of subset predictions without outcomes or
losses. This is computational phase separation, not blind preregistration.

Scope is contested single-seat city offices in the county's listed 34 cities
and County Supervisor 1st District. Federal/state county slices, unreviewed
school/special districts, measures/multiseat and ties/unopposed are excluded
using as-of inputs. Of 212 source contests, 139 are outside reviewed scope and
20 are unopposed/tied, leaving 53. The event is strict reversal of the selected
as-of top two named candidates in the original officially delegated final CSV.
It does not adjudicate the legal winner, other candidates or later recounts.

The base assumes all 212,152 estimated unprocessed ballots become countable and
have the same geographic/selected-pair fraction as counted county ballots.
Lower acceptance uses 80%; a separate-challenged sensitivity adds all 9,512
assuming no overlap and full acceptance. Future arrivals are excluded as a
scenario condition, never asserted zero or bounded. All three scenarios score
the same 53 contests; 159 rows are not 159 independent observations.

Predict reads exactly four as-of files and zero reserved outcomes. The 159-row
prediction seal, made before the first scoring command, is
`98b0437b03bb082dad87d3fa948c3f5fe70e4b0108ba2592432ca9d3ce3378eb`.
Score verifies it and regenerates the complete as-of recipe, normalizing only
actual computation clocks, before reading the named final CSV. All hashes and
clocks are in the neighboring protocol, seal and machine verification records.

## Result and acceptance boundary

Two of the 53 selected-pair comparisons reverse. Equal contest weights give:

| Construction | Brier | Log loss |
| --- | ---: | ---: |
| model | 0.03368266 | 0.09501048 |
| fixed half | 0.25000000 | 0.69314718 |
| cumulative share | 0.03777127 | 0.13645187 |
| untempered beta binomial | 0.03725110 | 0.12758258 |
| official queue margin rule | 0.13207547 | 0.36619096 |
| reported increment only conditional | 0.03869495 | 0.13811865 |
| disjoint increment conditional | 0.02887058 | 0.08424815 |

The lower-acceptance model Brier/log loss is 0.03515299/0.10109314; the
separate-challenged sensitivity is 0.03367871/0.09468787. These are diagnostics
under different conditions, not fitted alternatives or confidence bounds.
No probability superiority, calibration, useful latest cap, independent
confidence interval or supported predictive-interval coverage follows from one
county/election and two reversals. Do not select the conditional disjoint result
as a new model because it has the lowest observed loss.

Independent Pandas aggregation of all scored final pairs from the precinct TSV
agrees with the CSV outcomes. Independent direct mean losses agree within 1e-14.
636 SciPy distribution checks agree to maximum absolute error 9.65e-11, below
ADR-004's numerical tolerance. These verify arithmetic/extraction and are not
the independent final scientific re-audit required by #45.

The data closes the absence of a dated official queue estimate **for this
experiment** and enables conditional comparisons. It does not supply a
correction-separated batch ledger, authenticated historical versions, a
contest-specific countable pool, independent training/held-out geographic
replication or independent final review. EA-10/EA-13 and parent #45 therefore
remain open. Tests and source matches cannot erase those requirements.

## Failure, trust and recovery

Original inputs are external anonymous aggregates. Readers enforce byte/page/
row/numeric/shape bounds, expected headers, exact identities, voting-method and
category conservation, required report termination, and PDF widget/canonical
field consistency. A malformed/truncated or wrong-scope source is rejected;
unmatched labels remain explicit. No PDF repair, fuzzy join or final-total
backfill fills a gap. The archive is hash-bound before phase-restricted reads.

The pinned pypdf dependency is needed by the new offline PDF readers/tests; live
reporting imports it only when those readers are invoked. Independent research
checks additionally need Pandas/SciPy/Matplotlib and Poppler. No repository
permission or workflow token boundary is widened. AES-SEC-001 native controls
remain not applicable to this Python-only scope. Revert this logical change
series to remove the optional experiment; original production/King behavior is
independent of it. Licensing retains publisher rights in a dedicated LicenseRef.

## Reproduce

Historical note: the commands below describe the original source/runtime. After
PR #67 advanced pypdf to 6.21.2, current main correctly rejects the old protocol's
changed dependency file. Use the exact-source runner in [ADR-008](ADR-008-frozen-runtime-reproduction.md)
with a matching historical interpreter; do not edit the first protocol or seal.

From the original frozen repository root, with pinned EDT available:

```bash
python -m pip install -r election-analysis/ci/requirements-provenance.txt
PYTHONPATH=election-analysis python -m election_reporting.official_qualification --fixtures election-analysis/tests/fixtures --output /tmp/oc-qualification.json
PYTHONPATH=election-analysis python -m election_reporting.official_validation predict --protocol election-analysis/docs/engineering/oc-official-protocol-v1.json --fixtures election-analysis/tests/fixtures --output /tmp/oc-predictions.json
```

Hash the new prediction bytes and record their seal before scoring. Reproduction
has new computation clocks, so its file hash differs from the original seal:

```bash
PYTHONPATH=election-analysis python -m election_reporting.official_validation score --protocol election-analysis/docs/engineering/oc-official-protocol-v1.json --fixtures election-analysis/tests/fixtures --predictions /tmp/oc-predictions.json --predictions-sha256 RECORDED_NEW_HASH --output /tmp/oc-scores.json --plot /tmp/oc-diagnostic.png
python election-analysis/ci/verify_official_experiment.py --scope election-analysis --output /tmp/oc-independent-check.json
EDT_DIR=/path/to/engineering-docs-toolkit python election-analysis/ci/test_results.py --output /tmp/oc-checks
```

The independent verification command checks the committed original artifacts;
it does not replace or rewrite their first seal. Full reproduction requires
optional Pandas/SciPy/Matplotlib and Poppler in addition to pinned test tools.
