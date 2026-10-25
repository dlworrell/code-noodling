<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:6a2b429b-d08f-4ec3-9018-c75ea86c8559
-->

# Election Result Reporting

## Purpose

This subsystem answers a narrow question reproducibly: given one or more official
election-result snapshots, how exposed is each currently reported decision boundary to
change before certification, and how reliable is the evidence behind that estimate?

It ingests dated VoteWA county CSV exports and Washington `All Results` XLSX
workbooks, preserves source hashes, compares successive snapshots, estimates the
remaining contest-level vote pool, and creates a race-by-race canonical EDOM document.
The Engineering Documents Toolkit (EDT) validates that document and publishes
Markdown and HTML under `election-analysis/reports/<state>/<election day>/`. The same build also writes a compact
plain-text summary that can be pasted directly into Facebook.

Each supported selected boundary can have an **experimental, uncalibrated conditional scenario probability** and a separate **unvalidated** 0–100 evidence rubric. The report begins with a probability-sorted table so close
races and weak evidence can both be found quickly.

The software is experimental and uncalibrated. EDT checks document structure and
provenance; it does not validate a real-world forecast or certify an outcome.

## Lifecycle, stages and legacy migration (EA-01)

[ADR-003](engineering/ADR-003-result-lifecycle-and-stages.md) amends the earlier
forecast contract. Configuration schema 1 remains readable, but now produces
descriptive results with unknown lifecycle/stage and null probabilities. A passed
certification deadline never changes that state automatically. The watcher still
writes schema 1 and therefore cannot implicitly re-enable forecasts.

Schema 2 adds `[election]` fields `report_mode`, `forecast_target` and an
offset-aware `information_time`, plus `[election.status]` and `[[result_status]]`
records. Each contest status binds to a configured `source_id`, exact input
`snapshot_sha256` and `contest_id` (or an explicit `*` for that one source/hash).
Known lifecycle/stage metadata needs an HTTPS authority URL and an offset-aware
`evidence_as_of`. Known stages require `result_version`; RCV rounds require
`round_id`. See the Washington primary configuration for an actual historical
migration: the official recount evidence confirms completion for LD42 while the
underlying summary keeps its original plurality-count stage and bytes.

Current outstanding-ballot scenarios require dated counting evidence for both
the election and contest, a supported plurality stage and an explicit information
time. Only evidence available by that time may be used. Recount, litigation and
full RCV models remain unsupported. Historical, completed, unknown and incompatible
states withhold the calculation, retain descriptive margins and explain why.
Different lifecycles/stages/rounds/result versions are never subtracted as a new
batch. Every original version remains visible in JSON input lineage.

Analysis JSON is schema 4; config schema 3 adds the reviewed `[audit]` contract
in [ADR-004](engineering/ADR-004-scientific-evidence-contract.md). Read per-contest
`evidence`, `scientific_eligibility`, `data_status`, canonical identity, source
publication/count-update/retrieval times and generated time separately. Filename
labels with unknown timezones never establish freshness. The gate supports only
experimental scenarios; `validated_forecast_eligible` remains false until real
held-out evidence and independent review support promotion. Consumers must retain
null probability/pool fields and the named withholding reasons.

The historical Washington primary summary supersedes the earlier prospective
publication. The original can be reviewed at audited revision
`99c03904a7258994bcbc25c2e73ede722d00c461`; its inputs and audit evidence are preserved.
The remaining sections describe the conditional model only when these
prerequisites are met, rather than granting legacy files forecast eligibility.

## Repository layout

Everything for election reporting is in `election-analysis/`, a governed project
scope (Catylist ADR-0004). The workflows in `.github/workflows/` call
`election-analysis/ci/` for their logic.

```text
election-analysis/
├── README.md
├── election_reporting/           Python package (ci/ holds the CI publisher,
│                                 watch/ the source watcher)
├── data/
│   ├── watch.toml                source watcher: state, counties, schedule
│   ├── crosswalks/alaska/        Alaska precinct-to-borough crosswalk
│   └── elections/<state>/<election day>/
│       ├── election.toml         source, forecast, and rule configuration
│       ├── SOURCES.json          provenance of every snapshot
│       ├── input/<source>/       dated plain-text CSV snapshots
│       ├── raw/<source>/         original downloads (XLSX, trimmed JSON)
│       └── areas/<source>/       county and precinct breakdowns (Washington)
├── reports/<state>/<election day>/  generated JSON, EDOM, human summaries, EDT checks
├── ci/                           workflow logic (reports.sh, watch.sh)
├── tests/                        parser, analysis, publication, watcher and
│                                 breakdown tests (offline)
├── tools/alaska_crosswalk.py     regenerates the Alaska crosswalk
└── docs/                         this guide and engineering/ (ADR-001, ADR-002)
```

Commands below run from the repository root with `PYTHONPATH=election-analysis`.

See [`election-analysis/data/README.md`](../data/README.md) for the directory
details and input columns.

## Add an update

During an election's checking window the
[source watcher](#automatic-source-watching) adds updates on its own; merge its pull
request. To add one by hand:

1. Download the official result export.
2. Put it in `election-analysis/data/elections/<state>/<election day>/input/<source>/`.
3. Rename it so the filename contains the report timestamp, for example
   `2026-08-07-1600.csv`.
4. Keep all older files, and record the source in that election's `SOURCES.json`.
5. Run the GitHub Actions workflow named `Election Result Reports`.

Each election directory is built into `election-analysis/reports/<state>/<election day>/`; an
election with no snapshots yet is skipped.

The workflow tests the analyzer, checks out the pinned EDT revision, generates the
report, validates the generated EDOM, and uploads the complete report directory as a
CI artifact. When the regenerated reports differ from the committed ones on the
default branch, it opens (or updates) a pull request with them; see
[Automated report publication](#automated-report-publication). Reports reach `main`
only when that pull request is reviewed and merged.

## Automated report publication

ADR-0006 (AES-DEV-001-R003 and R005) forbids automation from pushing to the default
branch and requires automation-produced commits to be GitHub-verified. The workflow
therefore never runs `git push`. Its behavior depends on the triggering event:

| Event | Behavior |
| --- | --- |
| `pull_request` | Tests, generates, and uploads the `election-result-reports` artifact. Nothing is committed. Stale committed reports produce a warning, not a failure. |
| `push` to the default branch, `schedule`, or `workflow_dispatch` on the default branch with `commit_reports` | If the reports changed, a second job commits them to the workflow-owned branch `automation/election-reports` and opens or updates a pull request to the default branch. |
| `push` to any other branch, or dispatch from another branch | Tests, generates, and uploads the artifact only. |
| Reports unchanged | Nothing is published. |

The publishing job runs `python -m election_reporting.ci.publish_reports`. It
creates `automation/election-reports` from the analyzed commit (or force-resets it
there; this branch belongs to the workflow, so do not push to it by hand) and writes
the commit through the GraphQL `createCommitOnBranch` mutation with the workflow's
`GITHUB_TOKEN`. GitHub signs those commits, so they show **Verified** and satisfy a
`required_signatures` ruleset. If the branch already contains exactly the same
reports on top of the same commit, no new commit is made.

The commit and pull request title omit `[skip ci]` so the pull request's checks are
not suppressed. Note that GitHub does not start new workflow runs for events created
with `GITHUB_TOKEN`, so checks on the automation pull request do not start on their
own: a maintainer can start them by closing and reopening the pull request (or by
pushing a commit to it). Merging it does not retrigger report generation, because
`election-analysis/reports/**` is not in the workflow's `push` path filter.

Permissions are least privilege: the workflow default is `contents: read`; only the
publishing job receives `contents: write` and `pull-requests: write`. The workflow
never uses `pull_request_target`.

> **Maintainer setting required.** The pull-request step only works when a repository
> administrator enables **Settings → Actions → General → Workflow permissions →
> "Allow GitHub Actions to create and approve pull requests"**. Without it, the
> verified commit still lands on `automation/election-reports`, but the job fails
> with a message naming this setting; open the pull request from that branch by hand
> or enable the setting and re-run the workflow.

## Run locally

Python 3.11 or newer and EDT are required. From the repository root, with EDT
checked out next to it:

```bash
PYTHONPATH="election-analysis:../engineering-docs-toolkit" \
  python -m unittest discover -s election-analysis/tests -v

PYTHONPATH="election-analysis:../engineering-docs-toolkit" \
  python -m election_reporting \
    --config election-analysis/data/elections/washington/2026-08-04/election.toml \
    --output election-analysis/reports/washington/2026-08-04
```

The CI pins EDT commit `bca66c1e2d2a9bf9c05f9429e618299216a17f27` so identical
inputs use the same publication engine.

## Automatic source watching

`election-analysis/data/watch.toml` selects states and counties by name, one `[[watch]]`
table per state:

```toml
schema_version = 1

[paths]
elections = "elections"   # election-analysis/data/elections/<state>/<election day>/

[[watch]]
state = "Washington"
counties = ["King", "Snohomish", "Pierce", "Kittitas", "Chelan"]
election = "next"

[watch.schedule]
polls_close = "20:00"
```

`python -m election_reporting.watch` (see
[ADR-002](engineering/ADR-002-election-source-watcher.md)) works out the rest:

| Step | What it does |
| --- | --- |
| Places | Resolves county names against the Washington results site (`results.votewa.gov`), which serves the state and all 39 counties; an unknown name fails with the list of valid ones. |
| Election | `next` is the soonest election on the site whose checking window has not ended; a date picks a specific one. |
| Window | From polls closing (20:00) on election day until the end of the last certification day: state 30 days after a general election and 17 after a primary; counties 21 and 14 (RCW 29A.60.190 and 29A.60.250). Every 30 minutes until 02:00 after election night, then hourly. Each state's `[watch.schedule]` table can override any of these. |
| Sources | State: the Secretary of State's `All Results Excel` report, which includes ballots cast, overvotes, and undervotes, and the `Summary and Precinct Results Excel` report for county and precinct breakdowns (see [County and precinct results](#county-and-precinct-results)). King County: the county's own nightly `webresults-YYYYMMDD.csv` export, stored byte for byte. Other counties: the results site's data API. |
| Output | A plain-text CSV per place and snapshot in `input/`, the original in `raw/` (workbooks as-is, API JSON trimmed to results and gzipped), and a `SOURCES.json` record with the source URL, retrieval time, source timestamp, and SHA-256 of each file. A snapshot is saved only when its content changed, and never before any votes are reported. |
| States | Each state is planned and fetched on its own schedule and in its own directory; one state failing does not stop the others. |
| Config | Creates `election.toml` for a new election (state priority 100, counties 10, certification dates filled in) and appends sources when places are added later. |

The **Election Source Watch** workflow runs every 30 minutes, exits early when a run
is not due, and publishes new snapshots as a GitHub-verified commit on the
workflow-owned `automation/election-sources` branch with a pull request to the
default branch, using the same publisher as the reports. Snapshots waiting in an
unmerged pull request are carried forward into each run (only files recorded in
that branch's `SOURCES.json`, each checked against its SHA-256), because the state
overwrites its report files and an earlier version cannot be downloaded again.
Merging the pull request triggers report generation. Run it by hand with
**Run workflow → force** to fetch outside the window.

The results API reports valid votes but not ballots cast, overvotes, or undervotes
for county data, so for those counties the contest denominator is the vote total.
Every contest from the API also carries the counties it spans; a county source's
share of a multi-county contest is reported as not controlling, and the statewide
totals control when the state reports the contest.

### Alaska

```toml
[[watch]]
state = "Alaska"
counties = "all"    # or names such as ["Anchorage", "Bethel", "Kenai Peninsula"]
election = "next"
```

| Step | What it does |
| --- | --- |
| Places | Alaska has no counties. Its county equivalents are 19 boroughs and 11 census areas; names resolve against [`crosswalks/alaska/precinct-borough.csv`](../data/crosswalks/alaska/README.md), with or without "Borough" or "Census Area". |
| Election | The primary is the third Tuesday in August (AS 15.25.020) and the general the Tuesday after the first Monday in November of even years (AS 15.15.020); election IDs follow the Division of Elections (`26genr`). |
| Window | 20:00 Alaska time (polls close; results post after 21:00) until the State Review Board's target certification day: 22 days after a general election and 13 after a primary, from the Division's 2026 calendar. |
| Source | The precinct results CSV linked from `www.elections.alaska.gov/election-results/e/?id=<election>`, kept gzipped in `raw/alaska/`. A file last modified before polls closed is skipped, because the Division reuses one path a year and the link can still serve the previous election's file. |
| Output | `input/alaska/` holds statewide totals from every row. Each borough or census area gets `input/<borough>/` from its election-day precincts only: absentee, early, questioned and remote ballots are reported per House district, and federal-only overseas ballots statewide, so they appear only in the statewide totals. |
| Rules | `election.toml` sets `default_rule = "top_four"` for a primary and `"rcv"` for a general election; ballot measures and judicial retention questions stay `winner`. |
| TLS | `www.elections.alaska.gov` serves the wrong intermediate certificate. The watcher adds the published GlobalSign GCC R46 OV TLS CA 2025 intermediate for that host only (`watch/certs.py`, SHA-256 checked); verification is never turned off. |

Statewide totals from the watcher match the Division's own *Election Summary
Report* for the 2026 primary, and the 2024 general election reproduces its known
results (Ballot Measure 2 failing by 737 votes; U.S. Representative going to
ranked-choice rounds with no first-choice majority).

### Maine

```toml
[[watch]]
state = "Maine"
counties = "all"    # or names such as ["York", "Cumberland"]
election = "next"

[watch.schedule]
election_night_every_minutes = 1440
every_minutes = 1440
```

| Step | What it does |
| --- | --- |
| Election | June primaries (second Tuesday, even years) and November elections every year: general elections in even years, referendum elections in odd years. |
| Window | Maine posts no election-night state results. Official results arrive as office workbooks over the following weeks (the 2026 primary's final files are dated 21 days after it, the 2024 general's corrected files 30 days after), so the watcher checks once a day for 35 days from 20:00 Eastern on election day. No certification date is published, so `election.toml` gets none. |
| Source | Every `.xlsx` linked under the election's headings on `www.maine.gov/sos/elections-voting/election-results-data` (or the year's page for an earlier election). Cast vote record exports, candidate lists and the President-by-congressional-district view are skipped. Each snapshot keeps the workbooks in one reproducible ZIP with a `SOURCES.txt` of URLs, `Last-Modified` dates and SHA-256 checksums. |
| Parsing | Workbook layouts vary by office and year: names above or below the column-header row, misspelled headers, `BLANK` and `TBC` columns, county subtotal rows, unlabeled total rows, referendum questions grouped across columns, precinct-level ranked-choice first choices. Every block is reconciled to the workbook's own total row; a block that does not reconcile, or has no total row, is left out and reported as an error, never guessed. |
| Output | `input/maine/` holds statewide totals; each county gets `input/<county>-county/` from its municipal rows. `STATE UOCAVA` (military and overseas) rows have no municipality, so they are in statewide totals only. |
| Rules | Plurality (`default_rule = "winner"`), except contests posted in a ranked-choice section of the page, which the watcher adds to `election.toml` as `[[contest_rules]]` with `rule = "rcv"`. |
| HTTP | `www.maine.gov`'s firewall refuses every non-browser client, including honestly identified ones, so that host alone receives a browser User-Agent; every Maine snapshot's `SOURCES.json` note says so. The issue was reported to the Secretary of State's elections division on 2026-10-23. |

Every workbook of the 2024 general election (14 files, 262 contests) and the
2026 primary (29 files, 418 contests) parses and reconciles to its totals.

Locally:

```bash
export PYTHONPATH=election-analysis                           # from the repository root
python -m election_reporting.watch plan              # resolve, show, and write election.toml
python -m election_reporting.watch fetch --force     # fetch now, ignoring the window
```

## Supported source formats

King County CSV-shaped exports require `ContestOrder`, `Contest`, `Contest ID`,
`Choice Order`, `Choice`, `BallotsWith Contest`, and `Votes`.

The watcher's normalized CSV adds optional `Contest Overvotes`, `Contest
Undervotes`, and `Counties` columns, which the analyzer reads (see
[`election-analysis/data/README.md`](../data/README.md#input-columns)).

The official Washington `All Results` XLSX workbook is read directly without
third-party spreadsheet dependencies. Only `Summary Results` is ingested. Rows with a
`Choice ID` define choices; `Over Votes` and `Under Votes` remain outside candidate
totals. The contest denominator is the larger of the workbook's `Ballots Cast` value
and the auditable sum of valid votes, overvotes, and undervotes. This reconciles an
export where a county has vote tallies but temporarily reports zero ballots, while
preserving the larger reported count when detailed tallies are incomplete. Adjusted
denominators are disclosed in JSON and beside the affected race in human reports. The
reader limits the workbook's total uncompressed size to 50 MiB before parsing.

Normalized names are search labels, not authority to merge source records.
Only an explicit election/jurisdiction/office/source-ID/name crosswalk establishes
a canonical contest and a complete controlling representation. County slices and
unreviewed records remain separate. Area joins require reviewed office/ID/county
aliases; generic names and reused local IDs never authorize a join.

## Decision rules

The analyzer deliberately separates the decision boundary from the displayed rank:

- Washington candidate primaries with three or more named candidates use the
  second-versus-third top-two cutoff.
- Washington general elections set `[analysis] default_rule = "winner"` (the
  watcher writes it into a new general election's `election.toml`), so every
  candidate contest uses the first-versus-second boundary.
- Candidate contests with no more than two named candidates have no modeled top-two
  qualification boundary because both named candidates advance under the default rule.
- Binary ballot measures and precinct committee officer races use the first-versus-
  second winner boundary.
- Washington judicial and Superintendent of Public Instruction contests evaluate both
  the top-two cutoff, when applicable, and whether the leader remains above 50 percent.
  VoteWA's abbreviated `Justice Position` labels are recognized as statewide judicial
  contests and therefore also receive a county-slice warning when only King County is
  available.
- Generic `Write-in` totals are excluded from named-candidate ranking but remain in the
  judicial majority denominator.
- `[analysis] default_rule` replaces the Washington inference for every candidate
  contest that no `[[contest_rules]]` entry names; binary measures stay `winner`.
  Alaska elections use it:
  - `top_four` (Alaska primaries) models the fourth-versus-fifth cutoff, because the
    top four candidates advance to the general election.
  - `rcv` (Alaska general elections) reports a raw-first-ranking diagnostic
    without a winner or stopping-rule claim. Maine governing scope follows office
    and election type, regardless of the webpage section. Anonymous patterns can
    reproduce final allocation to **supplied official finalists**, conserving all
    six Maine CD2 exports; this is not an independent eligibility/elimination engine.

Inferred/default rules are descriptive. A numeric scenario requires a reviewed
`audit.rules` record with election/office scope, authority and supported passage
requirements. Unsupported supermajority/quorum rules withhold calculations.

Use `[[contest_rules]]` in `election.toml` to override a contest with `winner`,
`top_two`, `top_four`, `rcv`, `wa_judicial`, `automatic`, or `ignore` when the
inference is not correct.

## Remaining-vote calculation

`expected_final_ballots` is retained for legacy description and never enables a
forecast. The reviewed `audit.remaining_pools` entry requires a timezone-aware
`as_of`, compatible information/source times, exact jurisdiction/category coverage,
and either an official authority/count (including an explicit zero) or ordered
`low/base/high` sensitivity scenarios with a documented basis. Nonzero counts also
require explicit contest and selected-pair participation fractions. No final-total
backfill is permitted. Turnout overrun and missing counts do not prove zero.

The mapped number of future decision votes is `round(ballots * contest_fraction *
pair_fraction)` for a selected pair. Each low/base/high scenario is propagated
through the probability calculation and labeled sensitivity, never a confidence
interval. The exact strict-reversal threshold is `floor((M+R)/2)+1`; loss of a
strict majority uses `ceil((M+R)/2)`. Ties remain unresolved.

## County and precinct results

Washington elections declare county and precinct breakdowns with `[[areas]]`
tables in `election.toml`; the files and their columns are described in
[`election-analysis/data/README.md`](../data/README.md#county-and-precinct-breakdowns).
The report's **Geography: county and precinct results** section and
`county-breakdown.csv` (contest, county, ballots, votes per choice and county
leader for every controlling race) come from the latest county snapshot.

| Part | What it shows |
| --- | --- |
| Reconciliation | County sums against the statewide totals and precinct sums against county totals, per contest and choice (labels compared without case). Differences are listed as data-quality findings and never adjusted. A county with county votes but no precinct rows is reported as missing precinct detail, not as a mismatch. |
| County changes | Between the last two county snapshots: the counties that added the most ballots, and for the ten closest races the counties whose margin moved most. |
| Precinct changes | The same between the last two precinct snapshots. |
| Remaining ballots | Legacy date-only county counts remain descriptive and withhold numeric county scenarios until a reviewed publication/category/participation contract is supplied. |

On the 2026 primary (certified data, Sep 11 2026), 974 county-versus-statewide and
5,874 precinct-versus-county comparisons reconcile with no mismatches; the state's
precinct sheet has no King County rows (151 contest-county pairs).

## Statistical change probability and evidence rubric

`disjoint_tempered_beta_binomial_v2` uses preceding history capped at 300 effective
observations and a separately weighted latest verified new-ballot batch capped at
200. The latest batch is subtracted from the historical pool before weighting.
Only explicitly classified nonnegative compatible new-ballot increments contribute
as a new batch; corrections, transfers, recounts, unknown changes and duplicate
bytes do not. These caps are hand-set and unvalidated.

Finite beta-binomial tails use a full-support normalized log recurrence through
200,000 future decision votes. Larger calculations are withheld. The numerical
policy is 1e-8 absolute / 1e-6 relative tolerance (relative floor 1e-10), checked
against rational small-case and independent SciPy references including 5,000,
5,001, skewed and 200,000-vote cases. Numerical accuracy is not calibration.

Every probability names a selected pair or selected-leader majority event. It is
not a full top-two set probability, an RCV outcome probability, or a joint chance
that any boundary changes. Maximum boundary bands are triage only.

The evidence rubric is an unvalidated heuristic, capped at 50, starting at half
the configured source rubric with penalties for small/reconciled/partial pools.
Repeated cumulative snapshots receive no independent-validation bonus. Explicit
fields expose stage, source age, coverage, provenance, remaining-pool basis, rule
support and calibration. Rubric grades are not correctness probabilities.

The prospective validation scaffold enforces frozen protocol/model lineage,
pre-outcome input times, held-out election/contest groups, no tuning leakage and
preregistered baselines. It reports group-weighted Brier/log loss, calibration bins
and group bootstrap uncertainty when eligible inputs exist. No eligible real
held-out series has been acquired; the 539-record 2024 replay is descriptive
source ingestion and cannot establish forecast accuracy.

## Risk bands

The exposure label is derived directly from the modeled probability:

| Label | Modeled change probability |
| --- | ---: |
| Toss-up | 40% or higher |
| High | 25–40% |
| Meaningful | 10–25% |
| Low | 2–10% |
| Very low | under 2% |

The exact probability, threshold share, remaining-vote estimate, effective model name,
and reliability rationale are retained in `analysis.json`.

## Facebook summary

`facebook-summary.txt` is a copy-ready plain-text post rather than a Markdown table. It
includes the six highest-probability controlling decisions outside precinct committee
officer races, up to three PCO decisions at or above 10 percent, the overall exposure
counts, and the reliability-grade distribution. Each listed decision retains its
margin, probability, and reliability score.

When a statewide export requires ballot-denominator reconciliation, the post also
reports how many controlling contests were adjusted and explains that the adjustment
reduces the reliability score.

The post links to the complete report and repeats the model caveat so the summary does
not circulate without its central limitation: the values are conditional estimates,
not official results or race calls.

## Scope and authority

Each input source has a priority. When matching contests exist in more than one source,
the higher-priority current snapshot controls. The included configuration gives the
Washington aggregate priority over King County.

When only King County data exists for a congressional, legislative, appellate, or
statewide race, the analyzer marks it `controlling: false`. Its local movement remains
visible, but the report explicitly refuses to present that slice as the official
outcome forecast.

## Generated evidence

`election-analysis/reports/<state>/<election day>/analysis.json` is the version-2 machine-readable analysis and
includes every input SHA-256 digest, probability, model identifier, reliability score,
and reliability rationale. `canonical-document.edom.json` is the semantic publication
source. EDT writes the human report plus validation, reference-graph, and quality
evidence:

```text
election-analysis/reports/<state>/<election day>/
├── analysis.json
├── canonical-document.edom.json
├── election-analysis.md
├── election-analysis.html
├── facebook-summary.txt
├── county-breakdown.csv          when the election declares [[areas]]
└── edt/document/
    ├── validation.json
    ├── validation.md
    ├── reference-graph.json
    ├── reference-graph.md
    ├── quality.json
    └── quality.md
```

## Known limits

- The expected final ballot count is a planning forecast and must be updated if the
  election authority changes its forecast.
- Statistical probabilities are conditional on a fixed remaining-ballot estimate and
  a tempered exchangeability model; the reliability score exposes, but cannot remove,
  that model risk.
- The tool does not infer unreported ballots by geography; county remaining-ballot
  counts must be declared in `[[remaining_ballots]]`.
- It does not automatically determine recount eligibility, adjudicate write-in
  candidates, or replace jurisdiction-specific election law review.
- A missing controlling aggregate produces a scope warning rather than a false
  statewide conclusion.
- Certification dates are documentation metadata for the report; the watcher derives
  its checking window from the statutory deadlines, not from a published schedule.
- Contests that span counties are summed only through the statewide source; two
  watched counties' shares of a multi-county local contest are not combined.
