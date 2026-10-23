# Election Result Reporting

## Purpose

This subsystem answers a narrow question reproducibly: given one or more official
election-result snapshots, how exposed is each currently reported decision boundary to
change before certification, and how reliable is the evidence behind that estimate?

It ingests dated VoteWA county CSV exports and Washington `All Results` XLSX
workbooks, preserves source hashes, compares successive snapshots, estimates the
remaining contest-level vote pool, and creates a race-by-race canonical EDOM document.
The Engineering Documents Toolkit (EDT) validates that document and publishes
Markdown and HTML under `election/reports/<state>/<election day>/`. The same build also writes a compact
plain-text summary that can be pasted directly into Facebook.

Each modeled boundary has a conditional statistical change probability and a separate
0–100 reliability score. The report begins with a probability-sorted table so close
races and weak evidence can both be found quickly.

The result is a triage report, not a call by an election authority and not a substitute
for certified results.

## Repository layout

Everything for election reporting is in `election/`; only the workflows live in
`.github/workflows/`.

```text
election/
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
├── tests/                        parser, analysis, publication, watcher and
│                                 breakdown tests (offline)
├── tools/alaska_crosswalk.py     regenerates the Alaska crosswalk
└── docs/                         this guide and engineering/ (ADR-001, ADR-002)
```

Commands below run from the repository root with `PYTHONPATH=election`.

See [`election/data/README.md`](../data/README.md) for the directory
details and input columns.

## Add an update

During an election's checking window the
[source watcher](#automatic-source-watching) adds updates on its own; merge its pull
request. To add one by hand:

1. Download the official result export.
2. Put it in `election/data/elections/<state>/<election day>/input/<source>/`.
3. Rename it so the filename contains the report timestamp, for example
   `2026-08-07-1600.csv`.
4. Keep all older files, and record the source in that election's `SOURCES.json`.
5. Run the GitHub Actions workflow named `Election Result Reports`.

Each election directory is built into `election/reports/<state>/<election day>/`; an
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
`election/reports/**` is not in the workflow's `push` path filter.

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
PYTHONPATH="election:../engineering-docs-toolkit" \
  python -m unittest discover -s election/tests -v

PYTHONPATH="election:../engineering-docs-toolkit" \
  python -m election_reporting \
    --config election/data/elections/washington/2026-08-04/election.toml \
    --output election/reports/washington/2026-08-04
```

The CI pins EDT commit `649981ec78c97aafe4374284ad7e8c57c11a4640` so identical
inputs use the same publication engine.

## Automatic source watching

`election/data/watch.toml` selects states and counties by name, one `[[watch]]`
table per state:

```toml
schema_version = 1

[paths]
elections = "elections"   # election/data/elections/<state>/<election day>/

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
export PYTHONPATH=election                           # from the repository root
python -m election_reporting.watch plan              # resolve, show, and write election.toml
python -m election_reporting.watch fetch --force     # fetch now, ignoring the window
```

## Supported source formats

King County CSV-shaped exports require `ContestOrder`, `Contest`, `Contest ID`,
`Choice Order`, `Choice`, `BallotsWith Contest`, and `Votes`.

The watcher's normalized CSV adds optional `Contest Overvotes`, `Contest
Undervotes`, and `Counties` columns, which the analyzer reads (see
[`election/data/README.md`](../data/README.md#input-columns)).

The official Washington `All Results` XLSX workbook is read directly without
third-party spreadsheet dependencies. Only `Summary Results` is ingested. Rows with a
`Choice ID` define choices; `Over Votes` and `Under Votes` remain outside candidate
totals. The contest denominator is the larger of the workbook's `Ballots Cast` value
and the auditable sum of valid votes, overvotes, and undervotes. This reconciles an
export where a county has vote tallies but temporarily reports zero ballots, while
preserving the larger reported count when detailed tallies are incomplete. Adjusted
denominators are disclosed in JSON and beside the affected race in human reports. The
reader limits the workbook's total uncompressed size to 50 MiB before parsing.

Canonical keys reconcile the export naming differences for U.S. House districts,
state legislative positions, and Supreme Court positions. This is what lets a
higher-priority statewide total replace the matching King County slice while retaining
county-only races.

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
  - `rcv` (Alaska general elections) evaluates whether the first-choice leader is
    above 50 percent. Without a first-choice majority the race is flagged: later
    ranked-choice rounds decide it, the first-choice leader is not necessarily the
    winner, and Alaska tabulates rounds only after the final count on the 15th day.
    Round-by-round tabulation from cast vote records is not modeled yet.

Use `[[contest_rules]]` in `election.toml` to override a contest with `winner`,
`top_two`, `top_four`, `rcv`, `wa_judicial`, `automatic`, or `ignore` when the
inference is not correct.

## Remaining-vote calculation

For source-level reported ballots \(B_s\), configured expected final ballots \(F_s\),
and current ballots in contest \(B_c\), the estimated remaining contest ballots are

\[
R_c = \operatorname{round}\left[B_c\left(\frac{F_s}{B_s}-1\right)\right].
\]

For a two-side decision, the current participation rate of those sides scales \(R_c\)
to an estimated remaining decision-vote pool \(R\). If the current margin is \(M\),
the trailing side must receive approximately

\[
p_{\mathrm{required}} = \frac{1}{2} + \frac{M}{2R}
\]

of that remaining pool to reverse the boundary. The same algebra is applied to a
50-percent majority boundary by grouping all nonleading choices together.

`expected_final_ballots` must be accompanied by a documented basis and evidence score:

- King County uses its [official 45 percent turnout forecast](https://kingcounty.gov/en/dept/elections/results/2026/august-primary-election)
  and 1,454,559 registered voters, producing 654,552 expected final ballots with
  forecast evidence 75/100.
- Washington State uses the Secretary of State's [official 2022 primary turnout of
  40.43 percent](https://www.sos.wa.gov/elections/data-research/election-data-and-maps/reports-data-and-statistics/voter-turnout-election)
  applied to approximately 5.1 million current registered voters, producing 2,061,930
  expected final ballots with forecast evidence 45/100. This is a historical proxy,
  not an official 2026 turnout forecast.

## County and precinct results

Washington elections declare county and precinct breakdowns with `[[areas]]`
tables in `election.toml`; the files and their columns are described in
[`election/data/README.md`](../data/README.md#county-and-precinct-breakdowns).
The report's **Geography: county and precinct results** section and
`county-breakdown.csv` (contest, county, ballots, votes per choice and county
leader for every controlling race) come from the latest county snapshot.

| Part | What it shows |
| --- | --- |
| Reconciliation | County sums against the statewide totals and precinct sums against county totals, per contest and choice (labels compared without case). Differences are listed as data-quality findings and never adjusted. A county with county votes but no precinct rows is reported as missing precinct detail, not as a mismatch. |
| County changes | Between the last two county snapshots: the counties that added the most ballots, and for the ten closest races the counties whose margin moved most. |
| Precinct changes | The same between the last two precinct snapshots. |
| Remaining ballots | For each close race, the county with the most remaining ballots declared in `[[remaining_ballots]]`. With margin \(m\) and \(R\) remaining ballots scaled to the county's share of ballots cast for the two sides, the trailing side needs \((m + R) / 2R\) of them; above 100% that county cannot change the outcome alone. The needed share is compared with the county's current split. |

On the 2026 primary (certified data, Sep 11 2026), 974 county-versus-statewide and
5,874 precinct-versus-county comparisons reconcile with no mismatches; the state's
precinct sheet has no King County rows (151 contest-county pairs).

## Statistical change probability

For the two sides of a decision boundary, let (q) be the change side's current share
of their combined vote and (n) their combined observed votes. The model deliberately
tempers the apparent sample size:

\[
n_e = \min(n, 300), \qquad
\alpha = 1 + qn_e, \qquad
\beta = 1 + (1-q)n_e.
\]

When a comparable latest batch exists, its share contributes at most 200 additional
effective observations. Given (R) estimated remaining decision votes, the predictive
count follows a beta-binomial distribution. The implementation evaluates the tail
past the exact flip threshold directly when no more than 5,000 decision votes remain.
Larger tails use the distribution's mean and variance with a continuity-corrected
normal approximation.

The caps keep a large early count from producing false precision when later mail
batches differ by geography or return timing. The probability is conditional on the
remaining-ballot estimate and this exchangeability model; it is not an election call or
a probability calibrated against historical race outcomes.

## Reliability score

Probability and evidence quality answer different questions. Reliability is computed
independently as

\[
0.55F + 0.25H + 0.20S,
\]

where (F) is the configured forecast-evidence score, (H) is 25, 65, or 85 for one,
two, or at least three compatible snapshots, and
(S = \min(100, 20\log_{10}(n+1))). Reconciled ballot denominators subtract five
points, noncontrolling county slices are capped at 25, and the overall score is capped
at 85. Decision pools below 500 votes lose ten points; pools below 2,000 lose five.

| Grade | Score | Meaning |
| --- | ---: | --- |
| High | 75–85 | Strong input support; model limitations still apply |
| Moderate | 50–74 | Useful with material forecast or history uncertainty |
| Low | 25–49 | Directional only |
| Insufficient | 0–24 | Missing forecast or noncontrolling evidence |

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

`election/reports/<state>/<election day>/analysis.json` is the version-2 machine-readable analysis and
includes every input SHA-256 digest, probability, model identifier, reliability score,
and reliability rationale. `canonical-document.edom.json` is the semantic publication
source. EDT writes the human report plus validation, reference-graph, and quality
evidence:

```text
election/reports/<state>/<election day>/
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
