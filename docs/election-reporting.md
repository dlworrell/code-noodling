# Election Result Reporting

## Purpose

This subsystem answers a narrow question reproducibly: given one or more official
election-result snapshots, how exposed is each currently reported decision boundary to
change before certification, and how reliable is the evidence behind that estimate?

It ingests dated VoteWA county CSV exports and Washington `All Results` XLSX
workbooks, preserves source hashes, compares successive snapshots, estimates the
remaining contest-level vote pool, and creates a race-by-race canonical EDOM document.
The Engineering Documents Toolkit (EDT) validates that document and publishes
Markdown and HTML under `reports/elections/`.

Each modeled boundary has a conditional statistical change probability and a separate
0–100 reliability score. The report begins with a probability-sorted table so close
races and weak evidence can both be found quickly.

The result is a triage report, not a call by an election authority and not a substitute
for certified results.

## Repository layout

```text
election-data/
├── election.toml                 source, forecast, and rule configuration
└── input/
    ├── king-county/              dated county snapshots
    └── washington/               dated controlling aggregate snapshots
election_reporting/               Python implementation
reports/elections/                generated JSON, EDOM, Markdown, HTML, and EDT checks
tests/test_election_reporting.py  parser, analysis, and publication tests
```

## Add an update

1. Download the official result export.
2. Put it in the appropriate source directory.
3. Rename it so the filename contains the report timestamp, for example
   `2026-08-07-1600.csv.xls` or `2026-08-07-1700.xlsx`.
4. Keep all older files.
5. Run the GitHub Actions workflow named `Election Result Reports`.

The workflow tests the analyzer, checks out the pinned EDT revision, generates the
report, validates the generated EDOM, uploads the complete report directory as a CI
artifact, and—on a push or when requested during manual dispatch—commits changed
reports back to the current branch.

## Run locally

Python 3.11 or newer and EDT are required. From adjacent checkouts:

```bash
PYTHONPATH=".:../engineering-docs-toolkit" \
  python -m unittest discover -s tests -p 'test_election_reporting.py' -v

PYTHONPATH=".:../engineering-docs-toolkit" \
  python -m election_reporting \
    --config election-data/election.toml \
    --output reports/elections
```

The CI pins EDT commit `649981ec78c97aafe4374284ad7e8c57c11a4640` so identical
inputs use the same publication engine.

## Supported source formats

King County CSV-shaped exports require `ContestOrder`, `Contest`, `Contest ID`,
`Choice Order`, `Choice`, `BallotsWith Contest`, and `Votes`.

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

Use `[[contest_rules]]` in `election.toml` to override a contest with `winner`,
`top_two`, `wa_judicial`, `automatic`, or `ignore` when the inference is not correct.

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

## Scope and authority

Each input source has a priority. When matching contests exist in more than one source,
the higher-priority current snapshot controls. The included configuration gives the
Washington aggregate priority over King County.

When only King County data exists for a congressional, legislative, appellate, or
statewide race, the analyzer marks it `controlling: false`. Its local movement remains
visible, but the report explicitly refuses to present that slice as the official
outcome forecast.

## Generated evidence

`reports/elections/analysis.json` is the version-2 machine-readable analysis and
includes every input SHA-256 digest, probability, model identifier, reliability score,
and reliability rationale. `canonical-document.edom.json` is the semantic publication
source. EDT writes the human report plus validation, reference-graph, and quality
evidence:

```text
reports/elections/
├── analysis.json
├── canonical-document.edom.json
├── election-analysis.md
├── election-analysis.html
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
- The tool does not infer unreported ballots by geography.
- It does not automatically determine recount eligibility, adjudicate write-in
  candidates, or replace jurisdiction-specific election law review.
- A missing controlling aggregate produces a scope warning rather than a false
  statewide conclusion.
- Certification dates are documentation metadata; the report does not fetch or verify
  schedules from the network.
