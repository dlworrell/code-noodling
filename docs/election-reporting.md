# Election Result Reporting

## Purpose

This subsystem answers a narrow question reproducibly: given one or more official
election-result snapshots, how exposed is each currently reported decision boundary to
change before certification?

It ingests dated VoteWA CSV exports, preserves source hashes, compares successive
snapshots, estimates the remaining contest-level vote pool, and creates a race-by-race
canonical EDOM document. The Engineering Documents Toolkit (EDT) validates that
document and publishes Markdown and HTML under `reports/elections/`.

The result is a triage report, not a call by an election authority and not a substitute
for certified results.

## Repository layout

```text
election-data/
├── election.toml                 source, forecast, and rule configuration
└── input/
    ├── king-county/              dated county snapshots
    └── washington/               optional controlling aggregate snapshots
election_reporting/               Python implementation
reports/elections/                generated JSON, EDOM, Markdown, HTML, and EDT checks
tests/test_election_reporting.py  parser, analysis, and publication tests
```

## Add an update

1. Download the official result export.
2. Put it in the appropriate source directory.
3. Rename it so the filename contains the report timestamp, for example
   `2026-08-07-1600.csv.xls`.
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

## Risk bands

The bands are intentionally broad and are based primarily on the required remaining
share:

| Label | Heuristic flip band | Baseline required share |
| --- | ---: | ---: |
| Toss-up | 35–65% | at most 50.5% |
| High | 25–45% | 50.5–52% |
| Meaningful | 15–35% | 52–55% |
| Low | 5–15% | 55–60% |
| Very low | under 5% | above 60% |

A supplied history that shows the boundary changing promotes the assessment to
`Toss-up`. A sufficiently large latest batch that materially exceeds the required
share promotes risk one band; a batch giving the change side no more than 47 percent
demotes it one band. An absolute margin of three votes or fewer is also a toss-up when
the forecast leaves enough votes to reverse it. These adjustments are visible in
`analysis.json`.

These percentages are judgment bands, not fitted probabilities. Ballot batches are
not random samples; geography, return method, cure activity, and processing order can
all change their composition.

## Scope and authority

Each input source has a priority. When matching contests exist in more than one source,
the higher-priority current snapshot controls. The included configuration gives the
Washington aggregate priority over King County.

When only King County data exists for a congressional, legislative, appellate, or
statewide race, the analyzer marks it `controlling: false`. Its local movement remains
visible, but the report explicitly refuses to present that slice as the official
outcome forecast.

## Generated evidence

`reports/elections/analysis.json` is the machine-readable analysis and includes every
input SHA-256 digest. `canonical-document.edom.json` is the semantic publication
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
- The tool does not infer unreported ballots by geography.
- It does not automatically determine recount eligibility, adjudicate write-in
  candidates, or replace jurisdiction-specific election law review.
- A missing controlling aggregate produces a scope warning rather than a false
  statewide conclusion.
- Certification dates are documentation metadata; the report does not fetch or verify
  schedules from the network.
