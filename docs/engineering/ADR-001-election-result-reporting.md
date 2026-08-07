# ADR-001: Snapshot-Based Election Result Reporting with EDT

Status: Accepted
Date: 2026-08-06

## Context

Manual review of successive election exports can identify close decision boundaries,
but an undocumented spreadsheet analysis is difficult to repeat, audit, or update.
County-only files also create a specific correctness hazard: their displayed rank may
not be the controlling rank for a multicounty or statewide contest.

The repository already participates in AES/AEMS governance, and EDT is the ecosystem's
semantic documentation and report-validation engine.

## Decision

Add a standard-library Python subsystem that:

1. treats dated official county CSV exports and statewide XLSX workbooks as immutable
   snapshots;
2. hashes and parses every input with strict schema and numeric validation, reconciling
   statewide ballot denominators against valid-vote, overvote, and undervote totals;
3. models explicit winner, top-two cutoff, and Washington majority-status boundaries;
4. estimates conditional change probabilities with a tempered beta-binomial model and
   reports a separate evidence-reliability score;
5. estimates the remaining vote pool only from declared source-level forecasts;
6. emits a complete machine-readable analysis and canonical EDOM document;
7. delegates Markdown/HTML publication and document-quality evidence to a pinned EDT
   revision; and
8. retains source priority and scope warnings so a local slice cannot silently replace
   a controlling aggregate.

Generated reports live under `reports/elections/`. Source exports live under
`election-data/input/<source>/`. The GitHub Actions workflow may commit regenerated
reports after a deliberate manual dispatch or a direct input-data push.

## Consequences

Positive consequences:

- Additional updates require only a dated file drop.
- Every report records its exact input hashes and EDT validation evidence.
- Historical margins and latest-batch shares become mechanically reproducible.
- Every modeled boundary has a scan-friendly probability and reliability score while
  preserving the model inputs in JSON.
- State or district aggregates can supersede local slices without removing local
  history.
- Statewide export timing gaps cannot produce a denominator below its detailed tally,
  and every reconciliation remains visible in generated evidence.
- The model's formulas, risk bands, and limitations are reviewable in code and docs.

Tradeoffs:

- Numerical probabilities remain conditional on the remaining-ballot estimate and a
  documented exchangeability model because ballot batches are not independent random
  samples.
- Forecast quality depends on the configured expected final ballot count.
- EDT is an explicit build dependency and is pinned in CI for reproducibility.
- Jurisdiction-specific rules can require reviewed configuration overrides.

## Rejected alternatives

- A spreadsheet-only workflow was rejected because it does not provide a stable parser,
  test suite, or deterministic CI publication path.
- Live scraping during every build was rejected because it weakens provenance and
  reproducibility; source acquisition remains an explicit file-drop step.
- Treating the county export as authoritative for every row was rejected because it
  would generate materially wrong conclusions for broader contests.
