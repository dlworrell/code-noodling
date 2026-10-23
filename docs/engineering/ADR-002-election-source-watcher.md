# ADR-002: Scheduled Acquisition of Official Election Result Snapshots

Status: Proposed
Date: 2026-10-23

## Context

[ADR-001](ADR-001-election-result-reporting.md) made reports reproducible from
committed, dated source snapshots, and rejected live scraping during builds. It
left acquisition as a manual file drop. That has three problems:

- Snapshots are missed. The Washington Secretary of State overwrites its report
  files in place, so a version that was not saved cannot be downloaded later.
- Provenance is lost. No source URL or retrieval time was recorded for any of the
  four August 2026 inputs.
- Coverage is fixed to King County plus the statewide workbook, although the
  official results site serves every Washington county.

## Decision

Add a source watcher, `election_reporting.watch`, that acquires snapshots on a
schedule and proposes them through a pull request. Builds still read only
committed files.

1. **Selection by name.** `election-data/watch.toml` names a state and counties.
   The watcher resolves them against `results.votewa.gov`, picks the election
   (`next` or a date), and derives the checking window from the statutory
   certification deadlines: every 30 minutes from polls closing until 02:00, then
   hourly until the last certification day ends.
2. **Official sources only.** A host allowlist limits fetching to
   `results.votewa.gov`, `kingcounty.gov` and `cdn.kingcounty.gov` over HTTPS. The
   state uses its `All Results Excel` report; King County uses its own nightly
   export, stored byte for byte; other counties use the results site's data API.
3. **One election per directory.** `election-data/elections/<election day>/` holds
   `election.toml`, `input/` (plain-text CSV the analyzer reads), `raw/`
   (original downloads when they are not the input) and `SOURCES.json` (source URL,
   retrieval time, source timestamp and SHA-256 of every stored file). Reports go
   to `reports/elections/<election day>/`.
4. **One CSV layout.** Every non-county-export source is converted to the King
   County column layout plus optional contest overvote, undervote and county-span
   columns. Workbook conversion reuses the analyzer's own workbook parser, and a
   test shows the CSV and the workbook give identical analysis.
5. **County spans.** A contest's county span, from the results site, is carried
   into the analysis. A county source's share of a multi-county contest is not
   controlling; the statewide source (priority 100, over counties at 10) controls
   when it reports the contest.
6. **Pull requests, not pushes.** Per Catylist ADR-0006, a scheduled workflow
   commits new snapshots to the workflow-owned `automation/election-sources`
   branch through `createCommitOnBranch` (GitHub-verified) and opens or updates a
   pull request. Snapshots waiting in an unmerged pull request are carried forward
   into each run after their SHA-256 is checked. Nothing else on the default
   branch is changed by a run.

## Consequences

- Every new snapshot has recorded provenance and is reviewable before it is used.
- Following another Washington county is a one-word change to `watch.toml`.
- Each merged snapshot pull request regenerates the reports through the existing
  workflow.
- The workflow runs every 30 minutes all year and exits early outside a window; a
  run delayed by GitHub past its slot can skip one hourly check.
- County data from the results API has no ballots-cast, overvote or undervote
  counts, so those contests' denominators are vote totals.
- Two watched counties' shares of a multi-county local contest are not summed;
  only the statewide source gives a combined total.
- Another state needs its own source adapter.
- Maintainers merge snapshot pull requests during the window; merging is the human
  review step.

## Rejected alternatives

- **Auto-merging snapshots** would need an exception to ADR-0006's protected
  default branch.
- **Fetching during each report build** was rejected in ADR-001 and still is: it
  makes a report depend on the network at build time.
- **Keeping only the original XLSX and JSON files** would keep the analyzed input
  binary or very large. The originals are kept beside the CSV instead.
- **Using the results API for King County** would lose the county export's
  ballots, overvotes and undervotes.
