<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:c7d6364c-e0fe-4597-b941-eab09dea4086
-->

# ADR-002: Scheduled Acquisition of Official Election Result Snapshots

Status: Proposed
Date: 2026-10-23

Amended 2026-10-23: election files moved under `election-analysis/`. Paths below read
`election-data/` → `election-analysis/data/` and `reports/elections/` → `election-analysis/reports/`.

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

1. **Selection by name.** `election-data/watch.toml` has one `[[watch]]` table per
   state, naming the state, its counties and the state's schedule. A state may
   appear only once. The watcher resolves them against `results.votewa.gov`, picks the election
   (`next` or a date), and derives the checking window from the statutory
   certification deadlines: every 30 minutes from polls closing until 02:00, then
   hourly until the last certification day ends.
2. **Official sources only.** A host allowlist limits fetching to
   `results.votewa.gov`, `kingcounty.gov` and `cdn.kingcounty.gov` over HTTPS. The
   state uses its `All Results Excel` report; King County uses its own nightly
   export, stored byte for byte; other counties use the results site's data API.
3. **One election per directory.** `election-data/elections/<state>/<election day>/` holds
   `election.toml`, `input/` (plain-text CSV the analyzer reads), `raw/`
   (original downloads when they are not the input) and `SOURCES.json` (source URL,
   retrieval time, source timestamp and SHA-256 of every stored file). Reports go
   to `reports/elections/<state>/<election day>/`.
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

7. **Alaska from one precinct file.** Alaska reports only by precinct and House
   district, so the watcher reads the Division of Elections' statewide precinct
   CSV and writes statewide totals plus one source per borough or census area
   from a committed, population-weighted precinct crosswalk
   (`election-data/crosswalks/alaska/`). Borough sources hold election-day
   precinct votes only; district-level ballots are never apportioned by
   estimate. Primaries use a `top_four` rule and general elections an `rcv`
   rule that flags races without a first-choice majority.
8. **Fix TLS by pinning, never by disabling.** `www.elections.alaska.gov` is on
   the host allowlist with the correct published intermediate certificate added
   to the system trust store for that host alone, SHA-256 checked at load.

9. **Maine from office workbooks, reconciled or refused.** Maine posts official
   results only, as one workbook per office in layouts that vary between
   offices and years. The watcher combines every workbook listed for the
   election into one snapshot, checks each block against the workbook's own
   total row, and leaves out and reports any block that does not reconcile.
   It checks once a day for 35 days.
10. **One host, one browser User-Agent, recorded.** `www.maine.gov` refuses all
    non-browser clients. Requests to that host alone send a browser
    User-Agent; every other host gets the watcher's own, and each Maine
    snapshot records the exception.

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
- Another state needs its own source adapter; the per-state directories and
  `[[watch]]` tables mean adding one moves no existing data. Alaska needed the
  adapter, a crosswalk, two analysis rules and one pinned certificate.
- Alaska borough totals exclude absentee, early and questioned ballots, so they
  describe election-day voting, not final borough results.
- If the Alaska certificate chain is fixed or the intermediate rotates, the pin
  is harmless until it expires (2029-06-23) and is then removed.
- A new Maine workbook layout fails loudly (a reconciliation error in the run
  log) instead of producing wrong numbers; the parser then needs a fix.
- Maine county totals exclude military and overseas ballots, which the state
  reports statewide.
- If Maine's firewall starts accepting the watcher's own User-Agent, the
  override should be removed.
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
