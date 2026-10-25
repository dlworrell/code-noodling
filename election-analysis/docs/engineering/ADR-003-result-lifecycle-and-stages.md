<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:035e6367-45bc-4dfe-b80e-678d3d628d09
-->

# ADR-003: Dated lifecycle evidence and separate tabulation stages

Status: Proposed
Date: 2026-10-24
Issue: [EA-01 #46](https://github.com/dlworrell/code-noodling/issues/46)
Audit: [EA-AUD-2026-10-23 #45](https://github.com/dlworrell/code-noodling/issues/45)

## Context

ADR-001's remaining-ballot model can continue past a completed count because
certification dates are only scheduled dates. ADR-002 preserves bytes but does not
distinguish first rankings, transfer rounds or recount versions. The audited LD42
report consequently assigns a prospective reversal probability to an unchanged,
completed recount. This project remains experimental and uncalibrated.

## Decision

Amend ADR-001's forecast prerequisites and ADR-002's normalized-input contract:

1. Configuration schema 2 separates `report_mode` (`descriptive`, `historical`,
   `current`), `forecast_target` (`none`, `outstanding_ballots`, `recount`,
   `litigation`) and an offset-aware `information_time` under `[election]`.
   `[election.status]` records the election lifecycle and dated authority evidence.
2. Election and contest lifecycles are `unknown`, `awaiting_results`, `counting`,
   `final_tabulation`, `certified`, `recount_pending`, or `recount_completed`.
   Contest tabulation stage is separately `unknown`, `plurality_count`,
   `raw_first_rankings`, `rcv_round`, `final_round`, or `recount`.
3. `[[result_status]]` binds reviewed metadata to `source_id`, the exact input
   `snapshot_sha256` and `contest_id`. `*` explicitly covers every contest in that
   one source/hash, with an exact contest selector taking precedence. Duplicate
   selectors, unmatched hashes/IDs, unsupported values and undated known statuses
   fail with actionable errors. Known stages require `result_version`; an
   `rcv_round` also requires `round_id`. Original input bytes never change.
4. A known status requires an HTTPS `authority_url` and offset-aware
   `evidence_as_of`: the time the evidence was available for this analysis, not
   an inferred certification event time. Later evidence cannot support an earlier
   information time. Neither a deadline nor a turnout overrun establishes a status.
5. A prospective count scenario requires current mode, an outstanding-ballots
   target, an explicit information time and dated counting evidence for BOTH the
   election and the particular contest version. Only `plurality_count` is
   supported here. RCV, recount and litigation models remain unavailable.
   Noncontrolling slices also withhold forecasts. This is a lifecycle/stage gate,
   not the complete scientific eligibility gate in EA-14.
6. Unknown, historical, completed or incompatible states never execute the tail
   calculation. Probabilities/remaining-pool/required-share/batch fields are null,
   evidence score is zero with an explicit withholding reason, and descriptive
   margins remain visible. Null is never converted to numerical zero.
7. Batch and margin histories compare only the same known lifecycle, stage, round and result
   version. Every original source version stays in JSON input lineage; excluded
   histories are named. Transfers and recount changes are not new ballot batches.
8. Analysis JSON schema 3 carries all these fields and input-version lineage.
   Markdown, HTML and plain-text social summaries share the same context. A
   historical summary replaces the prospective headline and numerical ranking.

## Migration and recovery

Schema 1 remains readable as descriptive, stage/lifecycle unknown, target none,
information time unknown: tallies remain available, probabilities are withheld.
The watcher continues to emit schema 1 until it can emit reviewed status evidence;
it cannot implicitly re-enable forecasting. Migrating requires schema 2 plus
reviewed source/hash/contest records. Never attach final status to an entire
historical source directory. Correct a status annotation through a new PR;
retain old snapshots and the audited publication in Git history.

The Washington primary configuration is explicitly historical. Its selected LD42
snapshot is annotated from the official completed-recount table, which matches
13,125/13,121. Other legacy contest versions remain unknown rather than claiming
each old county snapshot is certified. The election-level certification evidence
does not upgrade those individual snapshots.

## Evidence and limits

- [Official primary/recount table](https://www.sos.wa.gov/elections/data-research/election-data-and-maps/election-results-and-voters-pamphlets/2026-primary-election).
- [Official certification announcement, August 21](https://www.sos.wa.gov/about-office/news/2026/secretary-state-certifies-candidates-and-measures-november-general-election).
- [Linked recount certificate](https://www.sos.wa.gov/sites/default/files/2026-10/2026_Primary_LD42_Recount_Certification_9.1.26_.pdf).

The table and announcement were read on October 24. The certificate is linked
for source lineage; its scanned contents are not used to invent a precise event
timestamp. The observation time is recorded separately in the remediation record.

This change does not resolve source-vintage reconciliation (EA-02), canonical
identity (EA-03), zero-pool/tie semantics (EA-04), full RCV interpretation/ingestion
(EA-05/06), or calibration (EA-13). Passing the gate is not a race call or a claim
of predictive validity. The independent numerical model is unchanged.

## Verification

Keep the original numerical checks on explicitly dated synthetic counting
fixtures. Add regressions for legacy unknown status, passed deadlines, all
completed/recount states, evidence after information time, incompatible rounds
and versions, raw-first-ranking majorities, selector failures, the actual LD42
input, and agreement across JSON/Markdown/HTML/social publication. Record exact
commands and source/input hashes in `audit-ea01/FEATURE-REBUILD.md`.
