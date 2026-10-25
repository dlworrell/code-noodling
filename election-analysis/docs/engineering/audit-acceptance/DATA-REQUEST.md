<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:09e40207-35ed-47e2-8ff9-7a67d1a01f77
-->

# Anonymous aggregate record request — draft, unsent

Purpose: evaluate the experimental election-count model in audit EA-AUD-2026-10-23,
GitHub #45/#55/#58. No request has been sent to a person or agency.

To the Orange County Registrar of Voters records custodian:

Please provide existing machine-readable aggregate records, if retained, that
reconcile the November 8 and November 9, 2024 cumulative reports and the reported
64,183-ballot increase. CSV or JSON is suitable. Individual voter identifiers,
addresses and linked individual vote selections are not requested.

Please include available fields or documentation for:

- Election, jurisdiction, report/version identity, publication and count-update
  timestamps, revision/supersession history and an authoritative historical copy.
- Batch identity using the published manifest's Container, Tabulator and Batch
  Name keys; date/time each included batch entered the cumulative tally.
- Aggregate candidate/contest counts by batch/method, distinguishing newly added
  ballots from replacements, rescans, adjudication corrections, recount changes
  and transfers. Preserve negative/corrected counts and identify overlap. A
  missing correction field is not assumed to mean zero.
- Unit definitions and any mapping needed between ballot sheets/cards, unique
  counted voter ballots and contest votes. The published manifest sums to
  2,757,630 in its Number of Ballots column, while the certified statement names
  1,417,397 counted ballots; these units must remain distinct until reconciled.
- Dated unprocessed/accepted/challenged/cure/rejected/conditional-registration/
  later-arrival categories at those report times, category overlaps, and any
  existing contest-geography or ballot-style allocation of the remaining pool.

Reference manifest SHA-256:
`36b6c0079cc3882e601dc7ef7024d987973e3a17bceb9263dabbc433c6e78cff`.
The November 8 form is marked AMENDED. Its source field labeled from the last
report reconciles instead to election-night additions; any existing field
interpretation and amendment history would help avoid a false latest-batch count.

If a requested record or field was not retained or cannot be provided, please
identify that limitation. There is no request to fabricate a historical record,
create individual voter data or infer a missing value from certified final totals.
