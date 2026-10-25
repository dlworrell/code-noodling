<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:2a8034f6-00f4-4176-8aac-57476e522efd
-->

# Orange County 2024 official evidence formats

PRM-0026 recovery output. Readers: `election_reporting/official_archives.py`.
Original corpus: `tests/fixtures/oc-official-2024.zip`; exact URLs/roles/hashes:
`oc-official-2024-manifest.json`. Originals are unchanged.

| Source | Recovered fields and units | Limits / unsupported semantics |
| --- | --- | --- |
| 60-page cumulative PDF, Run_12/13 | Election/publisher, internal run date/time, exact contest instruction, district ballots/registration, named total/mail/vote-center votes | pypdf layout text; <=8 MB, <=100 pages, <=20,000 xref objects, <=16 MB extracted text. Not a generic PDF or SOV reader. Run clock is not authenticated publication. Unknown measure passage rules remain unknown |
| One-page filled unprocessed PDF | Canonical AcroForm /V plus page-widget consistency; AMENDED label; processed, source field 2, six unprocessed categories, estimate total, challenged count, next update | All categories must conserve to the displayed estimate. Challenge overlap unknown. Field 2 label is not accepted as a latest batch. No accepted/countable or future-arrival assumption is inferred |
| Official final summary CSV | Exact 11-column header after status/county preamble; separate total-precinct row; candidate/method counts, ballots, undervotes, vote-for | <=8 MB/20,000 rows, UTF-8 BOM allowed. Exact within-contest identity; only explicit party suffix/asterisk and presidential ticket slash presentation normalized. Original name retained. Overvotes absent, remain null |
| Official media ZIP / results-final.txt | 119,664 precinct/contest/choice records, tab separated despite .txt; method counts plus under/over/invalid fields | Independent Pandas qualification only, not a production ingest reader. Choice ID 0 repeats for distinct write-in categories. Never deduplicate accounting by choice ID or multiply ballot accounting by candidate rows |
| Certified SOV PDF | Original 1,915-page reference and content inventory inspected | Retained, not fully parsed or treated as proof of forecasting calibration. Deliberately rejected by the 100-page count reader |

The independent Poppler oracle reconciles all candidate/cast numeric row triples
in original order for three count PDFs. It does not reuse the pypdf layout state
machine. Independent summary/precinct joins reconcile 412 exact named choices
across 172 exact contest titles in all three vote columns. Forty abbreviated
measure titles and four damaged accented names remain unmatched. Extra
rejected/unassigned write-in rows are recorded in the qualification report.
All 53 scored selected pairs match exactly across summary and precinct exports.

Malformed source tests cover wrong status/columns, bad/truncated row shape,
invalid bytes, wrong PDF scope/size, missing/end-following report data, negative
or inconsistent choice totals, unknown wrapped lines, wrong county/election
chronology and inconsistent category totals. Original hashes, rather than a
repaired export, identify every retained input.

Historical-byte authenticity, correction partitions, contest geographic queue
shares and countable/accepted pools remain unsupported. The format readers do
not silently promote parsed values to those meanings. See ADR-007 and
[INVESTIGATION.md](INVESTIGATION.md) for the next evidence contract.
