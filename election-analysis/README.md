<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:8376ba33-0e6e-4a9e-b237-1d4a2a31b1a8
-->

# Election result reporting

Watches official election results for Washington, Alaska and Maine and keeps
dated snapshots with their provenance. Reports preserve descriptive tallies and
explicit election/contest lifecycle and tabulation stages. Prospective count
scenarios require dated counting evidence and an explicit target/information time;
legacy, historical, completed and unsupported RCV states withhold probabilities.
See [ADR-003](docs/engineering/ADR-003-result-lifecycle-and-stages.md) and
[ADR-004](docs/engineering/ADR-004-scientific-evidence-contract.md) for migration.
Configuration schema 3 and analysis schema 4 require reviewed time, identity,
coverage, rules and remaining-pool evidence for any experimental conditional
scenario. Missing evidence withholds numbers while retaining descriptive tallies.
The [iteration ledger](docs/engineering/audit-iteration-20261024.md) records implementation, numerical checks, source gaps and calibration blockers.

**Experimental and uncalibrated.** The Washington primary report is a historical
audit correction. The [independent AI review and acceptance](docs/engineering/audit-acceptance/INDEPENDENT-REVIEW-ACCEPTANCE.md)
record the repaired implementation and documented-unvalidated evaluation outcome.
[EA-AUD-2026-10-23 #45](https://github.com/dlworrell/code-noodling/issues/45)
records the final work-item dispositions and residual scientific limits.

**A triage report, not a call by an election authority and not a substitute for
certified results.**

## November 3, 2026 collection gates

Live collection is pinned to the **2026-11-03** general election. Every state
waits until one hour after its scheduled final poll closing. The hard
`not_before` boundary runs **before any results-site discovery or download**,
including `plan`, manual workflow dispatch and `fetch --force`.

| State | Earliest collection, November 3 local time | UTC instant (November 4) |
|---|---|---|
| Maine | 21:00 EST | 02:00 UTC |
| Washington | 21:00 PST | 05:00 UTC |
| Alaska (statewide) | 21:00 HST in Adak / 22:00 AKST in Anchorage | 07:00 UTC |

Alaska waits for the western Aleutian polls as well as the mainland. These
instants are the configured policy derived from the official scheduled closing
times; they do not detect an emergency extension. GitHub may start cron jobs
late, so collection begins at or after the applicable boundary. Offline tests
and committed-data analysis remain available before election night.

Sources checked October 25, 2026: [Washington's 2026 calendar](https://www.sos.wa.gov/elections/elections-calendar/dates-and-deadlines),
[Alaska's election calendar](https://www.elections.alaska.gov/calendar/) and
[polling-place hours](https://www.elections.alaska.gov/election-polls/),
[Maine polling times, 21-A §626](https://legislature.maine.gov/statutes/21-A/title21-Asec626.html),
and [the federal Hawaii-Aleutian boundary, 49 CFR 71.12](https://www.ecfr.gov/current/title-49/subtitle-A/part-71/section-71.12).

`python -m election_reporting.watch status` displays eligibility without source
requests. `status --now 2026-11-04T02:00:00Z` simulates a boundary offline;
`--now` is refused for live commands. `--force` may override cadence or the
window end, but cannot override `not_before`. Fixed end dates keep later annual
cron occurrences idle; extend the configuration if certification is delayed.

## Layout

This directory is the `election-analysis` governed project scope declared in
the repository's [`aes-manifest.yaml`](../aes-manifest.yaml) (Catylist
ADR-0004). AES-SEC-001 is declared not applicable: the scope is Python only,
with no C or C++ surface; the declaration is revisited if that changes.

Everything for election reporting is in this directory. The workflows in
[`.github/workflows/`](../.github/workflows/) only pick triggers, permissions and
artifacts, and call the scripts in [`ci/`](ci/).

| Path | Contents |
|---|---|
| [`election_reporting/`](election_reporting/) | Python package: analyzer, EDT publisher, source watcher (`watch/`) and CI publisher (`ci/`) |
| [`data/`](data/) | Watcher settings (`watch.toml`), the Alaska crosswalk, and one directory per state and election with its configuration, snapshots and `SOURCES.json` ([details](data/README.md)) |
| [`reports/`](reports/) | Generated reports, one directory per state and election |
| [`ci/`](ci/) | Workflow logic: `reports.sh` (tests, report generation, change detection, publishing), `watch.sh` (tests, carry-forward, fetch, publishing) and `provenance.sh`/`provenance.py`/`test_results.py` (report-only REUSE, File-ID and scope-manifest checks, test results, attestation verification) |
| [`tests/`](tests/) | Offline tests for every part |
| [`tools/`](tools/) | One-off tools (the Alaska crosswalk generator) |
| [`LICENSES/`](LICENSES/), [`REUSE.toml`](REUSE.toml) | License texts, and REUSE licensing for files that cannot carry a header (data, reports, provenance records); authored files carry SPDX headers and a `Catalyst-File-ID` |
| [`.gitignore`](.gitignore) | Keeps data and report JSON/CSV versioned despite the repository-wide ignores |
| [`docs/`](docs/) | [User guide](docs/election-reporting.md) and engineering records ([ADR-001](docs/engineering/ADR-001-election-result-reporting.md), [ADR-002](docs/engineering/ADR-002-election-source-watcher.md), [provenance baseline](docs/engineering/provenance-baseline.json), [header equivalence](docs/engineering/header-equivalence-20261023.json), [attestation verification](docs/engineering/attestation-verification-20261023.json), [Nov 3 rehearsal](docs/engineering/nov3-rehearsal-20261023.json)) |

| Workflow | What it does |
|---|---|
| `election-source-watch.yml` | Every 30 minutes during a checking window, fetches new official snapshots into `data/` and opens a pull request |
| `election-result-reports.yml` | Runs the tests, builds and validates every election's report into `reports/`, and opens a pull request when they change |
| `election_provenance.yml` | Report-only: REUSE compliance, a version 4 `Catalyst-File-ID` on every authored file, and `scope-manifest.json` (path, kind, File-ID, SHA-256 of every file), plus test results; results in the job summary and an artifact, never failing the pull request. On pushes to main only, signs the scope manifest and the test results (GitHub artifact attestations) and verifies them |

## Run

From the repository root (EDT checked out next to it):

```bash
python -m pip install -r election-analysis/ci/requirements-provenance.txt
export PYTHONPATH="election-analysis:../engineering-docs-toolkit"
python -m unittest discover -s election-analysis/tests
python -m election_reporting \
  --config election-analysis/data/elections/washington/2026-08-04/election.toml \
  --output election-analysis/reports/washington/2026-08-04
python -m election_reporting.watch plan
```

See the [user guide](docs/election-reporting.md) for inputs, formulas, risk
bands, scope controls and the watcher.


The optional [Orange County official-evidence experiment](docs/engineering/ADR-007-official-queue-and-batch-evidence.md)
adds bounded PDF/CSV readers, original source files and a frozen conditional
comparison. Its official queue estimates still require acceptance/geographic
assumptions; a reported tally increment is not a verified correction-free batch.
It does not establish forecast calibration. Its source and validation limits
remain explicit in the independent review and accepted unvalidated outcome.


Frozen OC v1 must be reproduced from its original source and runtime, using
[`ci/reproduce_official_experiment.py`](ci/reproduce_official_experiment.py).
It verifies source hashes and installed pins, writes a new seal and compares the
full results while preserving the first run. See [ADR-008](docs/engineering/ADR-008-frozen-runtime-reproduction.md).
Current CI dependency upgrades do not rewrite historical protocols.
