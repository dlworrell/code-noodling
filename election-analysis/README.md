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
See [ADR-003](docs/engineering/ADR-003-result-lifecycle-and-stages.md) for migration.

**Experimental and uncalibrated.** The Washington primary report is a historical
audit correction. The remaining audit issues are tracked in
[EA-AUD-2026-10-23 #45](https://github.com/dlworrell/code-noodling/issues/45).

**A triage report, not a call by an election authority and not a substitute for
certified results.**

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
export PYTHONPATH="election-analysis:../engineering-docs-toolkit"
python -m unittest discover -s election-analysis/tests
python -m election_reporting \
  --config election-analysis/data/elections/washington/2026-08-04/election.toml \
  --output election-analysis/reports/washington/2026-08-04
python -m election_reporting.watch plan
```

See the [user guide](docs/election-reporting.md) for inputs, formulas, risk
bands, scope controls and the watcher.
