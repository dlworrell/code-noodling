# Election result reporting

Watches official election results for Washington, Alaska and Maine, keeps every
dated snapshot with its provenance, and publishes reports estimating how likely
each current winner, top-two or top-four cutoff, judicial majority, or
ranked-choice first-choice majority is to change before certification. Every
probability carries a separate 0–100 evidence-reliability score.

**A triage report, not a call by an election authority and not a substitute for
certified results.**

## Layout

Everything for election reporting is in this directory; only the two workflows
live in [`.github/workflows/`](../.github/workflows/).

| Path | Contents |
|---|---|
| [`election_reporting/`](election_reporting/) | Python package: analyzer, EDT publisher, source watcher (`watch/`) and CI publisher (`ci/`) |
| [`data/`](data/) | Watcher settings (`watch.toml`), the Alaska crosswalk, and one directory per state and election with its configuration, snapshots and `SOURCES.json` ([details](data/README.md)) |
| [`reports/`](reports/) | Generated reports, one directory per state and election |
| [`tests/`](tests/) | Offline tests for every part |
| [`tools/`](tools/) | One-off tools (the Alaska crosswalk generator) |
| [`docs/`](docs/) | [User guide](docs/election-reporting.md) and engineering records ([ADR-001](docs/engineering/ADR-001-election-result-reporting.md), [ADR-002](docs/engineering/ADR-002-election-source-watcher.md)) |

| Workflow | What it does |
|---|---|
| `election-source-watch.yml` | Every 30 minutes during a checking window, fetches new official snapshots into `data/` and opens a pull request |
| `election-result-reports.yml` | Runs the tests, builds and validates every election's report into `reports/`, and opens a pull request when they change |

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
