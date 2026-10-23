# Election data

Each election has its own directory, named by election day:

```text
election-data/
├── watch.toml                 what the source watcher watches, and when
└── elections/
    └── 2026-11-03/
        ├── election.toml      analyzer configuration for this election
        ├── SOURCES.json       source URL, retrieval time and SHA-256 per snapshot
        ├── input/             plain-text CSV snapshots the analyzer reads
        │   ├── washington/    2026-11-03-2015.csv, …
        │   ├── king-county/
        │   └── pierce-county/ …
        └── raw/               original downloads, when they are not the input
            ├── washington/    2026-11-03-2015.xlsx, …
            └── pierce-county/ 2026-11-03-2015.json.gz, …
```

Reports for an election are generated into `reports/elections/<state>/<election day>/`.

## Automatic: the source watcher

`watch.toml` names a state and the counties to follow; the watcher works out the
rest from the official results sites and adds new snapshots through a pull request
during the checking window. See
[Automatic source watching](../docs/election-reporting.md#automatic-source-watching).

```bash
python -m election_reporting.watch plan     # what watch.toml resolves to
```

`plan` also creates the election's `election.toml` when it does not exist, and only
ever appends sources for newly selected places, so hand edits are kept. Add a
turnout forecast (`expected_final_ballots`, `forecast_reliability`,
`forecast_basis`) to a source to enable remaining-ballot estimates; the probability
model is disabled for a source without one.

## By hand

Put official result exports in `elections/<state>/<election day>/input/<source>/`, then run
the `Election Result Reports` workflow or the local command in
[`docs/election-reporting.md`](../docs/election-reporting.md).

- Every filename must contain `YYYY-MM-DD`; an optional `HHMM` orders multiple
  reports on one day.
- `.csv`, `.csv.xls`, `.xls`, and `.txt` names are accepted when the content is
  UTF-8 CSV text. Commit the downloaded export itself. On iOS/macOS, copying a file
  out of the Files app or Safari downloads can produce an Apple binary property list
  (starting with `bplist00`) under the export's name; the parser rejects such files
  and asks for a fresh export.
- `.xlsx` is accepted for the official Washington `All Results` workbook, but prefer
  converting it: keep the workbook in `raw/` and the CSV from
  `election_reporting.watch.csvformat.from_summary_xlsx` in `input/`, so the analyzed
  input is plain text. The conversion gives identical analysis.
- Files in one directory must be successive snapshots from the same source.
- Never replace an older snapshot; trend and lead-change analysis depend on it.
  Record each file's source and retrieval time in `SOURCES.json`.

## Input columns

CSV input needs `ContestOrder`, `Contest`, `Contest ID`, `Choice Order`, `Choice`,
`BallotsWith Contest`, and `Votes` (the King County export columns). The watcher's
CSV adds optional columns:

| Column | Used by the analyzer |
|---|---|
| `Contest Overvotes`, `Contest Undervotes` | Yes. Blank means not reported. With both present, the contest denominator is never below votes plus overvotes plus undervotes. |
| `Counties` | Yes. A contest that spans several counties is not treated as controlling when it comes from a county source. |
| `Party`, `Jurisdiction`, `County`, `Write-in` | No; kept for people and later analysis. |

For a statewide XLSX workbook, the required `Summary Results` columns are
`Office Name`, `Contest ID`, `Ballot Name`, `Choice ID`, and `Total`.
`Ballots Cast` supplies each contest denominator; `Over Votes` and `Under Votes`
are not choices. If votes plus overvotes plus undervotes exceed `Ballots Cast`, the
parser uses that auditable sum and discloses both values in the report.

The statewide source has a higher priority than county sources, so for a contest
both report, the statewide totals control and county history is kept.
