# Election data

Each election has its own directory, named by election day:

```text
election-analysis/data/
├── watch.toml                 what the source watcher watches, and when
└── elections/
    └── washington/2026-11-03/
        ├── election.toml      analyzer configuration for this election
        ├── SOURCES.json       source URL, retrieval time and SHA-256 per snapshot
        ├── input/             plain-text CSV snapshots the analyzer reads
        │   ├── washington/    2026-11-03-2015.csv, …
        │   ├── king-county/
        │   └── pierce-county/ …
        ├── raw/               original downloads, when they are not the input
        │   ├── washington/    2026-11-03-2015.xlsx, …
        │   └── pierce-county/ 2026-11-03-2015.json.gz, …
        └── areas/washington/  county and precinct breakdowns (Washington)
            ├── county/        2026-11-03-2025.csv, …
            └── precinct/      2026-11-03-2025.csv.gz, …
```

Reports for an election are generated into `election-analysis/reports/<state>/<election day>/`.

## Automatic: the source watcher

`watch.toml` names each state and the counties (in Alaska, boroughs and census
areas) to follow; the watcher works out the
rest from the official results sites and adds new snapshots through a pull request
during the checking window. See
[Automatic source watching](../docs/election-reporting.md#automatic-source-watching).

```bash
PYTHONPATH=election-analysis python -m election_reporting.watch plan   # what watch.toml resolves to
```

`plan` also creates the election's `election.toml` when it does not exist, and only
ever appends sources for newly selected places, so hand edits are kept. Add a
turnout forecast (`expected_final_ballots`, `forecast_reliability`,
`forecast_basis`) to a source to enable remaining-ballot estimates; the probability
model is disabled for a source without one.

## By hand

Put official result exports in `elections/<state>/<election day>/input/<source>/`, then run
the `Election Result Reports` workflow or the local command in
[`election-analysis/docs/election-reporting.md`](../docs/election-reporting.md).

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

## County and precinct breakdowns

For Washington the watcher also downloads the Secretary of State's **Summary and
Precinct Results Excel** workbook on each run (`precinct_results = true`, the
default for Washington in `watch.toml`). The workbook is large (21.4 MB for the
2026 primary, 242 MB unpacked, 738,431 precinct rows) and is **not** stored; its
URL, size and SHA-256 are recorded in the `washington-areas` record of
`SOURCES.json`, along with the two files derived from it:

| File | Contents | Size (2026 primary) |
|---|---|---|
| `areas/washington/county/<stamp>.csv` | `Area`, `Office Name`, `Contest ID`, `Choice`, `Votes` for every contest in every county (the workbook's county Summary Results sheet), plus `Summary Results` rows holding the statewide totals from the All Results workbook of the same run. | 573 KB |
| `areas/washington/precinct/<stamp>.csv.gz` | `Precinct`, `County`, `Office Name`, `Contest ID`, `Choice`, `Votes`, non-zero rows only, gzipped reproducibly (no timestamp). | 1.8 MB |

`Choice` includes the `Ballots Cast`, `Over Votes` and `Under Votes` rows. The
workbook's precinct sheet is read as a stream, under a 1 GiB unpacked-size budget
and a 5-million-row budget; the download itself is capped at 128 MiB.

`election.toml` declares the files with one `[[areas]]` table per level; `plan`
adds them for Washington elections:

```toml
[[areas]]
source = "washington"            # a configured [[sources]] id
level = "county"                 # county or precinct
path = "areas/washington/county" # relative to the election directory

[[areas]]
source = "washington"
level = "precinct"
path = "areas/washington/precinct"
```

Each report then has a **Geography** section and a `county-breakdown.csv`:

- **Reconciliation.** County sums are checked against the statewide totals, and
  precinct sums against county totals, for the latest snapshot. Any difference is
  listed as a data-quality finding; nothing is adjusted. Contests are keyed by
  contest ID and office name, because local contest IDs repeat across counties.
  The statewide sheet words some office names differently (`Legislative District 3
  - State Representative Pos. 1`), so a statewide total is matched by contest ID
  only when both names have the same words.
- **Known gap.** The state's 2026 primary precinct sheet has no precinct rows for
  King County (151 contest-county pairs). The report lists this as missing
  precinct detail, not as a mismatch; King County's own export remains the source
  for its contests.
- **Changes.** With two snapshots, the counties and precincts that added the most
  ballots, and for the closest races the areas whose margin moved most.
- **Remaining ballots.** No official per-county remaining-ballot figure is in the
  results data, so counts are declared by hand from county ballot reports:

  ```toml
  [[remaining_ballots]]
  county = "King County"           # as named in the county file
  ballots = 120000
  as_of = 2026-11-04
  basis = "King County ballots-on-hand report, Nov 4 5 p.m."
  ```

  For each close race the county with the most declared remaining ballots is
  checked: whether its remaining ballots could change the outcome on their own,
  and whether they would at the county's current split. A count older than the
  latest county snapshot is noted. Without any counts the report says no county
  can be assessed.
