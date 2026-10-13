# Election snapshot inputs

Put official result exports in the source-specific directories below, then run the
`Election Result Reports` GitHub Actions workflow or the local command documented in
[`docs/election-reporting.md`](../docs/election-reporting.md).

```text
election-data/input/
├── king-county/
│   ├── 2026-08-06-1600.csv.xls
│   └── …newer-dated-snapshot.csv.xls
└── washington/
    ├── 2026-08-07-0118.xlsx
    ├── 2026-08-08-0018.xlsx
    └── …newer-dated-snapshot.xlsx
```

Requirements:

- Every filename must contain `YYYY-MM-DD`; an optional `HHMM` establishes the order
  of multiple reports on one day.
- `.csv`, `.csv.xls`, `.xls`, and `.txt` names are accepted when the content is UTF-8
  CSV text.
- `.xlsx` is accepted for the official Washington `All Results` workbook. The parser
  reads `Summary Results` directly with the Python standard library; it ignores the
  much larger precinct sheet because the summary already contains controlling totals.
- Files in one directory must be successive snapshots from the same reporting source.
- Do not replace an older snapshot. Retaining it is what makes batch and lead-change
  analysis possible.

Each source in `election.toml` may declare `expected_final_ballots`, a
`forecast_reliability` score from 0 through 100, and a human-readable `forecast_basis`.
The probability model is disabled when no expected-final count is available.

The parser expects the Washington VoteWA export columns `ContestOrder`, `Contest`,
`Contest ID`, `Choice Order`, `Choice`, `BallotsWith Contest`, and `Votes`. Extra
columns are retained in the source file but ignored by the analyzer.

For a statewide XLSX workbook, the required summary columns are `Office Name`,
`Contest ID`, `Ballot Name`, `Choice ID`, and `Total`. `Ballots Cast` supplies each
reported contest denominator; `Over Votes` and `Under Votes` are not treated as
choices. If the vote, overvote, and undervote sum is larger than `Ballots Cast`, the
parser uses that auditable sum and discloses both values in the generated report.

The committed Washington workbook is the current statewide snapshot. Adding a newer
dated statewide or controlling-district export lets the higher-priority source replace
the matching King County slice while preserving prior snapshots for trend analysis.
