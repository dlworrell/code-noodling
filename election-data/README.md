# Election snapshot inputs

Put official result exports in the source-specific directories below, then run the
`Election Result Reports` GitHub Actions workflow or the local command documented in
[`docs/election-reporting.md`](../docs/election-reporting.md).

```text
election-data/input/
├── king-county/
│   ├── 2026-08-06-1600.csv
│   └── 2026-08-07-1600.csv
└── washington/
    └── 2026-08-07-1700.csv
```

Requirements:

- Every filename must contain `YYYY-MM-DD`; an optional `HHMM` establishes the order
  of multiple reports on one day.
- `.csv`, `.csv.xls`, `.xls`, and `.txt` names are accepted when the content is UTF-8
  CSV text. A real binary Excel workbook must first be exported as CSV.
- Files in one directory must be successive snapshots from the same reporting source.
- Do not replace an older snapshot. Retaining it is what makes batch and lead-change
  analysis possible.

The parser expects the Washington VoteWA export columns `ContestOrder`, `Contest`,
`Contest ID`, `Choice Order`, `Choice`, `BallotsWith Contest`, and `Votes`. Extra
columns are retained in the source file but ignored by the analyzer.

The empty `washington/` directory is intentional. Adding a statewide or controlling
district export allows that higher-priority source to replace a King County-only slice
for matching multicounty contests.
