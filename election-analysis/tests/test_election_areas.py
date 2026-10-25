# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:7fd4c21f-f965-413f-a1bd-55f0d8859c4e

"""County and precinct breakdowns: parsing, streaming budgets, reconciliation, analysis (#14)."""

from __future__ import annotations

import csv
import gzip
import io
import tempfile
import unittest
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from election_reporting import areas
from election_reporting.analysis import analyze_election
from election_reporting.config import load_config
from election_reporting.ingest import iter_xlsx_sheet_rows
from election_reporting.model import ElectionReportError
from election_reporting.publish import build_edom, county_breakdown_csv

HEAD = ["Precinct", "County", "Office Name", "Contest ID", "Ballot Name", "Choice ID", "Party", "Total"]
GOV = ("U.S. Representative - Congressional District 3", "179111")


def column(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(65 + rest) + letters
    return letters


def workbook(sheets: dict[str, list[list[object]]]) -> bytes:
    """A minimal multi-sheet XLSX using inline strings."""

    def cell(ref: str, value: object) -> str:
        if isinstance(value, int):
            return f'<c r="{ref}"><v>{value}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        entries, rels = [], []
        for number, (name, rows) in enumerate(sheets.items(), start=1):
            entries.append(f'<sheet name="{name}" sheetId="{number}" r:id="rId{number}"/>')
            rels.append(
                f'<Relationship Id="rId{number}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                f'relationships/worksheet" Target="worksheets/sheet{number}.xml"/>'
            )
            body = "".join(
                f'<row r="{r}">' + "".join(cell(f"{column(c)}{r}", v) for c, v in enumerate(row) if v != "") + "</row>"
                for r, row in enumerate(rows, start=1)
            )
            archive.writestr(
                f"xl/worksheets/sheet{number}.xml",
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f"<sheetData>{body}</sheetData></worksheet>",
            )
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{''.join(entries)}</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rels) + "</Relationships>",
        )
    return buffer.getvalue()


def precinct_rows(counts: dict[tuple[str, str], tuple[int, int, int]]) -> list[list[object]]:
    """counts: (precinct, county) -> (ballots, Alpha, Beta)."""
    rows: list[list[object]] = [HEAD]
    for (precinct, county), (ballots, alpha, beta) in counts.items():
        rows.append([precinct, county, *GOV, "Ballots Cast", "", "", ballots])
        rows.append([precinct, county, *GOV, "Alpha", "179111-Alpha", "(Prefers A)", alpha])
        rows.append([precinct, county, *GOV, "Beta", "179111-Beta", "(Prefers B)", beta])
    return rows


def county_rows(counts: dict[str, tuple[int, int, int]]) -> list[list[object]]:
    rows: list[list[object]] = [["County", "Office Name", "Contest ID", "Ballot Name", "Choice ID", "Party", "Total"]]
    for county, (ballots, alpha, beta) in counts.items():
        rows.append([county, *GOV, "Ballots Cast", "", "", ballots])
        rows.append([county, *GOV, "Alpha", "179111-Alpha", "(Prefers A)", alpha])
        rows.append([county, *GOV, "Beta", "179111-Beta", "(Prefers B)", beta])
    return rows


def statewide(ballots: int, alpha: int, beta: int) -> bytes:
    """An All Results workbook; the statewide sheet words the office differently."""
    office = GOV[0]
    return workbook({"Summary Results": [
        ["Office Name", "Contest ID", "Ballot Name", "Choice ID", "Party", "Total"],
        [office, GOV[1], "Ballots Cast", "", "", ballots],
        [office, GOV[1], "ALPHA", "179111-Alpha", "(Prefers A)", alpha],
        [office, GOV[1], "Beta", "179111-Beta", "(Prefers B)", beta],
    ]})


PRECINCTS = {("P1", "Clark County"): (60, 40, 20), ("P2", "Clark County"): (40, 10, 30),
             ("P3", "Cowlitz County"): (50, 20, 30)}
COUNTIES = {"Clark County": (100, 50, 50), "Cowlitz County": (50, 20, 30)}


def full_workbook(counties=COUNTIES, precincts=PRECINCTS) -> bytes:
    return workbook({"Summary Results": county_rows(counties), "Precinct Results": precinct_rows(precincts)})


class StreamingTests(unittest.TestCase):
    def test_rows_stream_with_row_and_byte_budgets(self) -> None:
        raw = full_workbook()
        rows = list(iter_xlsx_sheet_rows(Path("w.xlsx"), raw, "Precinct Results"))
        self.assertEqual(rows[0][:2], ["Precinct", "County"])
        self.assertEqual(len(rows), 1 + 3 * len(PRECINCTS))
        with self.assertRaisesRegex(ElectionReportError, "5-row budget"):
            list(iter_xlsx_sheet_rows(Path("w.xlsx"), raw, "Precinct Results", max_rows=5))
        with self.assertRaisesRegex(ElectionReportError, "100-byte streamed sheet budget"):
            list(iter_xlsx_sheet_rows(Path("w.xlsx"), raw, "Precinct Results", max_bytes=100))

    def test_missing_sheet(self) -> None:
        with self.assertRaisesRegex(ElectionReportError, "no Precinct Results worksheet"):
            list(iter_xlsx_sheet_rows(Path("w.xlsx"), workbook({"Other": [["x"]]}), "Precinct Results"))


class ConversionTests(unittest.TestCase):
    def test_precinct_workbook_with_statewide_totals(self) -> None:
        conversion = areas.convert_area_workbook(full_workbook(), label="Results.xlsx", statewide=statewide(150, 70, 80))
        self.assertEqual((conversion.counties, conversion.precincts, conversion.contests, conversion.statewide_contests), (2, 3, 1, 1))
        rows = list(csv.DictReader(io.StringIO(conversion.county_csv)))
        self.assertEqual([r["Area"] for r in rows[:3]], ["Summary Results"] * 3)
        precinct = gzip.decompress(conversion.precinct_csv_gz).decode().splitlines()
        self.assertEqual(precinct[0], ",".join(areas.PRECINCT_COLUMNS))
        self.assertEqual(len(precinct), 1 + 9)
        again = areas.convert_area_workbook(full_workbook(), label="Results.xlsx", statewide=statewide(150, 70, 80))
        self.assertEqual(again.precinct_csv_gz, conversion.precinct_csv_gz)  # reproducible gzip

    def test_all_results_precinct_sheet_is_a_county_breakdown(self) -> None:
        rows = [["Precinct", "Office Name", "Contest ID", "Ballot Name", "Choice ID", "Party", "Total"]]
        rows += [["Clark County", *GOV, "Ballots Cast", "", "", 100], ["Clark County", *GOV, "Alpha", "a", "", 50]]
        conversion = areas.convert_area_workbook(workbook({"Precinct Results": rows}), label="All.xlsx")
        self.assertIsNone(conversion.precinct_csv_gz)
        self.assertEqual(conversion.counties, 1)

    def test_bad_counts_and_missing_columns_are_errors(self) -> None:
        bad = county_rows(COUNTIES)
        bad[1][-1] = "-3"
        with self.assertRaisesRegex(ElectionReportError, "non-negative whole number"):
            areas.convert_area_workbook(workbook({"Summary Results": bad}), label="x.xlsx")
        with self.assertRaisesRegex(ElectionReportError, "lacks columns: total"):
            areas.convert_area_workbook(workbook({"Summary Results": [["County", "Office Name", "Contest ID", "Ballot Name"]]}), label="x.xlsx")
        with self.assertRaisesRegex(ElectionReportError, "no county or precinct breakdown"):
            areas.convert_area_workbook(workbook({"Other": [["x"]]}), label="x.xlsx")


class ElectionDir:
    """An election directory with a statewide input snapshot and area snapshots."""

    def __init__(self, root: Path, remaining: str = "") -> None:
        self.root = root
        (root / "input" / "washington").mkdir(parents=True)
        (root / "election.toml").write_text(
            'schema_version = 1\n[election]\ntitle = "t"\n[analysis]\ndefault_rule = "winner"\n'
            '[[sources]]\nid = "washington"\npath = "input/washington"\n'
            'jurisdiction = "Washington State"\nscope = "state"\npriority = 100\n'
            '[[areas]]\nsource = "washington"\nlevel = "county"\npath = "areas/washington/county"\n'
            '[[areas]]\nsource = "washington"\nlevel = "precinct"\npath = "areas/washington/precinct"\n' + remaining
        )

    def statewide_input(self, stamp: str, alpha: int, beta: int) -> None:
        lines = ["ContestOrder,Contest,Contest ID,Choice Order,Choice,BallotsWith Contest,Votes"]
        lines += [f"1,{GOV[0]},{GOV[1]},1,Alpha,{alpha + beta},{alpha}", f"1,{GOV[0]},{GOV[1]},2,Beta,{alpha + beta},{beta}"]
        (self.root / "input" / "washington" / f"{stamp}.csv").write_text("\n".join(lines) + "\n")

    def areas(self, stamp: str, counties, precincts, totals) -> None:
        conversion = areas.convert_area_workbook(full_workbook(counties, precincts), label="R.xlsx", statewide=statewide(*totals))
        for level, data in (("county", conversion.county_csv.encode()), ("precinct", conversion.precinct_csv_gz)):
            directory = self.root / "areas" / "washington" / level
            directory.mkdir(parents=True, exist_ok=True)
            (directory / (f"{stamp}.csv" if level == "county" else f"{stamp}.csv.gz")).write_bytes(data)


class ReconciliationTests(unittest.TestCase):
    def snapshots(self, counties=COUNTIES, precincts=PRECINCTS, totals=(150, 70, 80)):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        election = ElectionDir(Path(tmp.name))
        election.areas("2026-11-04-0100", counties, precincts, totals)
        base = Path(tmp.name) / "areas" / "washington"
        county = areas.read_area_file(base / "county" / "2026-11-04-0100.csv", "washington", "county")
        precinct = areas.read_area_file(base / "precinct" / "2026-11-04-0100.csv.gz", "washington", "precinct")
        return areas.reconcile(county, precinct)

    def test_consistent_files_reconcile(self) -> None:
        result = self.snapshots()
        self.assertEqual(result["county_vs_statewide"]["checked"], 3)  # ALPHA matches Alpha despite case
        self.assertEqual(result["county_vs_statewide"]["mismatches"], [])
        self.assertEqual(result["precinct_vs_county"]["mismatches"], [])
        self.assertEqual(result["precinct_vs_county"]["checked"], 6)

    def test_mismatches_are_reported_not_accepted(self) -> None:
        result = self.snapshots(totals=(151, 70, 81))
        statewide = result["county_vs_statewide"]["mismatches"]
        self.assertEqual({(m["choice"], m["difference"]) for m in statewide}, {("ballots cast", -1), ("beta", -1)})
        precincts = dict(PRECINCTS)
        precincts[("P2", "Clark County")] = (40, 11, 30)
        result = self.snapshots(precincts=precincts)
        [mismatch] = result["precinct_vs_county"]["mismatches"]
        self.assertEqual((mismatch["area"], mismatch["choice"], mismatch["difference"]), ("Clark County", "alpha", 1))

    def test_a_county_without_precinct_rows_is_missing_detail(self) -> None:
        precincts = {k: v for k, v in PRECINCTS.items() if k[1] != "Cowlitz County"}
        result = self.snapshots(precincts=precincts)
        self.assertEqual(result["precinct_vs_county"]["mismatches"], [])
        [missing] = result["precinct_vs_county"]["precinct_detail_missing"]
        self.assertEqual((missing["county"], missing["county_ballots"]), ("Cowlitz County", 50))


class AnalysisTests(unittest.TestCase):
    def election(self, remaining: str = "") -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        election = ElectionDir(Path(tmp.name), remaining)
        election.statewide_input("2026-11-04-0100", 70, 80)
        election.areas("2026-11-04-0100", COUNTIES, PRECINCTS, (150, 70, 80))
        election.statewide_input("2026-11-05-1700", 100, 95)
        later_counties = {"Clark County": (130, 70, 60), "Cowlitz County": (65, 30, 35)}
        later_precincts = {("P1", "Clark County"): (90, 60, 30), ("P2", "Clark County"): (40, 10, 30),
                           ("P3", "Cowlitz County"): (65, 30, 35)}
        election.areas("2026-11-05-1700", later_counties, later_precincts, (195, 100, 95))
        return Path(tmp.name)

    def reviewed(self, path):
        from dataclasses import replace
        config = load_config(path)
        return replace(config, audit_contract={"schema_version": 1, "election_id": "synthetic:areas", "identities": [
            {"source_id": "washington", "contest_id": GOV[1], "source_contest_name": GOV[0],
             "canonical_id": "synthetic:cd3", "jurisdiction_id": "synthetic-district", "office": GOV[0],
             "authority_url": "https://example.test/reviewed-crosswalk", "reviewed": True,
             "jurisdictions": ["Clark County", "Cowlitz County"], "coverage_complete": True, "role": "controlling",
             "area_aliases": [{"contest_id": GOV[1], "name": GOV[0], "jurisdiction": county}
                              for county in ["Clark County", "Cowlitz County"]]}]})

    def test_breakdown_changes_and_report(self) -> None:
        root = self.election()
        analysis = analyze_election(self.reviewed(root / "election.toml"))
        result = analysis.areas
        self.assertEqual(result.reconciliation["county_vs_statewide"]["mismatches"], [])
        [race] = result.breakdown
        self.assertEqual([row["county"] for row in race["counties"]], ["Clark County", "Cowlitz County"])
        self.assertEqual(race["counties"][0]["votes"], {"Alpha": 70, "Beta": 60})
        self.assertEqual(result.county_changes["largest_new_ballots"][0], {"area": "Clark County", "new_ballots": 30})
        swings = result.precinct_changes["close_races"][0]["swings"]
        self.assertEqual(swings[0], {"area": "P1 (Clark County)", "new_ballots": 30, "margin_change": 10})
        self.assertIn("No county remaining-ballot counts are configured", result.notes[0])
        csv_text = county_breakdown_csv(analysis)
        self.assertIn("Clark County,130,Alpha,70,Alpha", csv_text)
        edom = build_edom(analysis)
        texts = [child["text"] for child in edom["root"]["children"]]
        self.assertIn("Geography: county and precinct results", texts)
        self.assertTrue(any("0 mismatches" in t for t in texts))

    def test_remaining_ballot_flags(self) -> None:
        remaining = (
            '[[remaining_ballots]]\ncounty = "Clark County"\nballots = 40\nas_of = 2026-11-05\nbasis = "test"\n'
            '[[remaining_ballots]]\ncounty = "Cowlitz County"\nballots = 10\nas_of = 2026-11-04\nbasis = "test"\n'
        )
        analysis = analyze_election(self.reviewed(self.election(remaining) / "election.toml"))
        [flag] = analysis.areas.remaining
        self.assertIn("forecast withheld", flag["status"])
        self.assertNotIn("needed_share", flag)
        self.assertTrue(any("Cowlitz County" in note for note in analysis.areas.notes))  # stale count

    def test_area_config_validation(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        base = ('schema_version = 1\n[election]\ntitle = "t"\n[[sources]]\nid = "washington"\npath = "i"\n'
                'jurisdiction = "W"\nscope = "state"\npriority = 100\n')
        for extra, message in (
            ('[[areas]]\nsource = "nope"\nlevel = "county"\npath = "a"\n', "not a configured source"),
            ('[[areas]]\nsource = "washington"\nlevel = "ward"\npath = "a"\n', "level must be one of"),
            ('[[areas]]\nsource = "washington"\nlevel = "county"\npath = "../a"\n', "inside the election directory"),
            ('[[remaining_ballots]]\ncounty = "X"\nballots = -1\nas_of = 2026-11-05\nbasis = "b"\n', "non-negative"),
        ):
            (root / "election.toml").write_text(base + extra)
            with self.assertRaisesRegex(ElectionReportError, message):
                load_config(root / "election.toml")


if __name__ == "__main__":
    unittest.main()
