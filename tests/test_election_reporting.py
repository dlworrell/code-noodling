from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from election_reporting.analysis import analyze_election
from election_reporting.config import load_config
from election_reporting.ingest import normalize_contest_key, read_snapshot
from election_reporting.publish import publish


HEADER = (
    '"ContestOrder","Contest","Contest ID","Choice Order","Choice",'
    '"BallotsWith Contest","Votes"\n'
)


def result_rows(contest: str, contest_id: str, ballots: int, choices: list[tuple[str, int]]) -> str:
    lines = []
    for order, (choice, votes) in enumerate(choices, start=1):
        lines.append(
            f'"1.00","{contest}","{contest_id}","{order}",'
            f'"{choice}","{ballots:,}","{votes:,}"'
        )
    return HEADER + "\n".join(lines) + "\n"


def write_summary_xlsx(path: Path, reported_ballots: int = 100) -> None:
    rows = [
        ["Office Name", "Contest ID", "Ballot Name", "Choice ID", "Party", "Total"],
        [
            "U.S. Representative - Congressional District 8",
            "180108",
            "Ballots Cast",
            "",
            "",
            reported_ballots,
        ],
        ["U.S. Representative - Congressional District 8", "180108", "First", "first", "", 50],
        ["U.S. Representative - Congressional District 8", "180108", "Second", "second", "", 25],
        ["U.S. Representative - Congressional District 8", "180108", "Third", "third", "", 24],
        ["U.S. Representative - Congressional District 8", "180108", "Over Votes", "", "", 1],
        ["U.S. Representative - Congressional District 8", "180108", "Under Votes", "", "", 0],
    ]

    def cell(reference: str, value: object) -> str:
        if isinstance(value, int):
            return f'<c r="{reference}"><v>{value}</v></c>'
        return (
            f'<c r="{reference}" t="inlineStr"><is><t>{escape(str(value))}'
            "</t></is></c>"
        )

    row_xml = []
    for row_number, values in enumerate(rows, start=1):
        cells = "".join(
            cell(f"{chr(ord('A') + column)}{row_number}", value)
            for column, value in enumerate(values)
        )
        row_xml.append(f'<row r="{row_number}">{cells}</row>')
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(row_xml)}</sheetData></worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Summary Results" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


class ElectionReportingTests(unittest.TestCase):
    def test_csv_text_with_xls_name_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2026-08-06-results.csv.xls"
            path.write_text(
                result_rows(
                    "County Assessor (Vote for 1)",
                    "45",
                    1000,
                    [("Alpha", 400), ("Beta", 300), ("Gamma", 290)],
                ),
                encoding="utf-8",
            )
            snapshot = read_snapshot(path, "county")
            contest = next(iter(snapshot.contests.values()))
            self.assertEqual(contest.ballots, 1000)
            self.assertEqual(contest.choices[2].votes, 290)

    def test_statewide_summary_xlsx_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2026-08-07-statewide.xlsx"
            write_summary_xlsx(path)
            snapshot = read_snapshot(path, "washington")
            contest = snapshot.contests["us-house:8"]
            self.assertEqual(contest.ballots, 100)
            self.assertEqual(contest.reported_ballots, 100)
            self.assertEqual(contest.overvotes, 1)
            self.assertEqual(contest.undervotes, 0)
            self.assertEqual(
                [(choice.name, choice.votes) for choice in contest.choices],
                [("First", 50), ("Second", 25), ("Third", 24)],
            )

    def test_statewide_xlsx_reconciles_underreported_ballots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2026-08-07-statewide.xlsx"
            write_summary_xlsx(path, reported_ballots=90)
            contest = read_snapshot(path, "washington").contests["us-house:8"]
            self.assertEqual(contest.reported_ballots, 90)
            self.assertEqual(contest.ballots, 100)
            self.assertLessEqual(
                sum(choice.votes for choice in contest.choices),
                contest.ballots,
            )

    def test_county_and_statewide_names_share_canonical_keys(self) -> None:
        pairs = [
            (
                "U.S. Representative Congressional District No. 8 (Vote for 1)",
                "U.S. Representative - Congressional District 8",
            ),
            (
                "Legislative District No. 1 Representative Position No. 2 (Vote for 1)",
                "State Representative Pos. 2 - Legislative District 1",
            ),
            (
                "Justice Position No. 3 (Vote for 1)",
                "Justice Position #03 - Supreme Court",
            ),
        ]
        for county_name, statewide_name in pairs:
            with self.subTest(county_name=county_name):
                self.assertEqual(
                    normalize_contest_key(county_name),
                    normalize_contest_key(statewide_name),
                )

    def test_top_two_cutoff_uses_second_and_third(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "2026-08-06.csv").write_text(
                result_rows(
                    "County Assessor (Vote for 1)",
                    "45",
                    1000,
                    [("First", 450), ("Second", 275), ("Third", 265)],
                ),
                encoding="utf-8",
            )
            config_path = root / "election.toml"
            config_path.write_text(
                "schema_version = 1\n"
                "[election]\n"
                'title = "Test"\n'
                "[[sources]]\n"
                'id = "county"\n'
                'path = "inputs"\n'
                'jurisdiction = "Test County"\n'
                'scope = "county"\n'
                "priority = 10\n"
                "expected_final_ballots = 2000\n",
                encoding="utf-8",
            )
            analysis = analyze_election(load_config(config_path))
            decision = analysis.races[0].decisions[0]
            self.assertEqual(decision.kind, "top_two_cutoff")
            self.assertEqual(decision.current_side, "Second")
            self.assertEqual(decision.change_side, "Third")
            self.assertEqual(decision.margin, 10)
            self.assertAlmostEqual(
                decision.risk.required_share or 0.0,
                0.51111,
                places=4,
            )
            self.assertGreater(decision.risk.change_probability or 0.0, 0.25)
            self.assertLess(decision.risk.change_probability or 0.0, 0.35)
            self.assertEqual(decision.risk.level, "high")

    def test_history_records_a_lead_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "2026-08-05.csv").write_text(
                result_rows(
                    "Local Proposition 1 (Vote for 1)",
                    "9",
                    100,
                    [("Yes", 45), ("No", 47)],
                ),
                encoding="utf-8",
            )
            (inputs / "2026-08-06.csv").write_text(
                result_rows(
                    "Local Proposition 1 (Vote for 1)",
                    "9",
                    120,
                    [("Yes", 58), ("No", 57)],
                ),
                encoding="utf-8",
            )
            config_path = root / "election.toml"
            config_path.write_text(
                "schema_version = 1\n"
                "[election]\n"
                'title = "Test"\n'
                "[[sources]]\n"
                'id = "local"\n'
                'path = "inputs"\n'
                'jurisdiction = "Local"\n'
                'scope = "local"\n'
                "priority = 1\n"
                "expected_final_ballots = 150\n",
                encoding="utf-8",
            )
            decision = analyze_election(load_config(config_path)).races[0].decisions[0]
            self.assertTrue(decision.risk.lead_changed)
            self.assertEqual(decision.risk.level, "high")
            self.assertGreater(decision.risk.change_probability or 0.0, 0.25)
            self.assertEqual(decision.risk.reliability_grade, "Low")

    def test_county_slice_is_not_marked_controlling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "2026-08-06.csv").write_text(
                result_rows(
                    "U.S. Representative Congressional District No. 8 (Vote for 1)",
                    "83",
                    100,
                    [("First", 50), ("Second", 25), ("Third", 24)],
                ),
                encoding="utf-8",
            )
            config_path = root / "election.toml"
            config_path.write_text(
                "schema_version = 1\n"
                "[election]\n"
                'title = "Test"\n'
                "[[sources]]\n"
                'id = "county"\n'
                'path = "inputs"\n'
                'jurisdiction = "Test County"\n'
                'scope = "county"\n'
                "priority = 10\n",
                encoding="utf-8",
            )
            race = analyze_election(load_config(config_path)).races[0]
            self.assertFalse(race.controlling)
            self.assertIn("county-only slice", race.scope_note)
            self.assertIsNone(race.decisions[0].risk.change_probability)
            self.assertEqual(race.decisions[0].risk.reliability_score, 0)

    def test_abbreviated_justice_contest_uses_judicial_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "2026-08-06.csv").write_text(
                result_rows(
                    "Justice Position No. 1 (Vote for 1)",
                    "117",
                    100,
                    [("First", 55), ("Second", 25), ("Third", 19)],
                ),
                encoding="utf-8",
            )
            config_path = root / "election.toml"
            config_path.write_text(
                "schema_version = 1\n"
                "[election]\n"
                'title = "Test"\n'
                "[[sources]]\n"
                'id = "county"\n'
                'path = "inputs"\n'
                'jurisdiction = "Test County"\n'
                'scope = "county"\n'
                "priority = 10\n"
                "expected_final_ballots = 150\n",
                encoding="utf-8",
            )
            race = analyze_election(load_config(config_path)).races[0]
            self.assertEqual(race.rule, "wa_judicial")
            self.assertFalse(race.controlling)
            self.assertEqual(
                [decision.kind for decision in race.decisions],
                ["top_two_cutoff", "majority_status"],
            )

    def test_three_vote_margin_has_statistical_probability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "2026-08-06.csv").write_text(
                result_rows(
                    "Local Proposition 1 (Vote for 1)",
                    "9",
                    85,
                    [("Yes", 43), ("No", 40)],
                ),
                encoding="utf-8",
            )
            config_path = root / "election.toml"
            config_path.write_text(
                "schema_version = 1\n"
                "[election]\n"
                'title = "Test"\n'
                "[[sources]]\n"
                'id = "local"\n'
                'path = "inputs"\n'
                'jurisdiction = "Local"\n'
                'scope = "local"\n'
                "priority = 1\n"
                "expected_final_ballots = 136\n",
                encoding="utf-8",
            )
            decision = analyze_election(load_config(config_path)).races[0].decisions[0]
            self.assertEqual(decision.risk.level, "high")
            self.assertGreater(decision.risk.change_probability or 0.0, 0.25)
            self.assertLess(decision.risk.change_probability or 0.0, 0.40)

    def test_tied_boundary_has_symmetric_probability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "2026-08-06.csv").write_text(
                result_rows(
                    "Local Proposition 1 (Vote for 1)",
                    "9",
                    100,
                    [("Yes", 50), ("No", 50)],
                ),
                encoding="utf-8",
            )
            config_path = root / "election.toml"
            config_path.write_text(
                "schema_version = 1\n"
                "[election]\n"
                'title = "Test"\n'
                "[[sources]]\n"
                'id = "local"\n'
                'path = "inputs"\n'
                'jurisdiction = "Local"\n'
                'scope = "local"\n'
                "priority = 1\n"
                "expected_final_ballots = 150\n",
                encoding="utf-8",
            )
            risk = analyze_election(load_config(config_path)).races[0].decisions[0].risk
            self.assertAlmostEqual(
                risk.change_probability or 0.0,
                0.45405,
                places=5,
            )

    def test_edt_publication_writes_human_and_quality_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "2026-08-06.csv").write_text(
                result_rows(
                    "Local Proposition 1 (Vote for 1)",
                    "9",
                    100,
                    [("Yes", 51), ("No", 48)],
                ),
                encoding="utf-8",
            )
            config_path = root / "election.toml"
            config_path.write_text(
                "schema_version = 1\n"
                "[election]\n"
                'title = "Test"\n'
                "[[sources]]\n"
                'id = "local"\n'
                'path = "inputs"\n'
                'jurisdiction = "Local"\n'
                'scope = "local"\n'
                "priority = 1\n"
                "expected_final_ballots = 120\n",
                encoding="utf-8",
            )
            output = root / "reports"
            publish(analyze_election(load_config(config_path)), output)
            self.assertTrue((output / "election-analysis.md").exists())
            self.assertTrue((output / "election-analysis.html").exists())
            payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
            risk = payload["races"][0]["decisions"][0]["risk"]
            self.assertIsInstance(risk["change_probability"], float)
            self.assertGreater(risk["reliability_score"], 0)
            markdown = (output / "election-analysis.md").read_text(encoding="utf-8")
            self.assertIn("## At-a-glance decisions", markdown)
            self.assertIn("| Change probability | Reliability |", markdown)
            quality_path = output / "edt" / "document" / "quality.json"
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            self.assertTrue(quality["publication_ready"])


if __name__ == "__main__":
    unittest.main()
