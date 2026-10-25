# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:43fed3b6-0de8-4b6e-a106-b1b52d4e5ae1

from __future__ import annotations

import json
import plistlib
import tempfile
import unittest
import zipfile
from dataclasses import replace
from datetime import datetime
from unittest import mock
from pathlib import Path
from xml.sax.saxutils import escape

from election_reporting.analysis import analyze_election
from election_reporting.config import load_config
from election_reporting.ingest import normalize_contest_key, read_snapshot
from election_reporting.model import ElectionReportError, ResultStatus, ResultStatusRecord
from election_reporting.publish import analysis_payload, facebook_summary, publish


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


def counting_config(path: Path):
    """Explicit synthetic counting evidence for the pre-existing arithmetic cases.

    This is test data, not an inference from filenames or a certification date.
    Lifecycle migration is tested separately against the actual schema-1 loader.
    """
    config = load_config(path)
    stamp = datetime.fromisoformat("2026-08-06T23:00:00+00:00")
    status = ResultStatus("counting", "plurality_count", "https://example.test/counting",
                          stamp, "synthetic-original-count-v1")
    records = tuple(
        ResultStatusRecord(source.source_id, read_snapshot(file, source.source_id).sha256, "*", status)
        for source in config.sources for file in sorted(source.path.glob("*.csv"))
    )
    return replace(config, report_mode="current", forecast_target="outstanding_ballots",
                   information_time=stamp, election_status=status, result_statuses=records)


def write_summary_xlsx(
    path: Path,
    reported_ballots: int = 100,
    extra_members: dict[str, str] | None = None,
) -> None:
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
        for name, payload in (extra_members or {}).items():
            archive.writestr(name, payload)


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

    def test_apple_binary_plist_is_rejected_with_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2026-08-06-1600.csv.xls"
            path.write_bytes(
                plistlib.dumps(
                    {"$archiver": "NSKeyedArchiver", "$version": 100000},
                    fmt=plistlib.FMT_BINARY,
                )
            )
            with self.assertRaisesRegex(
                ElectionReportError,
                r"Apple binary property list.*not election results.*Re-export",
            ):
                read_snapshot(path, "county")

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

    def test_unread_precinct_sheet_does_not_count_toward_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2026-08-07-statewide.xlsx"
            write_summary_xlsx(
                path,
                extra_members={"xl/worksheets/sheet2.xml": "<x/>" * 20_000},
            )
            with mock.patch(
                "election_reporting.ingest.MAX_XLSX_UNCOMPRESSED_BYTES", 16_384
            ):
                snapshot = read_snapshot(path, "washington")
            self.assertEqual(snapshot.contests["us-house:8"].ballots, 100)

    def test_oversized_summary_sheet_is_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2026-08-07-statewide.xlsx"
            write_summary_xlsx(path)
            with mock.patch(
                "election_reporting.ingest.MAX_XLSX_UNCOMPRESSED_BYTES", 512
            ):
                with self.assertRaisesRegex(
                    ElectionReportError, "uncompressed XLSX safety limit"
                ):
                    read_snapshot(path, "washington")

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
            analysis = analyze_election(counting_config(config_path))
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
            decision = analyze_election(counting_config(config_path)).races[0].decisions[0]
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
            race = analyze_election(counting_config(config_path)).races[0]
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
            race = analyze_election(counting_config(config_path)).races[0]
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
            decision = analyze_election(counting_config(config_path)).races[0].decisions[0]
            self.assertEqual(decision.risk.level, "high")
            self.assertGreater(decision.risk.change_probability or 0.0, 0.25)
            self.assertLess(decision.risk.change_probability or 0.0, 0.40)

    def test_unchanged_tie_is_symmetric_without_false_lead_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            for snapshot_date in ("2026-08-05", "2026-08-06"):
                (inputs / f"{snapshot_date}.csv").write_text(
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
            analysis = analyze_election(counting_config(config_path))
            risk = analysis.races[0].decisions[0].risk
            self.assertAlmostEqual(
                risk.change_probability or 0.0,
                0.45405,
                places=5,
            )
            self.assertFalse(risk.lead_changed)
            self.assertEqual(analysis.races[0].decisions[0].current_state, "Yes and No are tied.")
            social_report = facebook_summary(analysis)
            self.assertIn("Yes and No are tied at this boundary.", social_report)
            self.assertNotIn("leads No by 0 votes", social_report)

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
            publish(analyze_election(counting_config(config_path)), output)
            self.assertTrue((output / "election-analysis.md").exists())
            self.assertTrue((output / "election-analysis.html").exists())
            self.assertTrue((output / "facebook-summary.txt").exists())
            payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
            risk = payload["races"][0]["decisions"][0]["risk"]
            self.assertIsInstance(risk["change_probability"], float)
            self.assertGreater(risk["reliability_score"], 0)
            markdown = (output / "election-analysis.md").read_text(encoding="utf-8")
            self.assertIn("## At-a-glance decisions", markdown)
            self.assertIn("| Change probability | Reliability |", markdown)
            facebook = (output / "facebook-summary.txt").read_text(encoding="utf-8")
            self.assertIn("ELECTION RESULTS: WHAT COULD STILL CHANGE", facebook)
            self.assertIn("Local Proposition 1", facebook)
            self.assertIn("Modeled change probability:", facebook)
            self.assertIn("Reliability:", facebook)
            self.assertIn("not an official result, a race call", facebook)
            self.assertIn("Full race-by-race report and methodology:", facebook)
            self.assertNotIn("| Change probability |", facebook)
            quality_path = output / "edt" / "document" / "quality.json"
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            self.assertTrue(quality["publication_ready"])


class ResultStageTests(unittest.TestCase):
    """EA-01 acceptance cases use independently stated lifecycle expectations."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        inputs = self.root / "inputs"
        inputs.mkdir()
        for day, yes, no in (("05", 45, 47), ("06", 58, 57)):
            (inputs / f"2026-08-{day}.csv").write_text(
                result_rows("Local Proposition 1", "9", yes + no, [("Yes", yes), ("No", no)]),
                encoding="utf-8",
            )
        self.path = self.root / "election.toml"
        self.path.write_text(
            'schema_version = 1\n[election]\ntitle = "Stage fixture"\n'
            '[[sources]]\nid = "local"\npath = "inputs"\njurisdiction = "Local"\n'
            'scope = "local"\npriority = 1\ncertification_date = 2026-08-01\n'
            'expected_final_ballots = 150\n', encoding="utf-8",
        )
        self.active = counting_config(self.path)

    def with_latest_status(self, **changes):
        records = self.active.result_statuses
        return replace(self.active, result_statuses=(
            records[0], replace(records[1], status=replace(records[1].status, **changes)),
        ))

    def assert_withheld(self, config):
        with mock.patch("election_reporting.analysis._tempered_beta_binomial_probability",
                        side_effect=AssertionError("withheld forecasts must not execute the model")):
            analysis = analyze_election(config)
        race = analysis.races[0]
        risk = race.decisions[0].risk
        self.assertFalse(race.forecast_eligible)
        self.assertIsNone(risk.change_probability)
        self.assertIsNone(risk.estimated_remaining_votes)
        self.assertEqual(risk.reliability_score, 0)
        self.assertEqual(risk.level, "unknown")
        return analysis

    def test_legacy_and_elapsed_deadline_remain_unknown(self) -> None:
        legacy = load_config(self.path)
        analysis = self.assert_withheld(legacy)
        self.assertEqual(analysis.election_status.lifecycle, "unknown")
        self.assertEqual(analysis.races[0].result_status.lifecycle, "unknown")
        self.assertEqual(analysis.report_mode, "descriptive")
        # Adding a target/time cannot turn the passed August 1 deadline into evidence.
        current = replace(legacy, report_mode="current", forecast_target="outstanding_ballots",
                          information_time=self.active.information_time)
        self.assertIn("unknown", self.assert_withheld(current).races[0].forecast_reason)

    def test_lifecycle_matrix_only_counting_runs_count_model(self) -> None:
        for lifecycle in ("awaiting_results", "unknown", "final_tabulation", "certified",
                          "recount_pending", "recount_completed"):
            with self.subTest(lifecycle=lifecycle):
                analysis = self.assert_withheld(self.with_latest_status(lifecycle=lifecycle))
                self.assertIn(lifecycle, analysis.races[0].forecast_reason)
        active = analyze_election(self.active)
        self.assertTrue(active.races[0].forecast_eligible)
        self.assertIsInstance(active.races[0].decisions[0].risk.change_probability, float)

    def test_completed_election_blocks_counting_contest(self) -> None:
        config = replace(self.active, election_status=replace(self.active.election_status, lifecycle="certified"))
        self.assertIn("Election lifecycle is certified", self.assert_withheld(config).races[0].forecast_reason)

    def test_information_time_and_target_are_required(self) -> None:
        for target in ("none", "recount", "litigation"):
            with self.subTest(target=target):
                self.assert_withheld(replace(self.active, forecast_target=target))
        self.assert_withheld(replace(self.active, information_time=None))
        future = datetime.fromisoformat("2026-08-07T00:00:00+00:00")
        self.assertIn("postdates", self.assert_withheld(
            self.with_latest_status(evidence_as_of=future)
        ).races[0].forecast_reason)

    def test_distinct_stages_rounds_and_versions_are_not_batches(self) -> None:
        cases = [
            {"tabulation_stage": "raw_first_rankings"},
            {"tabulation_stage": "rcv_round", "round_id": "2"},
            {"tabulation_stage": "final_round", "lifecycle": "final_tabulation"},
            {"tabulation_stage": "recount", "lifecycle": "recount_completed"},
            {"result_version": "a-corrected-publication"},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                analysis = analyze_election(self.with_latest_status(**changes))
                race = analysis.races[0]
                self.assertEqual(len(race.decisions[0].margin_history), 1)
                self.assertIsNone(race.decisions[0].risk.latest_batch_votes)
                self.assertEqual(len(race.excluded_history), 1)
                self.assertEqual(len(analysis_payload(analysis)["inputs"]), 2)
        # Same RCV stage/version with different round identifiers is incompatible too.
        records = self.active.result_statuses
        statuses = tuple(replace(record, status=replace(record.status,
            tabulation_stage="rcv_round", round_id=str(index))) for index, record in enumerate(records, 1))
        race = analyze_election(replace(self.active, result_statuses=statuses)).races[0]
        self.assertEqual(len(race.decisions[0].margin_history), 1)

    def test_raw_rcv_majority_is_descriptive_and_never_a_final_winner(self) -> None:
        latest = self.root / "inputs" / "2026-08-06.csv"
        latest.write_text(result_rows("Maine CD2", "9", 402936,
            [("Golden", 196388), ("Theriault", 194130), ("Write-in", 421)]), encoding="utf-8")
        config = counting_config(self.path)
        config = replace(config, default_rule="rcv", result_statuses=tuple(
            replace(record, status=replace(record.status, tabulation_stage="raw_first_rankings"))
            for record in config.result_statuses
        ))
        analysis = self.assert_withheld(config)
        race = analysis.races[0]
        self.assertIn("50.23%", race.decisions[0].current_state)
        self.assertIn("not a final winner", race.scope_note)
        self.assertIn("raw_first_rankings", str(analysis_payload(analysis)))

    def test_metadata_cannot_silently_follow_changed_bytes(self) -> None:
        latest = self.root / "inputs" / "2026-08-06.csv"
        latest.write_text(latest.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ElectionReportError, "matches no input"):
            analyze_election(self.active)

    def test_schema_two_parses_dated_hash_bound_metadata(self) -> None:
        digest = self.active.result_statuses[-1].snapshot_sha256
        text = self.path.read_text().replace("schema_version = 1", "schema_version = 2")
        text += (
            '\n[[result_status]]\nsource_id = "local"\ncontest_id = "9"\n'
            f'snapshot_sha256 = "{digest}"\n'
            'lifecycle = "recount_completed"\ntabulation_stage = "recount"\n'
            'result_version = "official-recount"\nauthority_url = "https://example.test/recount"\n'
            'evidence_as_of = 2026-08-06T23:00:00Z\n'
        )
        self.path.write_text(text)
        config = load_config(self.path)
        self.assertEqual(config.result_statuses[0].status.lifecycle, "recount_completed")
        for invalid, message in (
            (text.replace('lifecycle = "recount_completed"', 'lifecycle = "finished"'), "lifecycle"),
            (text.replace('tabulation_stage = "recount"', 'tabulation_stage = "rcv_round"'), "round_id"),
            (text.replace('evidence_as_of = 2026-08-06T23:00:00Z', 'evidence_as_of = 2026-08-06T23:00:00'), "time zone"),
            (text.replace('https://example.test/recount', 'http://example.test/recount'), "HTTPS"),
            (text + text[text.index('[[result_status]]'):], "duplicates"),
        ):
            with self.subTest(message=message):
                self.path.write_text(invalid)
                with self.assertRaisesRegex(ElectionReportError, message):
                    load_config(self.path)

    def test_historical_publication_agrees_in_all_formats(self) -> None:
        config = replace(self.with_latest_status(lifecycle="recount_completed", tabulation_stage="recount"),
                         report_mode="historical", forecast_target="none")
        analysis = self.assert_withheld(config)
        output = self.root / "reports"
        publish(analysis, output)
        payload = json.loads((output / "analysis.json").read_text())
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(payload["report_mode"], "historical")
        self.assertIsNone(payload["races"][0]["decisions"][0]["risk"]["change_probability"])
        for filename in ("election-analysis.md", "election-analysis.html", "facebook-summary.txt"):
            text = (output / filename).read_text()
            self.assertIn("Report mode: historical", text)
            self.assertIn("recount_completed", text)
            self.assertIn("Forecast target: none", text)
            self.assertNotIn("WHAT COULD STILL CHANGE", text)
        social = (output / "facebook-summary.txt").read_text()
        self.assertIn("Unestimated probabilities are not numerical zero", social)
        self.assertNotIn("Modeled change probability: 0", social)
        self.assertNotIn("Very low", social)

    def test_actual_ld42_completed_recount_keeps_four_vote_margin(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data/elections/washington/2026-08-04/election.toml"
        analysis = self.assert_withheld(load_config(path))
        self.assertEqual(len(analysis.races), 244)
        self.assertEqual(sum(len(race.decisions) for race in analysis.races), 166)
        race = next(race for race in analysis.races if race.contest_id == "169896")
        self.assertEqual(race.result_status.lifecycle, "recount_completed")
        self.assertEqual(race.decisions[0].margin, 4)
        self.assertIsNone(race.decisions[0].risk.change_probability)
        self.assertIn("leads Eamonn Collins by 4 votes", facebook_summary(analysis))
        self.assertTrue(all(decision.risk.change_probability is None
                            for item in analysis.races for decision in item.decisions))


if __name__ == "__main__":
    unittest.main()
