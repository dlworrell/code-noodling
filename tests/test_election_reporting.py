from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from election_reporting.analysis import analyze_election
from election_reporting.config import load_config
from election_reporting.ingest import read_snapshot
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
            self.assertAlmostEqual(decision.risk.required_share or 0.0, 0.50926, places=4)
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
            self.assertEqual(decision.risk.level, "toss_up")

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

    def test_three_vote_margin_is_toss_up(self) -> None:
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
            self.assertEqual(decision.risk.level, "toss_up")

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
            quality_path = output / "edt" / "document" / "quality.json"
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            self.assertTrue(quality["publication_ready"])


if __name__ == "__main__":
    unittest.main()
