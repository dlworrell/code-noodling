# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:e075c51b-4526-447d-a618-8a3d3396c651

"""Readable reports (R1-R7) and votewa count times (C1-C3), offline.

R1 a withheld summary never prints a heading with nothing under it;
R2 the closest-races list uses one report per contest, across spellings;
R3 ranked-choice contests are described by first choices, never a winner;
R4 gate codes are translated into plain reasons, counted per contest;
R5 the HTML has real headings and tables, and escapes source text;
R6 a partial share never rounds up to 100.0%;
R7 county and partial reports are grouped after the controlling contests.
C1 a votewa capture's election.lastUpdated becomes the count-update time;
C2 a raw file that no longer matches the ledger hash is ignored;
C3 a count time later than the retrieval is ignored.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from election_reporting.evidence import votewa_count_time
from election_reporting.html_report import edom_to_html
from election_reporting.model import ChoiceResult, ElectionAnalysis, RaceAnalysis
from election_reporting.publish import _close_races, _share, _withholding_reasons, build_edom, facebook_summary


def race(contest: str, votes: dict[str, int], *, source: str = "state", controlling: bool = False,
         rule: str = "winner", reason: str = "identity; remaining_pool; Descriptive report: withheld.") -> RaceAnalysis:
    choices = tuple(ChoiceResult(name, count, order) for order, (name, count) in enumerate(votes.items()))
    total = sum(votes.values())
    return RaceAnalysis(
        contest_id=contest, contest=contest, contest_order=0.0, source_id=source, jurisdiction=source.title(),
        snapshot="2026-11-05T10:00", ballots=total, reported_ballots=total, overvotes=0, undervotes=0,
        valid_votes=total, rule=rule, controlling=controlling, scope_note="test", choices=choices,
        decisions=(), overall_risk="not_applicable", forecast_reason=reason,
    )


def analysis(*races: RaceAnalysis) -> ElectionAnalysis:
    return ElectionAnalysis(title="Test General", as_of="2026-11-05T10:00", races=races, snapshots=())


class ReadableReportTests(unittest.TestCase):
    def test_r1_no_empty_headings(self) -> None:
        text = facebook_summary(analysis())
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.isupper() and len(line) > 3 and not line.startswith("🗳"):
                with self.subTest(heading=line):
                    self.assertTrue(lines[index + 1].strip(), f"{line} has nothing under it")
        self.assertNotIn("DESCRIPTIVE MARGINS", text)

    def test_r2_one_report_per_contest_across_spellings(self) -> None:
        county = race("Legislative District No. 32 State Senator (Vote for 1)", {"A": 60, "B": 59}, source="king-county")
        state = race("Legislative District 32 - State Senator", {"B": 600, "A": 500})
        rows = _close_races(analysis(county, state))
        self.assertEqual([row[0] for row in rows], [state])
        self.assertIn("B leads A by 100 votes", facebook_summary(analysis(county, state)))

    def test_r3_ranked_choice_is_first_choices_only(self) -> None:
        text = facebook_summary(analysis(race("U.S. Senator", {"P": 510, "S": 490}, rule="rcv")))
        self.assertIn("P leads S by 20 votes in first choices (ranked-choice tabulation not modeled)", text)

    def test_r4_plain_reasons(self) -> None:
        reasons = dict(_withholding_reasons(analysis(race("X", {"A": 2, "B": 1}), race("Y", {"A": 3, "B": 1}))))
        self.assertEqual(reasons["there is no estimate of ballots left to count"], 2)
        self.assertEqual(reasons["the contest has not been matched to a reviewed contest list"], 2)
        self.assertFalse(any("remaining_pool" in reason for reason in reasons))

    def test_r5_html_headings_tables_and_escaping(self) -> None:
        html = edom_to_html(build_edom(analysis(race("<script>alert(1)</script>", {"A": 2, "B": 1}))), "T")
        self.assertIn("<h2", html)
        self.assertIn("<table>", html)
        self.assertIn("<style>", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("|---|", html)

    def test_r6_share_never_rounds_to_whole(self) -> None:
        self.assertEqual(_share(99_996, 100_000), ">99.9%")
        self.assertEqual(_share(1, 100_000), "<0.1%")
        self.assertEqual(_share(5, 10), "50.0%")
        self.assertEqual(_share(1, 0), "—")

    def test_r7_partial_reports_grouped_after_controlling(self) -> None:
        edom = build_edom(analysis(race("County slice", {"A": 2, "B": 1}), race("Whole", {"A": 3, "B": 1}, controlling=True)))
        texts = [child["text"] for child in edom["root"]["children"]]
        self.assertLess(texts.index("Whole"), texts.index("County and partial reports"))
        self.assertLess(texts.index("County and partial reports"), texts.index("County slice (State portion)"))


class VotewaCountTimeTests(unittest.TestCase):
    def capture(self, last_updated: str, retrieved: str = "2026-11-05T21:14:40+00:00") -> tuple[Path, dict]:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        body = gzip.compress(json.dumps({"election": {"lastUpdated": last_updated, "asOf": retrieved}}).encode())
        (directory / "raw" / "x").mkdir(parents=True)
        (directory / "raw/x/1.json.gz").write_bytes(body)
        return directory, {"kind": "votewa-json", "raw": "raw/x/1.json.gz", "raw_sha256": hashlib.sha256(body).hexdigest(),
                           "retrieved_at": retrieved}

    def test_c1_last_updated_is_the_count_time(self) -> None:
        directory, record = self.capture("2026-11-04T04:22:16.6392716Z")
        self.assertEqual(votewa_count_time(directory, record), "2026-11-04T04:22:16.639271+00:00")
        self.assertIsNone(votewa_count_time(directory, {**record, "kind": "state-xlsx"}))

    def test_c2_hash_mismatch_is_ignored(self) -> None:
        directory, record = self.capture("2026-11-04T04:22:16Z")
        self.assertIsNone(votewa_count_time(directory, {**record, "raw_sha256": "0" * 64}))

    def test_c3_count_after_retrieval_is_ignored(self) -> None:
        directory, record = self.capture("2026-11-06T00:00:00Z")
        self.assertIsNone(votewa_count_time(directory, record))


if __name__ == "__main__":
    unittest.main()
