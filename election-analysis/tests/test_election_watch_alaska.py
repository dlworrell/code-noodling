# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:0532be65-a5dc-4449-81b2-cd721b45cc0c

"""Alaska source adapter, borough conversion and Alaska contest rules (offline)."""

from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from election_reporting.analysis import analyze_election
from election_reporting.config import load_config
from election_reporting.model import ElectionReportError
from election_reporting.watch import alaska, certs
from election_reporting.watch.config import load_watch_settings
from election_reporting.watch.http import Response, check_url
from election_reporting.watch.run import fetch_snapshots
from election_reporting.watch.schedule import watch_window

from test_election_watch import FakeWeb

ELECTION = Path(__file__).resolve().parents[1]  # election-analysis/
ANCHORAGE = ZoneInfo("America/Anchorage")
PAGE = "https://www.elections.alaska.gov/election-results/e/?id=26genr"
ENR = "https://www.elections.alaska.gov/enr26/results/GA_ENR_Precinct_State_of_Alaska.csv"
ELECTION_NIGHT = datetime(2026, 11, 3, 22, 7, tzinfo=ANCHORAGE)
COLUMNS = (
    "Precinct_name", "Reporting_flag", "Pct_Id", "Contest_Id", "Contest_seq_nbr",
    "Contest_title", "candidate_name", "cand_seq_nbr", "Party_Code", "total_ballots",
    "total_votes", "total_under_votes", "total_over_votes",
)


def enr(rows: list[tuple]) -> bytes:
    """Precinct results CSV; each row is (precinct, id, contest, title, [(name, votes)], under, over)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(COLUMNS)
    for precinct, precinct_id, contest_id, title, choices, under, over in rows:
        ballots = sum(votes for _, votes in choices) + under + over
        for order, (name, votes) in enumerate(choices, start=1):
            writer.writerow(
                (precinct, "0", precinct_id, contest_id, contest_id, title, name, order,
                 "REP" if order == 1 else "DEM", ballots, votes, under, over)
            )
    return b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8")


GOVERNOR = [("Begich/Hnilicka", 0), ("Kreiss-Tomkins/Johnson", 0), ("Bronson/Church", 0)]


def governor(*votes: int) -> list[tuple[str, int]]:
    return [(name, count) for (name, _), count in zip(GOVERNOR, votes)]


SAMPLE = enr(
    [
        ("01-600 Ketchikan No. 1", "01-600", "5", "Governor / Lieutenant Governor", governor(40, 30, 10), 2, 1),
        # South Tongass is 96.8% Ketchikan Gateway: counted there.
        ("01-670 South Tongass", "01-670", "5", "Governor / Lieutenant Governor", governor(20, 10, 5), 1, 0),
        ("05-865 Seward/Lowell Point", "05-865", "5", "Governor / Lieutenant Governor", governor(15, 25, 5), 0, 0),
        ("District 1 - Absentee", "01", "5", "Governor / Lieutenant Governor", governor(100, 90, 9), 3, 0),
        ("HD99 Fed Overseas Absentee", "99-999", "5", "Governor / Lieutenant Governor", governor(1, 2, 0), 0, 0),
        ("01-600 Ketchikan No. 1", "01-600", "68", "BM#1 - 23RCF2", [("YES", 50), ("NO", 30)], 3, 0),
        ("05-865 Seward/Lowell Point", "05-865", "68", "BM#1 - 23RCF2", [("YES", 20), ("NO", 25)], 0, 0),
    ]
)


def data_root(directory: Path) -> Path:
    root = directory / "election-data"
    shutil.copytree(ELECTION / "data" / "crosswalks", root / "crosswalks")
    return root


def write_watch(root: Path, counties: str = '["Ketchikan Gateway", "Kenai Peninsula"]') -> Path:
    path = root / "watch.toml"
    path.write_text(
        f'schema_version = 1\n[[watch]]\nstate = "Alaska"\ncounties = {counties}\nelection = "next"\n',
        encoding="utf-8",
    )
    return path


def read_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def by_choice(rows: list[dict[str, str]], contest: str) -> dict[str, int]:
    return {row["Choice"]: int(row["Votes"]) for row in rows if row["Contest"] == contest}


class AlaskaPlacesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.crosswalk = alaska.load_crosswalk(ELECTION / "data")

    def test_election_days_follow_statute(self) -> None:
        days = {ref.public_id: ref.day for ref in alaska.list_elections(date(2026, 10, 23))}
        self.assertEqual(days["26prim"], date(2026, 8, 18))
        self.assertEqual(days["26genr"], date(2026, 11, 3))
        self.assertEqual(days["24prim"], date(2024, 8, 20))
        self.assertEqual(days["24genr"], date(2024, 11, 5))
        self.assertEqual(days["28genr"], date(2028, 11, 7))

    def test_borough_names_resolve_without_their_suffix(self) -> None:
        places = alaska.resolve_places(self.crosswalk, ("Bethel", "anchorage", "Kenai Peninsula Borough"), False)
        self.assertEqual(
            [p.place_id for p in places],
            ["alaska", "bethel-census-area", "anchorage-municipality", "kenai-peninsula-borough"],
        )
        self.assertEqual([p.scope for p in places], ["state", "county", "county", "county"])

    def test_all_and_unknown_names(self) -> None:
        self.assertEqual(len(alaska.resolve_places(self.crosswalk, (), True)), 31)
        with self.assertRaisesRegex(ElectionReportError, "Valid names: .*Yukon-Koyukuk Census Area"):
            alaska.resolve_places(self.crosswalk, ("Anchorge",), False)

    def test_split_precinct_goes_to_its_main_borough(self) -> None:
        self.assertEqual(self.crosswalk.precincts["01-670"], "Ketchikan Gateway Borough")
        self.assertEqual(len(self.crosswalk.precincts), 402)


class AlaskaConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        crosswalk = alaska.load_crosswalk(ELECTION / "data")
        places = alaska.resolve_places(crosswalk, ("Ketchikan Gateway", "Kenai Peninsula", "Bethel"), False)
        self.outputs = alaska.convert(SAMPLE, crosswalk, places)

    def test_statewide_includes_district_level_and_overseas_ballots(self) -> None:
        rows = read_rows(self.outputs["alaska"])
        self.assertEqual(
            by_choice(rows, "Governor / Lieutenant Governor"),
            {"Begich/Hnilicka": 176, "Kreiss-Tomkins/Johnson": 157, "Bronson/Church": 29},
        )
        first = next(r for r in rows if r["Contest"] == "Governor / Lieutenant Governor")
        # Ballots, undervotes and overvotes count once per precinct, not per candidate row.
        self.assertEqual(first["BallotsWith Contest"], str(176 + 157 + 29 + 6 + 1))
        self.assertEqual((first["Contest Undervotes"], first["Contest Overvotes"]), ("6", "1"))
        self.assertEqual(first["Jurisdiction"], "Alaska")
        self.assertEqual(first["Counties"], "")

    def test_boroughs_hold_their_election_day_precincts_only(self) -> None:
        ketchikan = read_rows(self.outputs["ketchikan-gateway-borough"])
        self.assertEqual(
            by_choice(ketchikan, "Governor / Lieutenant Governor"),
            {"Begich/Hnilicka": 60, "Kreiss-Tomkins/Johnson": 40, "Bronson/Church": 15},
        )
        row = ketchikan[0]
        self.assertEqual(row["County"], "Ketchikan Gateway Borough")
        self.assertEqual(row["Counties"], "Kenai Peninsula; Ketchikan Gateway")
        kenai = read_rows(self.outputs["kenai-peninsula-borough"])
        self.assertEqual(by_choice(kenai, "BM#1 - 23RCF2"), {"YES": 20, "NO": 25})
        # No Bethel precincts in the file: no output rather than an empty file.
        self.assertNotIn("bethel-census-area", self.outputs)

    def test_csv_is_readable_by_the_analyzer(self) -> None:
        from election_reporting.ingest import read_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2026-11-03-2207.csv"
            path.write_text(self.outputs["ketchikan-gateway-borough"], encoding="utf-8")
            snapshot = read_snapshot(path, "ketchikan")
        governor_contest = next(c for c in snapshot.contests.values() if c.contest_id == "5")
        self.assertEqual(governor_contest.counties, ("Kenai Peninsula", "Ketchikan Gateway"))

    def test_unknown_precinct_in_a_house_district_is_an_error(self) -> None:
        crosswalk = alaska.load_crosswalk(ELECTION / "data")
        places = alaska.resolve_places(crosswalk, (), False)
        bad = enr([("01-999 New Precinct", "01-999", "5", "Governor", governor(1, 1, 1), 0, 0)])
        with self.assertRaisesRegex(ElectionReportError, "01-999"):
            alaska.convert(bad, crosswalk, places)

    def test_missing_columns_are_reported(self) -> None:
        crosswalk = alaska.load_crosswalk(ELECTION / "data")
        with self.assertRaisesRegex(ElectionReportError, "lack columns: .*Pct_Id"):
            alaska.convert(b"Precinct_name,Contest_title\nx,y\n", crosswalk, alaska.resolve_places(crosswalk, (), False))


def alaska_web(last_modified: str = "Wed, 04 Nov 2026 06:50:00 GMT", body: bytes = SAMPLE) -> FakeWeb:
    return FakeWeb(
        {
            PAGE: Response(PAGE, f'<a href="{ENR}">Precinct results (CSV)</a>'.encode()),
            ENR: Response(ENR, body, {"last-modified": last_modified}),
        }
    )


class AlaskaFetchTests(unittest.TestCase):
    def test_fetch_writes_statewide_and_borough_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = data_root(Path(tmp))
            config = load_watch_settings(write_watch(root)).watches[0]
            result = fetch_snapshots(config, alaska_web(), ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual(result.errors, [])
            directory = root / "elections" / "alaska" / "2026-11-03"
            self.assertEqual(
                sorted(r.place for r in result.added),
                ["alaska", "kenai-peninsula-borough", "ketchikan-gateway-borough"],
            )
            stamp = "2026-11-03-2150"  # Last-Modified 06:50 UTC is 21:50 AKST
            raw = directory / "raw" / "alaska" / f"{stamp}.csv.gz"
            self.assertEqual(gzip.decompress(raw.read_bytes()), SAMPLE)
            self.assertFalse((directory / "raw" / "ketchikan-gateway-borough").exists())
            records = {r["place"]: r for r in json.loads((directory / "SOURCES.json").read_text())["snapshots"]}
            self.assertEqual(records["alaska"]["source_url"], ENR)
            self.assertIn("Election-day precinct votes only", records["kenai-peninsula-borough"]["note"])

            election = (directory / "election.toml").read_text()
            self.assertIn('default_rule = "rcv"', election)
            self.assertIn('jurisdiction = "Kenai Peninsula Borough, Alaska"', election)
            self.assertIn("certification_date = 2026-11-25", election)
            analysis = analyze_election(load_config(directory / "election.toml"))
            rules = {race.contest: race.rule for race in analysis.races}
            self.assertEqual(rules, {"Governor / Lieutenant Governor": "rcv", "BM#1 - 23RCF2": "winner"})

            again = fetch_snapshots(config, alaska_web(), ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual(again.added, [])

    def test_file_from_an_earlier_election_is_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = data_root(Path(tmp))
            config = load_watch_settings(write_watch(root)).watches[0]
            # The primary's file, still at the shared path before new results post.
            web = alaska_web(last_modified="Mon, 31 Aug 2026 21:07:47 GMT")
            result = fetch_snapshots(config, web, ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual(result.added, [])
            self.assertIn("before polls closed", result.skipped[0])

    def test_no_results_link_yet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = data_root(Path(tmp))
            config = load_watch_settings(write_watch(root)).watches[0]
            web = FakeWeb({PAGE: Response(PAGE, b"<p>Results will be posted after 9 p.m.</p>")})
            result = fetch_snapshots(config, web, ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual(result.skipped, ["alaska: no precinct results file posted yet"])

    def test_window_uses_alaska_time_and_certification_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_watch_settings(write_watch(data_root(Path(tmp)), counties='"all"')).watches[0]
            self.assertTrue(config.all_counties)
            general = watch_window(config, date(2026, 11, 3), "general", has_state=True)
            self.assertEqual(general.start, datetime(2026, 11, 3, 20, 0, tzinfo=ANCHORAGE))
            self.assertEqual(general.end, datetime(2026, 11, 26, 0, 0, tzinfo=ANCHORAGE))
            primary = watch_window(config, date(2026, 8, 18), "primary", has_state=True)
            self.assertEqual(primary.end, datetime(2026, 9, 1, 0, 0, tzinfo=ANCHORAGE))


class AlaskaHttpTests(unittest.TestCase):
    def test_alaska_host_is_allowed_and_has_its_pinned_intermediate(self) -> None:
        check_url(ENR)
        self.assertEqual(
            certs.der_sha256(certs.GLOBALSIGN_GCC_R46_OV_TLS_CA_2025),
            certs.GLOBALSIGN_GCC_R46_OV_TLS_CA_2025_SHA256,
        )
        context = certs.context_for("www.elections.alaska.gov")
        self.assertTrue(context.check_hostname)
        self.assertIsNone(certs.context_for("results.votewa.gov"))


def contest_csv(choices: list[tuple[str, int]]) -> str:
    lines = ["ContestOrder,Contest,Contest ID,Choice Order,Choice,BallotsWith Contest,Votes"]
    ballots = sum(votes for _, votes in choices)
    for order, (name, votes) in enumerate(choices, start=1):
        lines.append(f"1,U.S. Senator,6,{order},{name},{ballots},{votes}")
    return "\n".join(lines) + "\n"


class AlaskaRuleTests(unittest.TestCase):
    def analyze(self, rule: str, choices: list[tuple[str, int]]):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "input" / "alaska").mkdir(parents=True)
            (directory / "input" / "alaska" / "2026-11-03-2207.csv").write_text(contest_csv(choices))
            (directory / "election.toml").write_text(
                'schema_version = 1\n[election]\ntitle = "t"\n[analysis]\n'
                f'default_rule = "{rule}"\n[[sources]]\nid = "alaska"\npath = "input/alaska"\n'
                'jurisdiction = "Alaska State"\nscope = "state"\npriority = 100\n'
            )
            return analyze_election(load_config(directory / "election.toml")).races[0]

    def test_top_four_compares_fourth_and_fifth(self) -> None:
        race = self.analyze("top_four", [("A", 50), ("B", 40), ("C", 30), ("D", 20), ("E", 15), ("F", 1)])
        self.assertEqual(race.rule, "top_four")
        [decision] = race.decisions
        self.assertEqual((decision.kind, decision.current_side, decision.change_side, decision.margin),
                         ("top_four_cutoff", "D", "E", 5))

    def test_rcv_without_a_first_choice_majority_is_flagged(self) -> None:
        race = self.analyze("rcv", [("A", 48), ("B", 45), ("C", 7)])
        [decision] = race.decisions
        self.assertEqual(decision.kind, "majority_status")
        self.assertIn("no candidate is above 50%", decision.current_state)
        self.assertIn("first-choice leader is not necessarily the winner", race.scope_note)

    def test_rcv_with_a_first_choice_majority_is_not_flagged(self) -> None:
        race = self.analyze("rcv", [("A", 55), ("B", 45)])
        self.assertIn("above 50%", race.decisions[0].current_state)
        self.assertNotIn("not necessarily the winner", race.scope_note)

    def test_unknown_default_rule_is_rejected(self) -> None:
        with self.assertRaisesRegex(ElectionReportError, "analysis.default_rule must be one of"):
            self.analyze("instant_runoff", [("A", 1), ("B", 2)])


if __name__ == "__main__":
    unittest.main()
