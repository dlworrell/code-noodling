from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from election_reporting.analysis import analyze_election
from election_reporting.config import load_config
from election_reporting.ingest import read_snapshot
from election_reporting.model import ElectionReportError, SourceConfig
from election_reporting.watch import csvformat
from election_reporting.watch.config import load_watch_config
from election_reporting.watch.http import Response, check_url
from election_reporting.watch.run import carry_forward, fetch_snapshots
from election_reporting.watch.schedule import is_due, watch_window
from election_reporting.watch.sources import resolve_places
from election_reporting.watch.store import ElectionStore
from test_election_reporting import write_summary_xlsx

PACIFIC = ZoneInfo("America/Los_Angeles")
API = "https://results.votewa.gov/results/public/api"
STATE_ID = "01000000-f977-1a83-1ed4-08ddd69e9e69"
REPORT = f"https://results.votewa.gov/cdn/results/{STATE_ID}/All%20Results_abc.xlsx"
KC_YEAR = "https://kingcounty.gov/en/dept/elections/results/2026"
KC_PAGE = "https://kingcounty.gov/en/dept/elections/results/2026/november-general"
KC_FILE = (
    "https://cdn.kingcounty.gov/-/media/king-county/depts/elections/results/2026/11/"
    "webresults-20261103.csv"
)

WATCH_TOML = """
schema_version = 1
[watch]
state = "Washington"
counties = ["King", "Pierce County"]
election = "next"
"""


def text(value: str) -> list[dict[str, str]]:
    return [{"languageId": "en", "text": value}]


def jurisdiction() -> dict:
    return {
        "shortName": "washington",
        "childLocalities": [
            {"name": text("King County"), "shortName": "king-county-wa"},
            {"name": text("Pierce County"), "shortName": "pierce-county-wa"},
            {"name": text("Snohomish County"), "shortName": "snohomish-county-wa"},
        ],
        "elections": [
            {"publicElectionId": "20260804", "electionDate": "2026-08-04", "name": text("2026 Primary")},
            {"publicElectionId": "20261103", "electionDate": "2026-11-03", "name": text("2026 General")},
        ],
    }


def option(name: str, votes: int, party: str = "", write_in: bool = False) -> dict:
    return {
        "name": text(name),
        "voteCount": votes,
        "party": {"abbreviation": party or None, "name": None},
        "isWriteIn": write_in,
    }


def votewa_data(as_of: str, first: int = 60, second: int = 40) -> dict:
    return {
        "jurisdiction": {"clientTexts": ["large display text dropped when trimming"]},
        "election": {"asOf": as_of, "name": text("2026 General")},
        "precincts": [{"name": text("31-001")}],
        "voterTurnout": [{"precinctName": "31-001", "ballotsCast": 10.0}],
        "ballotItems": [
            {
                "id": "contest-a",
                "name": text("Pierce County Executive"),
                "voteTotal": first + second,
                "overVoteCount": 1,
                "underVoteCount": 4,
                "crossCounties": [],
                "summaryResults": {
                    "ballotOptions": [option("Alpha", first, "DEM"), option("Beta", second, "REP")]
                },
            },
            {
                "id": "contest-b",
                "name": text("Port of Tacoma Commissioner"),
                "voteTotal": 30,
                "crossCounties": ["Pierce County", "King County"],
                "summaryResults": {
                    "ballotOptions": [option("Gamma", 20), option("Write-in", 10, write_in=True)]
                },
            },
        ],
    }


KC_CSV = (
    '"ContestOrder","Contest","Contest ID","Choice Order","Choice","BallotsWith Contest","Votes"\r\n'
    '"1.00","King County Assessor","45","1","Alpha","1,000","600"\r\n'
    '"1.00","King County Assessor","45","2","Beta","1,000","390"\r\n'
)


class FakeWeb:
    def __init__(self, routes: dict[str, Response]) -> None:
        self.routes = routes
        self.requests: list[str] = []

    def __call__(self, url: str) -> Response:
        self.requests.append(url)
        check_url(url)
        if url not in self.routes:
            raise ElectionReportError(f"HTTP 404 from {url}")
        return self.routes[url]


def json_response(url: str, payload: dict) -> Response:
    return Response(url, json.dumps(payload).encode("utf-8"))


def web(directory: Path, pierce_as_of: str = "2026-11-04T04:15:00Z") -> FakeWeb:
    workbook = directory / "report.xlsx"
    write_summary_xlsx(workbook)
    routes = {
        f"{API}/jurisdictions/washington": json_response(
            f"{API}/jurisdictions/washington", jurisdiction()
        ),
        f"{API}/elections/washington/20261103": json_response(
            "x",
            {
                "jurisdictionId": STATE_ID,
                "publicReportCategories": [
                    {"reports": [{"reportName": "All Results Excel", "blobName": "All Results_abc.xlsx"}]}
                ],
            },
        ),
        REPORT: Response(
            REPORT,
            workbook.read_bytes(),
            {"last-modified": "Wed, 04 Nov 2026 04:20:00 GMT"},
        ),
        f"{API}/elections/pierce-county-wa/20261103/data": json_response(
            "x", votewa_data(pierce_as_of)
        ),
        KC_YEAR: Response(
            KC_YEAR,
            b'<a href="/en/dept/elections/results/2026/august-primary-election">Aug</a>'
            b'<a href="/en/dept/elections/results/2026/november-general">Nov</a>',
        ),
        KC_PAGE: Response(KC_PAGE, f'<a href="{KC_FILE}?rev=1&amp;hash=2">Nov 3</a>'.encode()),
        KC_FILE + "?rev=1&hash=2": Response(
            KC_FILE, KC_CSV.encode("utf-8"), {"last-modified": "Wed, 04 Nov 2026 04:30:00 GMT"}
        ),
    }
    return FakeWeb(routes)


def write_watch(directory: Path, extra: str = "") -> Path:
    path = directory / "watch.toml"
    path.write_text(WATCH_TOML + extra, encoding="utf-8")
    return path


ELECTION_NIGHT = datetime(2026, 11, 3, 21, 7, tzinfo=PACIFIC)


class WatchConfigTests(unittest.TestCase):
    def test_defaults_and_county_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_watch_config(write_watch(Path(directory)))
            self.assertEqual(config.state, "Washington")
            self.assertEqual(config.counties, ("King", "Pierce County"))
            self.assertEqual(config.election_night_every_minutes, 30)
            self.assertEqual(config.every_minutes, 60)
            self.assertEqual(config.elections_root, Path(directory) / "elections")

    def test_unsupported_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watch.toml"
            path.write_text('schema_version = 1\n[watch]\nstate = "Oregon"\n', encoding="utf-8")
            with self.assertRaisesRegex(ElectionReportError, "watch.state"):
                load_watch_config(path)

    def test_county_names_resolve_and_unknown_names_are_listed(self) -> None:
        places = resolve_places(jurisdiction(), "Washington", ("king", "Pierce County"), True)
        self.assertEqual([p.place_id for p in places], ["washington", "king-county", "pierce-county"])
        self.assertTrue(places[1].county_files)
        self.assertFalse(places[2].county_files)
        with self.assertRaisesRegex(ElectionReportError, "Kings.*Valid names: King County"):
            resolve_places(jurisdiction(), "Washington", ("Kings",), True)


class ScheduleTests(unittest.TestCase):
    def config(self):
        with tempfile.TemporaryDirectory() as directory:
            return load_watch_config(write_watch(Path(directory)))

    def test_general_election_window_runs_to_state_certification(self) -> None:
        window = watch_window(self.config(), date(2026, 11, 3), "general", has_state=True)
        self.assertEqual(window.start, datetime(2026, 11, 3, 20, 0, tzinfo=PACIFIC))
        self.assertEqual(window.election_night_end, datetime(2026, 11, 4, 2, 0, tzinfo=PACIFIC))
        self.assertEqual(window.end, datetime(2026, 12, 4, 0, 0, tzinfo=PACIFIC))

    def test_primary_window_uses_primary_certification(self) -> None:
        window = watch_window(self.config(), date(2026, 8, 4), "primary", has_state=True)
        self.assertEqual(window.end.date(), date(2026, 8, 22))

    def test_due_every_half_hour_on_election_night_then_hourly(self) -> None:
        config = self.config()
        window = watch_window(config, date(2026, 11, 3), "general", has_state=True)
        due = lambda h, m, d=3: is_due(config, window, datetime(2026, 11, d, h, m, tzinfo=PACIFIC))
        self.assertFalse(due(19, 37))  # before polls close
        self.assertTrue(due(20, 7))
        self.assertTrue(due(20, 37))
        self.assertTrue(due(1, 37, 4))  # still election night
        self.assertTrue(due(9, 7, 4))  # hourly slot
        self.assertFalse(due(9, 37, 4))  # off slot
        self.assertFalse(is_due(config, window, datetime(2026, 12, 4, 0, 7, tzinfo=PACIFIC)))


class CsvFormatTests(unittest.TestCase):
    def test_workbook_csv_matches_workbook_analysis_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / "2026-08-07-0118.xlsx"
            write_summary_xlsx(workbook, reported_ballots=90)
            converted = Path(directory) / "2026-08-07-0118.csv"
            converted.write_text(
                csvformat.from_summary_xlsx(workbook.read_bytes(), label=workbook.name, jurisdiction="Washington"),
                encoding="utf-8",
            )
            self.assertEqual(
                read_snapshot(workbook, "wa").contests, read_snapshot(converted, "wa").contests
            )

    def test_results_data_keeps_tallies_parties_and_county_spans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2026-11-03-2015.csv"
            path.write_text(
                csvformat.from_votewa_data(votewa_data("x"), jurisdiction="Pierce County", county="Pierce County"),
                encoding="utf-8",
            )
            snapshot = read_snapshot(path, "pierce")
            executive = snapshot.contests["pierce county executive"]
            self.assertEqual(executive.ballots, 105)  # 100 votes + 1 over + 4 under
            self.assertEqual((executive.overvotes, executive.undervotes), (1, 4))
            port = snapshot.contests["port of tacoma commissioner"]
            self.assertEqual(port.counties, ("Pierce County", "King County"))
            self.assertIsNone(port.overvotes)
            self.assertIn("DEM", path.read_text(encoding="utf-8"))

    def test_repeated_contest_names_stay_distinct(self) -> None:
        data = votewa_data("x")
        data["ballotItems"][1]["name"] = text("Pierce County Executive")
        rendered = csvformat.from_votewa_data(data, jurisdiction="Pierce County")
        self.assertIn("Pierce County Executive [contest-a]", rendered)
        self.assertIn("Pierce County Executive [contest-b]", rendered)

    def test_trimmed_data_drops_display_text_and_precinct_list(self) -> None:
        trimmed = json.loads(csvformat.trim_votewa_data(votewa_data("x")))
        self.assertEqual(sorted(trimmed), ["ballotItems", "election", "voterTurnout"])


class HttpTests(unittest.TestCase):
    def test_only_official_https_hosts_are_fetched(self) -> None:
        check_url("https://results.votewa.gov/results/public/api/jurisdictions/washington")
        for url in (
            "http://results.votewa.gov/x",
            "https://example.com/x",
            "https://results.votewa.gov.example.com/x",
        ):
            with self.assertRaises(ElectionReportError):
                check_url(url)


class FetchTests(unittest.TestCase):
    def test_fetch_writes_inputs_raw_files_provenance_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fake = web(root)
            result = fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            self.assertTrue(result.due)
            self.assertEqual(result.errors, [])
            election = root / "elections" / "2026-11-03"
            self.assertEqual(
                sorted(str(p.relative_to(election)) for p in election.rglob("*") if p.is_file()),
                [
                    "SOURCES.json",
                    "election.toml",
                    "input/king-county/2026-11-03-2030.csv",
                    "input/pierce-county/2026-11-03-2015.csv",
                    "input/washington/2026-11-03-2020.csv",
                    "raw/pierce-county/2026-11-03-2015.json.gz",
                    "raw/washington/2026-11-03-2020.xlsx",
                ],
            )
            self.assertEqual(
                (election / "input/king-county/2026-11-03-2030.csv").read_bytes(),
                KC_CSV.encode("utf-8"),
            )
            store = ElectionStore(election)
            kinds = {r.place: r.kind for r in store.records}
            self.assertEqual(
                kinds,
                {"washington": "state-xlsx", "king-county": "county-csv", "pierce-county": "votewa-json"},
            )
            analysis = analyze_election(load_config(election / "election.toml"))
            port = next(r for r in analysis.races if r.contest.startswith("Port of Tacoma"))
            self.assertFalse(port.controlling)
            self.assertIn("spans Pierce County, King County", port.scope_note)

    def test_unchanged_sources_add_nothing_and_changes_add_a_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
            again = fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual(again.added, [])
            fake = web(root, pierce_as_of="2026-11-04T05:45:00Z")
            fake.routes[f"{API}/elections/pierce-county-wa/20261103/data"] = json_response(
                "x", votewa_data("2026-11-04T05:45:00Z", first=70)
            )
            later = fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual([(r.place, r.snapshot) for r in later.added], [("pierce-county", "2026-11-03-2145")])

    def test_not_due_fetches_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fake = web(root)
            early = datetime(2026, 11, 3, 12, 0, tzinfo=PACIFIC)
            result = fetch_snapshots(config, fake, early, log=lambda _: None)
            self.assertFalse(result.due)
            self.assertEqual(fake.requests, [f"{API}/jurisdictions/washington"])
            self.assertFalse((root / "elections").exists())

    def test_zero_vote_results_are_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fake = web(root)
            data = votewa_data("2026-11-04T04:00:00Z", first=0, second=0)
            for opt in data["ballotItems"][1]["summaryResults"]["ballotOptions"]:
                opt["voteCount"] = 0
            fake.routes[f"{API}/elections/pierce-county-wa/20261103/data"] = json_response("x", data)
            del fake.routes[KC_PAGE]
            fake.routes[KC_YEAR] = Response(KC_YEAR, b"no election pages yet")
            fake.routes[REPORT] = Response(REPORT, b"", {})
            del fake.routes[f"{API}/elections/washington/20261103"]
            fake.routes[f"{API}/elections/washington/20261103"] = json_response("x", {})
            fake.routes[f"{API}/elections/washington/20261103/data"] = json_response("x", data)
            result = fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual(result.added, [])
            self.assertIn("pierce-county: no votes reported yet", result.skipped)
            self.assertIn("washington: no votes reported yet", result.skipped)
            self.assertIn("king-county: no county results page yet", result.skipped)
            election = root / "elections" / "2026-11-03"
            self.assertTrue((election / "election.toml").exists())
            self.assertFalse((election / "SOURCES.json").exists())

    def test_one_failing_source_does_not_stop_the_others(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fake = web(root)
            del fake.routes[f"{API}/elections/pierce-county-wa/20261103/data"]
            result = fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual(len(result.added), 2)
            self.assertTrue(result.errors[0].startswith("pierce-county: HTTP 404"))


class CarryForwardTests(unittest.TestCase):
    def test_carries_only_missing_snapshots_and_keeps_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = root / "pending"
            main = root / "main"
            for target in (pending, main):
                target.mkdir()
                config = load_watch_config(write_watch(target))
                if target is pending:
                    fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
            (main / "elections" / "2026-11-03").mkdir(parents=True)
            edited = main / "elections" / "2026-11-03" / "election.toml"
            edited.write_text("# hand edited\n", encoding="utf-8")
            copied = carry_forward(pending / "elections", main / "elections", log=lambda _: None)
            self.assertEqual(copied, 3)
            self.assertEqual(edited.read_text(encoding="utf-8"), "# hand edited\n")
            self.assertEqual(len(ElectionStore(main / "elections" / "2026-11-03").records), 3)
            self.assertEqual(carry_forward(pending / "elections", main / "elections", log=lambda _: None), 0)

    def test_tampered_snapshot_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = root / "pending"
            pending.mkdir()
            config = load_watch_config(write_watch(pending))
            fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
            next((pending / "elections").rglob("*.csv")).write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ElectionReportError, "does not match its recorded SHA-256"):
                carry_forward(pending / "elections", root / "main", log=lambda _: None)


class RepositoryTests(unittest.TestCase):
    def test_snapshot_files_are_not_git_ignored(self) -> None:
        import subprocess

        repo = Path(__file__).resolve().parents[1]
        if not (repo / ".git").exists():
            self.skipTest("not a git checkout")
        for path in (
            "election-data/elections/2026-11-03/SOURCES.json",
            "election-data/elections/2026-11-03/input/washington/2026-11-03-2015.csv",
            "election-data/elections/2026-11-03/raw/washington/2026-11-03-2015.xlsx",
            "election-data/elections/2026-11-03/raw/pierce-county/2026-11-03-2015.json.gz",
            "reports/elections/2026-11-03/analysis.json",
        ):
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", "--no-index", path], cwd=repo
            ).returncode == 0
            self.assertFalse(ignored, f"{path} is ignored by .gitignore")


class ScopeTests(unittest.TestCase):
    def test_single_county_contest_from_a_county_source_stays_controlling(self) -> None:
        from election_reporting.analysis import _scope_status
        from election_reporting.model import ContestResult

        source = SourceConfig("pierce", Path("."), "Pierce County", "county", 10)
        contest = ContestResult("1", "Pierce County Executive", 1.0, 10, 10, None, None, (), "k", ("Pierce County",))
        self.assertTrue(_scope_status(source, contest)[0])


if __name__ == "__main__":
    unittest.main()
