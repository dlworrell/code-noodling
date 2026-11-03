# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:fa4beb2e-7430-49df-8033-d42cf382cd8d

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from election_reporting.analysis import analyze_election
from election_reporting.config import load_config
from election_reporting.ingest import read_snapshot
from election_reporting.model import ElectionReportError, SourceConfig
from election_reporting.watch import csvformat
from election_reporting.watch.config import load_watch_settings
from election_reporting.watch.http import LARGE_MAX_BYTES, Response, check_url
from election_reporting.watch.run import carry_forward, fetch_snapshots
from election_reporting.watch.schedule import is_due, watch_window
from election_reporting.watch.sources import resolve_places
from election_reporting.watch.store import ElectionStore
from test_election_areas import full_workbook
from test_election_reporting import write_summary_xlsx

PACIFIC = ZoneInfo("America/Los_Angeles")
API = "https://results.votewa.gov/results/public/api"
STATE_ID = "01000000-f977-1a83-1ed4-08ddd69e9e69"
REPORT = f"https://results.votewa.gov/cdn/results/{STATE_ID}/All%20Results_abc.xlsx"
PRECINCT_REPORT = f"https://results.votewa.gov/cdn/results/{STATE_ID}/Results_abc.xlsx"
KC_YEAR = "https://kingcounty.gov/en/dept/elections/results/2026"
KC_PAGE = "https://kingcounty.gov/en/dept/elections/results/2026/november-general"
KC_FILE = (
    "https://cdn.kingcounty.gov/-/media/king-county/depts/elections/results/2026/11/"
    "webresults-20261103.csv"
)

WATCH_TOML = """
schema_version = 1
[[watch]]
state = "Washington"
counties = ["King", "Pierce County"]
election = "next"
"""


def load_watch_config(path: Path):
    return load_watch_settings(path).watches[0]


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
        self.caps: dict[str, int | None] = {}

    def __call__(self, url: str, max_bytes: int | None = None) -> Response:
        self.requests.append(url)
        self.caps[url] = max_bytes
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
            self.assertEqual(config.elections_root, Path(directory) / "elections" / "washington")
            self.assertEqual(config.timezone.key, "America/Los_Angeles")

    def test_unsupported_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watch.toml"
            path.write_text('schema_version = 1\n[[watch]]\nstate = "Oregon"\n', encoding="utf-8")
            with self.assertRaisesRegex(ElectionReportError, r"watch\[1\]\.state"):
                load_watch_config(path)

    def test_schedule_is_per_state_and_states_are_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_watch(Path(directory), '[watch.schedule]\nevery_minutes = 120\n')
            self.assertEqual(load_watch_config(path).every_minutes, 120)
            path.write_text(WATCH_TOML + WATCH_TOML.replace("schema_version = 1", ""), encoding="utf-8")
            with self.assertRaisesRegex(ElectionReportError, "repeated: washington"):
                load_watch_settings(path)

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
            election = root / "elections" / "washington" / "2026-11-03"
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

    # WATCH-02 acceptance cases (King County exports replaced in place):
    # K1 a corrected body B at the same URL is kept as a new immutable snapshot;
    # K2 a correction that changes only the query string is kept the same way;
    # K3 serving the same body A again adds nothing (idempotent);
    # K4 the earlier snapshot's bytes are untouched by the correction.

    def king_snapshots(self, root: Path):
        election = root / "elections" / "washington" / "2026-11-03"
        return sorted((r.snapshot, r.input_sha256) for r in ElectionStore(election).records
                      if r.place == "king-county")

    def corrected(self, fake, url_key: str, page_href: str, body: str, modified: str):
        fake.routes[KC_PAGE] = Response(KC_PAGE, f'<a href="{page_href}">Nov 3</a>'.encode())
        fake.routes[url_key] = Response(KC_FILE, body.encode("utf-8"), {"last-modified": modified})
        return fake

    def test_k1_corrected_body_at_the_same_url_is_a_new_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
            fake = self.corrected(web(root), KC_FILE + "?rev=1&hash=2", f"{KC_FILE}?rev=1&amp;hash=2",
                                  KC_CSV + "\n", "Wed, 04 Nov 2026 05:10:00 GMT")
            later = fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual([r.place for r in later.added], ["king-county"])
            self.assertEqual(len(self.king_snapshots(root)), 2)

    def test_k2_query_only_revision_is_a_new_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
            fake = self.corrected(web(root), KC_FILE + "?rev=2&hash=3", f"{KC_FILE}?rev=2&amp;hash=3",
                                  KC_CSV + "\n", "Wed, 04 Nov 2026 05:10:00 GMT")
            later = fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual([r.place for r in later.added], ["king-county"])

    def test_k3_same_body_again_adds_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
            before = self.king_snapshots(root)
            again = fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual([r for r in again.added if r.place == "king-county"], [])
            self.assertEqual(self.king_snapshots(root), before)
            self.assertIn("king-county: unchanged", again.skipped)

    def test_k4_earlier_snapshot_bytes_are_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
            election = root / "elections" / "washington" / "2026-11-03"
            first = (election / "input/king-county/2026-11-03-2030.csv").read_bytes()
            fake = self.corrected(web(root), KC_FILE + "?rev=1&hash=2", f"{KC_FILE}?rev=1&amp;hash=2",
                                  KC_CSV + "\n", "Wed, 04 Nov 2026 05:10:00 GMT")
            fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual((election / "input/king-county/2026-11-03-2030.csv").read_bytes(), first)

    # K5-K6 (Em's review of df4c59e): controls.

    def test_k5_substantive_count_change_is_kept_with_new_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
            changed = KC_CSV.replace('"Alpha","1,000","600"', '"Alpha","1,000","610"')
            self.assertNotEqual(changed, KC_CSV)
            fake = self.corrected(web(root), KC_FILE + "?rev=1&hash=2", f"{KC_FILE}?rev=1&amp;hash=2",
                                  changed, "Wed, 04 Nov 2026 05:10:00 GMT")
            later = fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            added = [r for r in later.added if r.place == "king-county"]
            self.assertEqual(len(added), 1)
            election = root / "elections" / "washington" / "2026-11-03"
            self.assertIn('"Alpha","1,000","610"', (election / added[0].input).read_text(encoding="utf-8"))

    def test_k6_same_body_under_a_new_query_adds_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
            before = self.king_snapshots(root)
            fake = self.corrected(web(root), KC_FILE + "?rev=9&hash=9", f"{KC_FILE}?rev=9&amp;hash=9",
                                  KC_CSV, "Wed, 04 Nov 2026 05:10:00 GMT")
            later = fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual([r for r in later.added if r.place == "king-county"], [])
            self.assertEqual(self.king_snapshots(root), before)
            self.assertIn("king-county: unchanged", later.skipped)

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
            election = root / "elections" / "washington" / "2026-11-03"
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


class AreaFetchTests(unittest.TestCase):
    def with_precinct_report(self, root: Path) -> FakeWeb:
        fake = web(root)
        metadata = json.loads(fake.routes[f"{API}/elections/washington/20261103"].body)
        metadata["publicReportCategories"][0]["reports"].append(
            {"reportName": "Summary and Precinct Results Excel", "blobName": "Results_abc.xlsx"}
        )
        fake.routes[f"{API}/elections/washington/20261103"] = json_response("x", metadata)
        fake.routes[PRECINCT_REPORT] = Response(
            PRECINCT_REPORT, full_workbook(), {"last-modified": "Wed, 04 Nov 2026 04:25:00 GMT"}
        )
        return fake

    def test_area_files_records_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fake = self.with_precinct_report(root)
            result = fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual(result.errors, [])
            self.assertEqual(fake.caps[PRECINCT_REPORT], LARGE_MAX_BYTES)
            election = root / "elections" / "washington" / "2026-11-03"
            self.assertTrue((election / "areas/washington/county/2026-11-03-2025.csv").exists())
            self.assertTrue((election / "areas/washington/precinct/2026-11-03-2025.csv.gz").exists())
            self.assertFalse(any(p.suffix == ".xlsx" and "Results_abc" in p.name for p in election.rglob("*")))
            [record] = [r for r in ElectionStore(election).records if r.place == "washington-areas"]
            self.assertEqual(record.kind, "state-area-xlsx")
            self.assertIn("3 precincts", record.note)
            toml = (election / "election.toml").read_text(encoding="utf-8")
            self.assertIn('level = "precinct"', toml)
            self.assertIn('default_rule = "winner"', toml)  # a general election
            analysis = analyze_election(load_config(election / "election.toml"))
            self.assertEqual(analysis.areas.reconciliation["precinct_vs_county"]["mismatches"], [])
            again = fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            self.assertIn("washington-areas: unchanged", again.skipped)
            # Carried forward and verified like any other snapshot.
            main = root / "main"
            self.assertEqual(carry_forward(root / "elections", main, log=lambda _: None), 4)
            self.assertTrue((main / "washington/2026-11-03/areas/washington/precinct/2026-11-03-2025.csv.gz").exists())

    def test_a_failing_area_fetch_does_not_stop_the_statewide_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_watch_config(write_watch(root))
            fake = self.with_precinct_report(root)
            fake.routes[PRECINCT_REPORT] = Response(PRECINCT_REPORT, b"not a workbook")
            result = fetch_snapshots(config, fake, ELECTION_NIGHT, log=lambda _: None)
            self.assertEqual(len(result.added), 3)
            self.assertEqual(len(result.errors), 1)
            self.assertTrue(result.errors[0].startswith("washington-areas: "))


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
            (main / "elections" / "washington" / "2026-11-03").mkdir(parents=True)
            edited = main / "elections" / "washington" / "2026-11-03" / "election.toml"
            edited.write_text("# hand edited\n", encoding="utf-8")
            copied = carry_forward(pending / "elections", main / "elections", log=lambda _: None)
            self.assertEqual(copied, 3)
            self.assertEqual(edited.read_text(encoding="utf-8"), "# hand edited\n")
            self.assertEqual(len(ElectionStore(main / "elections" / "washington" / "2026-11-03").records), 3)
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


    # WATCH-01 acceptance cases (paths in the pending branch's SOURCES.json are data):
    # P1 a parent (..) path is refused before any read outside the pending tree;
    # P2 an absolute path is refused;
    # P3 a symlink inside the pending tree is refused before it is followed;
    # P4 a symlink inside the target tree is refused before any write through it;
    # P5 one bad path in an election directory leaves no partial copy of any of its records;
    # P6 normal relative paths still carry (test_carries_only_missing_snapshots...).

    def pending_tree(self, root: Path) -> Path:
        pending = root / "pending"
        pending.mkdir()
        config = load_watch_config(write_watch(pending))
        fetch_snapshots(config, web(root), ELECTION_NIGHT, log=lambda _: None)
        return pending / "elections" / "washington" / "2026-11-03"

    def rewrite(self, election: Path, place: str, **fields) -> None:
        sources = election / "SOURCES.json"
        data = json.loads(sources.read_text(encoding="utf-8"))
        for item in data["snapshots"]:
            if item["place"] == place:
                item.update(fields)
        sources.write_text(json.dumps(data), encoding="utf-8")

    def assert_refused(self, root: Path) -> Path:
        target = root / "main" / "elections"
        with self.assertRaisesRegex(ElectionReportError, "refusing stored path"):
            carry_forward(root / "pending" / "elections", target, log=lambda _: None)
        return target

    def test_p1_parent_path_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            election = self.pending_tree(root)
            secret = root / "outside.csv"
            secret.write_bytes(b"secret")
            self.rewrite(election, "king-county", input="../../../../outside.csv",
                         input_sha256=hashlib.sha256(b"secret").hexdigest())
            target = self.assert_refused(root)
            self.assertFalse(any(target.rglob("outside.csv")) if target.exists() else False)

    def test_p2_absolute_path_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            election = self.pending_tree(root)
            self.rewrite(election, "king-county", input=str(root / "x.csv"))
            self.assert_refused(root)

    def test_p3_symlink_in_pending_tree_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            election = self.pending_tree(root)
            outside = root / "elsewhere"
            outside.mkdir()
            record = next(r for r in ElectionStore(election).records if r.place == "king-county")
            real = election / record.input
            (outside / real.name).write_bytes(real.read_bytes())
            shutil.rmtree(real.parent)
            real.parent.symlink_to(outside, target_is_directory=True)
            self.assert_refused(root)

    def test_p4_symlink_in_target_tree_is_refused_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.pending_tree(root)
            outside = root / "elsewhere"
            outside.mkdir()
            target_election = root / "main" / "elections" / "washington" / "2026-11-03"
            target_election.mkdir(parents=True)
            (target_election / "input").symlink_to(outside, target_is_directory=True)
            self.assert_refused(root)
            self.assertEqual(list(outside.iterdir()), [])

    def test_p5_bad_raw_path_leaves_no_partial_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            election = self.pending_tree(root)
            self.rewrite(election, "pierce-county", raw="../../../escape.json.gz")
            target = self.assert_refused(root)
            # Nothing from this election directory is written, not even the valid records.
            election_target = target / "washington" / "2026-11-03"
            written = [p for p in election_target.rglob("*") if p.is_file()] if election_target.exists() else []
            self.assertEqual(written, [])


    # P7-P10 (Em's review of 6cf9e53): the whole chain, from root to election
    # directory to SOURCES.json and files, in both trees, is confined before any
    # read or write; and no file I/O ever touches a path outside the two roots.

    def monitored_carry(self, root: Path):
        """Run carry_forward while recording every path read or written; return (error, paths)."""
        seen: list[Path] = []
        originals = {name: getattr(Path, name) for name in ("read_bytes", "read_text", "write_bytes", "write_text")}

        def spy(name):
            def wrapper(self_path, *args, **kwargs):
                seen.append(Path(os.path.realpath(self_path)))
                return originals[name](self_path, *args, **kwargs)
            return wrapper

        error = None
        with mock.patch.multiple(Path, **{name: spy(name) for name in originals}):
            try:
                carry_forward(root / "pending" / "elections", root / "main" / "elections", log=lambda _: None)
            except ElectionReportError as exc:
                error = exc
        return error, seen

    def assert_no_outside_io(self, root: Path, seen) -> None:
        allowed = [Path(os.path.realpath(root / "pending")), Path(os.path.realpath(root / "main"))]
        outside = [p for p in seen if not any(p == a or a in p.parents for a in allowed)]
        self.assertEqual(outside, [])

    def outside_dir(self, root: Path) -> Path:
        outside = root / "elsewhere"
        outside.mkdir(exist_ok=True)
        return outside

    def test_p7_target_election_directory_symlink_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.pending_tree(root)
            outside = self.outside_dir(root)
            (root / "main" / "elections" / "washington").mkdir(parents=True)
            (root / "main" / "elections" / "washington" / "2026-11-03").symlink_to(outside, target_is_directory=True)
            error, seen = self.monitored_carry(root)
            self.assertIsNotNone(error)
            self.assertEqual(list(outside.iterdir()), [])
            self.assert_no_outside_io(root, seen)

    def test_p8_source_ledger_symlink_is_refused_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            election = self.pending_tree(root)
            outside = self.outside_dir(root)
            ledger = election / "SOURCES.json"
            (outside / "SOURCES.json").write_bytes(ledger.read_bytes())
            ledger.unlink()
            ledger.symlink_to(outside / "SOURCES.json")
            error, seen = self.monitored_carry(root)
            self.assertIsNotNone(error)
            self.assert_no_outside_io(root, seen)

    def test_p9_target_ledger_symlink_is_refused_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.pending_tree(root)
            outside = self.outside_dir(root)
            (outside / "victim.json").write_text("{}", encoding="utf-8")
            target_election = root / "main" / "elections" / "washington" / "2026-11-03"
            target_election.mkdir(parents=True)
            (target_election / "SOURCES.json").symlink_to(outside / "victim.json")
            error, seen = self.monitored_carry(root)
            self.assertIsNotNone(error)
            self.assertEqual((outside / "victim.json").read_text(encoding="utf-8"), "{}")
            self.assert_no_outside_io(root, seen)

    def test_p10_symlinked_directory_in_pending_tree_is_not_followed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            election = self.pending_tree(root)
            outside = self.outside_dir(root)
            # Move the real election directory outside and link it back in.
            shutil.move(str(election), str(outside / "2026-11-03"))
            election.symlink_to(outside / "2026-11-03", target_is_directory=True)
            error, seen = self.monitored_carry(root)
            self.assertIsNone(error)  # not followed, so nothing is found to carry
            self.assertFalse((root / "main" / "elections").exists()
                             and any((root / "main" / "elections").rglob("*.csv")))
            self.assert_no_outside_io(root, seen)

    def test_p11_valid_tree_carries_with_no_outside_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.pending_tree(root)
            error, seen = self.monitored_carry(root)
            self.assertIsNone(error)
            self.assertTrue(any((root / "main" / "elections").rglob("*.csv")))
            self.assert_no_outside_io(root, seen)


    # P12-P14 (Em's rereview of 2a7a42f): the segment from the pending checkout
    # (trusted, chosen by the workflow) down to its nested elections directory
    # comes from the automation branch and is checked before any listing or read.

    def checkout_layout(self, root: Path, symlinked: bool) -> tuple[Path, Path]:
        """A pending checkout with election-analysis/data/elections, optionally via a symlink."""
        self.pending_tree(root)  # real snapshots under root/pending/elections
        checkout = root / ".pending"
        data_parent = checkout / "election-analysis"
        data_parent.mkdir(parents=True)
        if symlinked:
            outside = self.outside_dir(root)
            shutil.move(str(root / "pending" / "elections"), str(outside / "elections"))
            (data_parent / "data").symlink_to(outside, target_is_directory=True)
        else:
            (data_parent / "data").mkdir()
            shutil.move(str(root / "pending" / "elections"), str(data_parent / "data" / "elections"))
        return checkout, data_parent / "data" / "elections"

    def carry_with_base(self, root: Path, checkout: Path, source: Path):
        seen: list[Path] = []
        originals = {name: getattr(Path, name) for name in ("read_bytes", "read_text", "write_bytes", "write_text")}

        def spy(name):
            def wrapper(self_path, *args, **kwargs):
                seen.append(Path(os.path.realpath(self_path)))
                return originals[name](self_path, *args, **kwargs)
            return wrapper

        error = None
        with mock.patch.multiple(Path, **{name: spy(name) for name in originals}):
            try:
                carry_forward(source, root / "main" / "elections", log=lambda _: None, source_base=checkout)
            except ElectionReportError as exc:
                error = exc
        return error, seen

    def test_p12_symlink_between_checkout_and_elections_root_is_refused_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, source = self.checkout_layout(root, symlinked=True)
            error, seen = self.carry_with_base(root, checkout, source)
            self.assertIsNotNone(error)
            self.assertIn("symlink", str(error))
            self.assertEqual(seen, [])  # nothing read or written at all
            self.assertFalse((root / "main").exists())

    def test_p13_real_nested_elections_root_carries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, source = self.checkout_layout(root, symlinked=False)
            error, seen = self.carry_with_base(root, checkout, source)
            self.assertIsNone(error)
            self.assertTrue(any((root / "main" / "elections").rglob("*.csv")))
            allowed = [Path(os.path.realpath(checkout)), Path(os.path.realpath(root / "main"))]
            self.assertEqual([p for p in seen if not any(a == p or a in p.parents for a in allowed)], [])

    def test_p14_source_outside_the_trusted_root_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.pending_tree(root)
            (root / ".pending").mkdir()
            with self.assertRaisesRegex(ElectionReportError, "not inside"):
                carry_forward(root / "pending" / "elections", root / "main" / "elections",
                              log=lambda _: None, source_base=root / ".pending")

    def test_p15_workflow_passes_the_trusted_root(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "ci" / "watch.sh").read_text(encoding="utf-8")
        self.assertIn('carry --from "$pending" --trusted-root "$checkout"', script)


class RepositoryTests(unittest.TestCase):
    def test_snapshot_files_are_not_git_ignored(self) -> None:
        import subprocess

        repo = Path(__file__).resolve().parents[2]
        if not (repo / ".git").exists():
            self.skipTest("not a git checkout")
        for path in (
            "election-analysis/data/elections/washington/2026-11-03/SOURCES.json",
            "election-analysis/data/elections/washington/2026-11-03/input/washington/2026-11-03-2015.csv",
            "election-analysis/data/elections/washington/2026-11-03/raw/washington/2026-11-03-2015.xlsx",
            "election-analysis/data/elections/washington/2026-11-03/raw/pierce-county/2026-11-03-2015.json.gz",
            "election-analysis/reports/washington/2026-11-03/analysis.json",
            "election-analysis/data/elections/washington/2026-11-03/areas/washington/county/2026-11-03-2025.csv",
            "election-analysis/data/elections/washington/2026-11-03/areas/washington/precinct/2026-11-03-2025.csv.gz",
            "election-analysis/reports/washington/2026-11-03/county-breakdown.csv",
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
