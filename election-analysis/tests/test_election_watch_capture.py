# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:937f55c1-413b-41ad-b9fe-14c7278dc1f2

"""Capture-only collection (election_reporting/watch/capture.py).

Owner decision, 2026-11-03: keep unmodified copies of the official result
files of the platform states (Enhanced Voting, Clarity) and North Carolina.
These tests use fake responses only; no live request is made.
"""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from election_reporting.model import ElectionReportError
from election_reporting.watch import capture
from election_reporting.watch.http import Response, check_url
from election_reporting.watch.originals import MANIFEST, OriginalsKeeper

from test_election_watch import FakeWeb

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "election-analysis" / "data" / "capture.toml"
NIGHT = datetime(2026, 11, 4, 5, 0, tzinfo=timezone.utc)  # every floor has passed


def target(**overrides) -> capture.Target:
    values = dict(state="georgia", platform="enhanced-voting", election=date(2026, 11, 3),
                  not_before=datetime(2026, 11, 4, 0, 0, tzinfo=timezone.utc),
                  end=datetime(2026, 12, 4, tzinfo=timezone.utc),
                  host="results.sos.ga.gov", jurisdiction="georgia")
    values.update(overrides)
    return capture.Target(**values)


def as_json(url: str, payload) -> Response:
    return Response(url, json.dumps(payload).encode("utf-8"), {"content-type": "application/json"})


def georgia_routes() -> dict[str, Response]:
    api = "https://results.sos.ga.gov/results/public/api"
    eid = "11032026GeneralAndSpecialElection"
    return {
        f"{api}/jurisdictions/georgia": as_json(f"{api}/jurisdictions/georgia", {
            "elections": [{"electionDate": "2026-05-19", "publicElectionId": "GeneralPrimary51926"},
                          {"electionDate": "2026-11-03T00:00:00", "publicElectionId": eid}],
            "childLocalities": [{"id": "AAA-1", "shortName": "appling-county-ga"},
                                {"id": "BBB-2", "shortName": "bacon-county-ga"}],
        }),
        f"{api}/elections/georgia/{eid}": as_json(f"{api}/elections/georgia/{eid}", {
            "jurisdictionId": "j-1",
            "publicReportCategories": [{"reports": [{"reportName": "Total Votes Excel", "blobName": "total votes.xlsx"},
                                                    {"reportName": "Not yet", "blobName": None}]}],
        }),
        "https://results.sos.ga.gov/cdn/results/j-1/total%20votes.xlsx":
            Response("https://results.sos.ga.gov/cdn/results/j-1/total%20votes.xlsx", b"PK workbook",
                     {"content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}),
        f"{api}/elections/georgia/{eid}/data": as_json(f"{api}/elections/georgia/{eid}/data", {
            "ballotItems": [], "localityElections": [{"jurisdictionId": "aaa-1"}],  # Bacon has none
        }),
        f"{api}/elections/appling-county-ga/{eid}/data":
            as_json(f"{api}/elections/appling-county-ga/{eid}/data", {"ballotItems": ["local"]}),
    }


def clarity_routes() -> dict[str, Response]:
    base = "https://results.enr.clarityelections.com/IA"
    elections = gzip.compress(json.dumps([
        {"EID": "126082", "Date": "6/2/2026 12:00:00 AM"},
        {"EID": "127077", "Date": "11/3/2026 12:00:00 AM"},
    ]).encode())
    settings = {"settings": {"electiondetails": {"participatingcounties": [
        "Adair|127078|383200|11/3/2026 9:00:00 PM CST|16",
        "O'Brien|127150|383300|11/3/2026 9:05:00 PM CST|16",
        "Broken|x|y",
    ]}}}
    routes = {
        f"{base}/elections.json": Response(f"{base}/elections.json", elections, {"content-encoding": "gzip"}),
        f"{base}/127077/current_ver.txt": Response(f"{base}/127077/current_ver.txt", b"383162\r\n"),
        f"{base}/127077/383162/json/en/electionsettings.json": as_json(
            f"{base}/127077/383162/json/en/electionsettings.json", settings),
    }
    for path in ("127077/383162/reports/detailxml.zip", "127077/383162/reports/summary.zip",
                 "Adair/127078/383200/reports/detailxml.zip", "O%27Brien/127150/383300/reports/detailxml.zip"):
        routes[f"{base}/{path}"] = Response(f"{base}/{path}", f"zip {path}".encode(), {"content-type": "application/zip"})
    return routes


class ConfigTests(unittest.TestCase):
    def test_c1_deployed_config_floors_are_last_poll_closings(self) -> None:
        targets = {t.state: t for t in capture.load_targets(CONFIG)}
        self.assertEqual(sorted(targets), ["colorado", "georgia", "idaho", "iowa", "north-carolina",
                                           "rhode-island", "utah", "virginia"])
        expected = {"georgia": 0, "virginia": 0, "rhode-island": 1, "colorado": 2, "iowa": 2, "utah": 3, "idaho": 4}
        for state, hour in expected.items():
            self.assertEqual(targets[state].not_before, datetime(2026, 11, 4, hour, tzinfo=timezone.utc), state)
        self.assertEqual(targets["north-carolina"].not_before, datetime(2026, 11, 4, 0, 30, tzinfo=timezone.utc))
        self.assertFalse(targets["north-carolina"].counties)
        for t in targets.values():
            self.assertEqual(t.election, date(2026, 11, 3))
            self.assertEqual(t.end, datetime(2026, 12, 4, tzinfo=timezone.utc))

    def test_c2_invalid_entries_are_refused(self) -> None:
        cases = [
            'platform = "scraper"',
            'not_before = 2026-11-04T00:00:00',  # no offset
            'end = 2026-11-03T00:00:00Z',  # before the floor
            'every_minutes = 5',
            'host = ""',
        ]
        for case in cases:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "capture.toml"
                lines = {"platform": 'platform = "enhanced-voting"', "not_before": "not_before = 2026-11-04T00:00:00Z",
                         "end": "end = 2026-12-04T00:00:00Z", "every_minutes": "every_minutes = 180",
                         "host": 'host = "results.sos.ga.gov"'}
                lines[case.split(" ")[0]] = case
                path.write_text("schema_version = 1\n[[capture]]\nstate = \"georgia\"\nelection = 2026-11-03\n"
                                'jurisdiction = "georgia"\n' + "\n".join(lines.values()) + "\n")
                with self.assertRaises(ElectionReportError, msg=case):
                    capture.load_targets(path)


class DueTests(unittest.TestCase):
    def test_d1_floor_night_cadence_and_end(self) -> None:
        t = target(night_hours=8, every_minutes=180)
        floor = t.not_before
        self.assertFalse(capture.due(t, floor - timedelta(seconds=1))[0])
        self.assertTrue(capture.due(t, floor)[0])
        self.assertTrue(capture.due(t, floor + timedelta(hours=7, minutes=59))[0])
        self.assertTrue(capture.due(t, floor + timedelta(hours=9, minutes=10))[0])   # 550 min: slot 0-29 of 180
        self.assertFalse(capture.due(t, floor + timedelta(hours=9, minutes=40))[0])  # 580 min: not due
        self.assertFalse(capture.due(t, t.end)[0])

    def test_d2_a_held_state_makes_no_request(self) -> None:
        def no_requests(url, max_bytes=None):
            raise AssertionError(f"request before the floor: {url}")
        with tempfile.TemporaryDirectory() as directory:
            keeper = OriginalsKeeper(Path(directory))
            lines: list[str] = []
            result = capture.capture(capture.load_targets(CONFIG), keeper,
                                     datetime(2026, 11, 3, 23, 59, tzinfo=timezone.utc), fetch=no_requests, log=lines.append)
            self.assertEqual(result.fetched, [])
            self.assertTrue(all("held until" in line for line in lines))
            self.assertIsNone(keeper.save())


class PlatformTests(unittest.TestCase):
    def run_capture(self, targets, routes):
        web = FakeWeb(routes)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        keeper = OriginalsKeeper(Path(directory.name), clock=lambda: NIGHT)
        result = capture.capture(targets, keeper, NIGHT, fetch=web, log=lambda _: None)
        keeper.save()
        return web, result, Path(directory.name)

    def test_p1_enhanced_voting_statewide_reports_and_counties(self) -> None:
        web, result, directory = self.run_capture([target()], georgia_routes())
        self.assertEqual(result.failures, [])
        self.assertEqual(sorted(web.requests), sorted(georgia_routes()))  # Bacon (no election) not requested
        report = "https://results.sos.ga.gov/cdn/results/j-1/total%20votes.xlsx"
        self.assertIsNotNone(web.caps[report])  # large-file cap
        manifest = json.loads((directory / MANIFEST).read_text())
        self.assertEqual({entry["state"] for entry in manifest["files"]}, {"georgia"})
        self.assertEqual(len(manifest["files"]), 5)
        self.assertIn(b"PK workbook", [p.read_bytes() for p in (directory / "georgia").glob("*.xlsx")])

    def test_p2_clarity_statewide_and_county_detail(self) -> None:
        t = target(state="iowa", platform="clarity", host="", jurisdiction="IA")
        web, result, directory = self.run_capture([t], clarity_routes())
        self.assertEqual(result.failures, [])
        self.assertEqual(sorted(web.requests), sorted(clarity_routes()))
        kept = {p.read_bytes() for p in (directory / "iowa").iterdir() if not p.name.endswith(".header.json")}
        self.assertIn(clarity_routes()["https://results.enr.clarityelections.com/IA/elections.json"].body, kept)  # gzip as served

    def test_p3_north_carolina_one_statewide_zip(self) -> None:
        url = "https://s3.amazonaws.com/dl.ncsbe.gov/ENRS/2026_11_03/results_pct_20261103.zip"
        t = target(state="north-carolina", platform="ncsbe", host="", jurisdiction="", counties=False)
        web, result, _ = self.run_capture([t], {url: Response(url, b"PK zip")})
        self.assertEqual(web.requests, [url])
        self.assertEqual(result.fetched, [url])

    def test_p4_a_failure_never_stops_the_rest(self) -> None:
        routes = georgia_routes()
        del routes["https://results.sos.ga.gov/results/public/api/elections/georgia/11032026GeneralAndSpecialElection"]
        nc = "https://s3.amazonaws.com/dl.ncsbe.gov/ENRS/2026_11_03/results_pct_20261103.zip"
        routes[nc] = Response(nc, b"PK zip")
        targets = [target(), target(state="north-carolina", platform="ncsbe", host="", jurisdiction="", counties=False)]
        web, result, _ = self.run_capture(targets, routes)
        self.assertEqual(len(result.failures), 1)
        self.assertIn(nc, result.fetched)
        self.assertTrue(any("/appling-county-ga/" in url for url in result.fetched))

    def test_p5_election_not_listed_yet_is_quiet(self) -> None:
        routes = georgia_routes()
        api = "https://results.sos.ga.gov/results/public/api/jurisdictions/georgia"
        routes[api] = as_json(api, {"elections": [], "childLocalities": []})
        web, result, _ = self.run_capture([target()], routes)
        self.assertEqual(web.requests, [api])
        self.assertEqual(result.failures, [])

    def test_p7_every_election_listed_for_the_day_is_captured(self) -> None:
        routes = georgia_routes()
        api = "https://results.sos.ga.gov/results/public/api"
        listing = json.loads(routes[f"{api}/jurisdictions/georgia"].body)
        listing["elections"].insert(0, {"electionDate": "2026-11-03", "publicElectionId": "RECOUNT1"})
        routes[f"{api}/jurisdictions/georgia"] = as_json(f"{api}/jurisdictions/georgia", listing)
        routes[f"{api}/elections/georgia/RECOUNT1"] = as_json(f"{api}/elections/georgia/RECOUNT1", {})
        routes[f"{api}/elections/georgia/RECOUNT1/data"] = as_json(f"{api}/elections/georgia/RECOUNT1/data", {})
        web, result, _ = self.run_capture([target()], routes)
        self.assertEqual(result.failures, [])
        self.assertIn(f"{api}/elections/georgia/RECOUNT1/data", web.requests)
        self.assertIn(f"{api}/elections/georgia/11032026GeneralAndSpecialElection/data", web.requests)

    def test_p6_time_budget_stops_counties_then_states(self) -> None:
        ticks = iter(range(0, 1000, 100))  # each clock read advances 100 s; budget 240 s
        web = FakeWeb({**georgia_routes(), **clarity_routes()})
        with tempfile.TemporaryDirectory() as directory:
            result = capture.capture(
                [target(), target(state="iowa", platform="clarity", host="", jurisdiction="IA")],
                OriginalsKeeper(Path(directory)), NIGHT, fetch=web, log=lambda _: None,
                budget_seconds=240, clock=lambda: next(ticks))
        self.assertTrue(result.out_of_time)
        self.assertLess(len(web.requests), len(georgia_routes()) + len(clarity_routes()))


class AllowlistTests(unittest.TestCase):
    def test_a1_capture_prefixes_are_exact(self) -> None:
        check_url("https://s3.amazonaws.com/dl.ncsbe.gov/ENRS/2026_11_03/results_pct_20261103.zip")
        check_url("https://results.enr.clarityelections.com/IA/127077/current_ver.txt")
        for url in ("https://s3.amazonaws.com/other-bucket/x.zip",
                    "https://results.enr.clarityelections.com/GA/1/current_ver.txt",
                    "https://results.sos.ga.gov.example.com/results/public/api/x",
                    "https://s3.amazonaws.com/dl.ncsbe.gov/ENRS/../../other/x",
                    "http://results.sos.ga.gov/results/public/api/x",
                    "https://results.sos.ga.gov/admin"):
            with self.assertRaises(ElectionReportError, msg=url):
                check_url(url)


class IntegrationTests(unittest.TestCase):
    def test_i1_watch_sh_runs_capture_before_the_gate_and_never_fails_the_step(self) -> None:
        script = (ROOT / "election-analysis" / "ci" / "watch.sh").read_text()
        call = script.index("python3 -m election_reporting.watch.capture")
        self.assertLess(call, script.index("--require-eligible"))
        self.assertIn('|| echo "::warning::capture-only collection', script)
        self.assertIn("[[ ${1:-} == fetch && -n ${ORIGINALS_DIR:-}", script)

    def test_i2_cli_before_any_floor_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status = capture.main([directory, "--config", str(CONFIG), "--now", "2026-11-03T20:00:00+00:00"])
            self.assertEqual(status, 0)
            self.assertFalse((Path(directory) / MANIFEST).exists())

    def test_i3_cli_refuses_a_bad_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "capture.toml"
            bad.write_text("schema_version = 2\n")
            self.assertEqual(capture.main([directory, "--config", str(bad)]), 2)


if __name__ == "__main__":
    unittest.main()
