"""Maine source adapter: results page, office workbooks and conversion (offline)."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
from datetime import date, datetime
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

from election_reporting.analysis import analyze_election
from election_reporting.config import load_config
from election_reporting.watch import http, maine
from election_reporting.watch.config import load_watch_settings
from election_reporting.watch.http import Response
from election_reporting.watch.run import fetch_snapshots
from election_reporting.watch.schedule import is_due, watch_window

from test_election_watch import FakeWeb

EASTERN = ZoneInfo("America/New_York")
FILES = "https://www.maine.gov/sos/sites/maine.gov.sos/files/inline-files/"
SENATOR = FILES + "US%20Senator%20FINAL.xlsx"
STATE_SENATE = FILES + "State%20Senator%20FINAL.xlsx"
REFERENDUM = FILES + "State%20Referendum%20FINAL.xlsx"
CG2 = FILES + "Nov26%20CG2%201st%20Choices.xlsx"
CVR = FILES + "Nov26%20CVR%20Export%201.xlsx"


def column(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def workbook(rows: list[list[object]], sheet: str = "Sheet1") -> bytes:
    """A minimal XLSX with one worksheet of inline strings and numbers."""

    def cell(reference: str, value: object) -> str:
        if isinstance(value, (int, float)):
            return f'<c r="{reference}"><v>{value}</v></c>'
        return f'<c r="{reference}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'

    xml_rows = []
    for number, values in enumerate(rows, start=1):
        cells = "".join(
            cell(f"{column(i)}{number}", value) for i, value in enumerate(values) if value != ""
        )
        xml_rows.append(f'<row r="{number}">{cells}</row>')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets><sheet name="{sheet}" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"<sheetData>{''.join(xml_rows)}</sheetData></worksheet>",
        )
    return buffer.getvalue()


# Statewide office: names, residences and parties in three header rows, county
# subtotals, a statewide UOCAVA row and a statewide total (2024 US Senator layout).
SENATOR_XLSX = workbook(
    [
        ["COUNTY", "MUNICIPALITY", "Cherry, Jason S.", "King, Angus S. Jr.", "Kouzounas, Demi", "BLANK", "TBC"],
        ["", "", "Unity", "Brunswick", "Saco"],
        ["", "", "Independent", "Independent", "Republican"],
        ["AND", "Auburn", 309, 6659, 4000, 275, 11243],
        ["AND", "Durham", 61, 1551, 1095, 47, 2754],
        ["", "AND Total", 370, 8210, 5095, 322, 13997],
        ["YOR", "York", 100, 5000, 3000, 100, 8200],
        ["", "YOR Total", 100, 5000, 3000, 100, 8200],
        ["", "STATE UOCAVA", 5, 300, 50, 3, 358],
        ["", "STATE TOTAL", 475, 13510, 8145, 425, 22555],
    ]
)

# Districts stacked in blocks; the second block's candidate names sit above its
# column-header row, which lists parties instead (2024 State Senator layout).
STATE_SENATE_XLSX = workbook(
    [
        ["DIST", "CTY", "Municipality", "Bernard, Susan Y.", "McLaughlin, Vaughn James"],
        ["", "", "", "Caribou", "Fort Fairfield"],
        ["", "", "", "Republican", "Democratic", "Blank", "TBC"],
        [1, "ARO", "Allagash", 86, 59, 9, 154],
        [1, "", "STATE UOCAVA", 50, 142, 14, 206],
        ["", "", "Total", 136, 201, 23, 360],
        [],
        ["", "", "", "Rush, Matthew J.", "Stewart, Harold L. III"],
        ["", "", "", "Houlton", "Presque Isle"],
        ["DIST", "CTY", "Municipalitly", "Democratic", "Republican", "Blank", "TBC"],
        [2, "ARO", "Amity", 38, 111, 11, 160],
        [2, "PEN", "Woodville", 22, 111, 15, 148],
        ["", "", "Total", 60, 222, 26, 308],
    ]
)

# Referendum: questions grouped across columns, plain "UOCAVA", statewide totals.
REFERENDUM_XLSX = workbook(
    [
        ["", "", "Question 1", "", "", "Question 2", "", "", "TBC"],
        ["CTY", "MUNICIPALITY", "YES", "NO", "BLANK", "YES", "NO", "BLANK"],
        ["AND", "AUBURN", 8862, 3178, 466, 6307, 5698, 501, 12506],
        ["AND", "COUNTY TOTALS", 8862, 3178, 466, 6307, 5698, 501, 12506],
        ["UOCAVA", "", 52, 7, 5, 47, 10, 7, 64],
        ["STATEWIDE TOTALS", "", 8914, 3185, 471, 6354, 5708, 508, 12570],
    ],
    sheet="FINAL",
)

# Ranked-choice first choices by precinct: OFFICE column, an unlabeled total
# row that still repeats the office, then a row of shares (2026 primary layout).
CG2_XLSX = workbook(
    [
        ["", "", "", "", "", "Golden, Jared Forrest", "Theriault, Austin", "Others", "BLANK"],
        ["", "", "", "", "", "Lewiston", "Fort Kent"],
        ["", "", "", "", "", "Democratic", "Republican"],
        ["OFFICE", "DIST", "CTY", "TOWN", "WARD-PRECINCT", "", "", "", "", "TBC"],
        ["Representative To Congress", 2, "AND", "Auburn", "1 (1-1/2-1)", 168, 138, 1, 7, 314],
        ["Representative To Congress", 2, "PEN", "Bangor", "1", 300, 310, 2, 8, 620],
        ["Representative To Congress", 2, "", "STATE UOCAVA", "", 20, 10, 0, 1, 31],
        ["Representative To Congress", "", "", "", "", 488, 458, 3, 16, 965],
        ["", "", "", "", "", 0.5057, 0.4746, 0.0031, 0.0166, 1],
    ]
)

PAGE = f"""
<h1>Election Results/Data</h1>
<h2>November 3, 2026 - General Election - Non-Ranked Choice Offices</h2>
<h3>Federal Offices</h3>
<p><a href="/sos/sites/maine.gov.sos/files/inline-files/US%20Senator%20FINAL.xlsx">United States Senator (Excel)</a></p>
<p><a href="/sos/sites/maine.gov.sos/files/inline-files/President%20by%20CD.xlsx">U.S. President by Congressional District (Excel)</a></p>
<h3>State Offices</h3>
<p><a href="{STATE_SENATE}">State Senator (Excel)</a></p>
<h2>November 3, 2026 - General Election - Ranked Choice Office - Representative to Congress</h2>
<p><a href="{CG2}">Rep. to Congress District 2 (Excel)</a></p>
<h4>Cast Vote Records</h4>
<p><a href="{CVR}">Rep. to Congress District 2 (Excel)</a></p>
<h2>November 3, 2026 - Referendum Election</h2>
<h3>State Referendum</h3>
<p><a href="{REFERENDUM}">Tabulation of Votes (Excel)</a></p>
<h2>June 9, 2026 - Primary Election</h2>
<h3>U.S. Senate</h3>
<p><a href="/sos/sites/maine.gov.sos/files/inline-files/US%20Senate%20DEM.xlsx">Democratic</a></p>
"""


def read_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def votes(rows: list[dict[str, str]], contest: str) -> dict[str, int]:
    return {r["Choice"]: int(r["Votes"]) for r in rows if r["Contest"] == contest}


def parse(raw: bytes, label: str, rcv: bool = False):
    contests, problems = maine.parse_workbook(raw, label=label, rcv=rcv, url=FILES + "x.xlsx")
    return {c.name: c for c in contests}, problems


class MaineElectionTests(unittest.TestCase):
    def test_election_days(self) -> None:
        days = {ref.name: ref.day for ref in maine.list_elections(date(2026, 10, 23))}
        self.assertEqual(days["2026 Primary Election"], date(2026, 6, 9))
        self.assertEqual(days["2026 General Election"], date(2026, 11, 3))
        self.assertEqual(days["2025 Referendum Election"], date(2025, 11, 4))
        self.assertEqual(days["2024 Primary Election"], date(2024, 6, 11))

    def test_county_names(self) -> None:
        places = maine.resolve_places(("York", "kennebec county"), False)
        self.assertEqual([p.place_id for p in places], ["maine", "york-county", "kennebec-county"])
        self.assertEqual(len(maine.resolve_places((), True)), 17)
        with self.assertRaisesRegex(Exception, "Valid names: Androscoggin"):
            maine.resolve_places(("Yrok",), False)

    def test_page_links_for_the_election_only(self) -> None:
        links = maine.links_on_page(PAGE, maine.RESULTS_PAGE, date(2026, 11, 3))
        self.assertEqual(
            [(link.label, link.rcv) for link in links],
            [
                ("United States Senator", False),
                ("State Senator", False),
                ("Rep. to Congress District 2", True),
                ("State Referendum", False),
            ],
        )
        self.assertEqual(links[0].url, SENATOR)


class MaineWorkbookTests(unittest.TestCase):
    def test_statewide_office_with_county_subtotals(self) -> None:
        contests, problems = parse(SENATOR_XLSX, "United States Senator")
        self.assertEqual(problems, [])
        senator = contests["United States Senator"]
        self.assertEqual([c[0] for c in senator.choices], ["Cherry, Jason S.", "King, Angus S. Jr.", "Kouzounas, Demi"])
        self.assertEqual(senator.choices[2][1], "Republican")
        self.assertEqual(len(senator.rows), 4)  # three towns and UOCAVA; subtotals dropped
        self.assertEqual(senator.rows[-1][0], "")  # UOCAVA has no county

    def test_district_blocks_with_names_above_the_header(self) -> None:
        contests, problems = parse(STATE_SENATE_XLSX, "State Senator")
        self.assertEqual(problems, [])
        self.assertEqual(sorted(contests), ["State Senator District 1", "State Senator District 2"])
        second = contests["State Senator District 2"]
        self.assertEqual([c[:2] for c in second.choices], [("Rush, Matthew J.", "Democratic"), ("Stewart, Harold L. III", "Republican")])

    def test_referendum_questions(self) -> None:
        contests, problems = parse(REFERENDUM_XLSX, "State Referendum")
        self.assertEqual(problems, [])
        question = contests["State Referendum Question 2"]
        self.assertEqual([c[0] for c in question.choices], ["YES", "NO"])
        self.assertEqual(sum(r[1][0] for r in question.rows), 6354)
        self.assertEqual(sum(r[3] for r in question.rows), 12570)

    def test_ranked_choice_precinct_workbook(self) -> None:
        contests, problems = parse(CG2_XLSX, "Rep. to Congress District 2", rcv=True)
        self.assertEqual(problems, [])
        cg2 = contests["Rep. to Congress District 2"]
        self.assertTrue(cg2.rcv)
        self.assertEqual([c[0] for c in cg2.choices], ["Golden, Jared Forrest", "Theriault, Austin", "Write-in"])
        self.assertTrue(cg2.choices[2][2])

    def test_block_that_does_not_reconcile_is_left_out(self) -> None:
        broken = workbook(
            [
                ["DIST", "CTY", "Municipality", "A, One", "BLANK", "TBC"],
                [1, "ARO", "Allagash", 10, 1, 11],
                ["", "", "Total", 10, 1, 11],
                ["DIST", "CTY", "Municipality", "B, Two", "BLANK", "TBC"],
                [2, "ARO", "Amity", 10, 1, 11],
                ["", "", "Total", 99, 1, 100],
                ["DIST", "CTY", "Municipality", "C, Three", "BLANK", "TBC"],
                [3, "PEN", "Bangor", 5, 0, 5],
            ]
        )
        contests, problems = parse(broken, "State Senator")
        self.assertEqual(list(contests), ["State Senator District 1"])
        self.assertIn("column 4 sums to 10 but the total row says 99", problems[0])
        self.assertIn("no total row", problems[1])

    def test_county_offices_carry_the_county_name(self) -> None:
        commissioners = workbook(
            [
                ["DIST", "CTY", "Municipality", "Poirier, Roland L", "BLANK", "TBC"],
                [2, "AND", "Lewiston", 100, 10, 110],
                ["", "", "Total", 100, 10, 110],
                ["DIST", "CTY", "Municipality", "Smith, Ann", "BLANK", "TBC"],
                [2, "YOR", "Saco", 50, 5, 55],
                ["", "", "Total", 50, 5, 55],
            ]
        )
        contests, problems = parse(commissioners, "County Commissioner")
        self.assertEqual(problems, [])
        self.assertEqual(sorted(contests), ["County Commissioner - Androscoggin District 2", "County Commissioner - York District 2"])

    def test_conversion_keeps_uocava_statewide_only(self) -> None:
        contests, _ = parse(SENATOR_XLSX, "United States Senator")
        places = maine.resolve_places(("Androscoggin", "York"), False)
        outputs = maine.convert(list(contests.values()), places)
        statewide = read_rows(outputs["maine"])
        self.assertEqual(votes(statewide, "United States Senator")["King, Angus S. Jr."], 13510)
        self.assertEqual(statewide[0]["BallotsWith Contest"], "22555")
        self.assertEqual(statewide[0]["Contest Undervotes"], "425")
        androscoggin = read_rows(outputs["androscoggin-county"])
        self.assertEqual(votes(androscoggin, "United States Senator")["King, Angus S. Jr."], 8210)
        self.assertEqual(androscoggin[0]["Counties"], "Androscoggin; York")


def maine_web() -> FakeWeb:
    stamp = {"last-modified": "Mon, 23 Nov 2026 15:00:00 GMT"}
    return FakeWeb(
        {
            maine.RESULTS_PAGE: Response(maine.RESULTS_PAGE, PAGE.encode()),
            SENATOR: Response(SENATOR, SENATOR_XLSX, stamp),
            STATE_SENATE: Response(STATE_SENATE, STATE_SENATE_XLSX, stamp),
            CG2: Response(CG2, CG2_XLSX, {"last-modified": "Tue, 24 Nov 2026 16:30:00 GMT"}),
            REFERENDUM: Response(REFERENDUM, REFERENDUM_XLSX, stamp),
        }
    )


def write_watch(root: Path) -> Path:
    path = root / "watch.toml"
    path.write_text(
        'schema_version = 1\n[[watch]]\nstate = "Maine"\ncounties = ["Androscoggin", "York"]\n'
        'election = "next"\n[watch.schedule]\nelection_night_every_minutes = 1440\nevery_minutes = 1440\n',
        encoding="utf-8",
    )
    return path


class MaineFetchTests(unittest.TestCase):
    def test_fetch_combines_workbooks_into_one_snapshot(self) -> None:
        now = datetime(2026, 11, 24, 20, 7, tzinfo=EASTERN)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_watch_settings(write_watch(root)).watches[0]
            result = fetch_snapshots(config, maine_web(), now, log=lambda _: None)
            self.assertEqual(result.errors, [])
            self.assertEqual(sorted(r.place for r in result.added), ["androscoggin-county", "maine", "york-county"])
            directory = root / "elections" / "maine" / "2026-11-03"
            stamp = "2026-11-24-1130"  # newest workbook, 16:30 UTC in Eastern time
            with zipfile.ZipFile(directory / "raw" / "maine" / f"{stamp}.zip") as archive:
                names = archive.namelist()
                manifest = archive.read("SOURCES.txt").decode()
            self.assertEqual(len(names), 5)
            self.assertIn(CG2, manifest)
            self.assertNotIn(CVR, manifest)
            records = {r["place"]: r for r in json.loads((directory / "SOURCES.json").read_text())["snapshots"]}
            self.assertIn("browser User-Agent", records["maine"]["note"])
            self.assertIn("STATE UOCAVA", records["york-county"]["note"])

            election = (directory / "election.toml").read_text()
            self.assertIn('default_rule = "winner"', election)
            self.assertNotIn("certification_date =", election)
            self.assertIn('contest_id = "rep-to-congress-district-2"\nrule = "rcv"', election)
            analysis = analyze_election(load_config(directory / "election.toml"))
            rules = {race.contest: race.rule for race in analysis.races}
            self.assertEqual(rules["Rep. to Congress District 2"], "rcv")
            self.assertEqual(rules["United States Senator"], "winner")
            self.assertEqual(rules["State Referendum Question 1"], "winner")

            again = fetch_snapshots(config, maine_web(), now, log=lambda _: None)
            self.assertEqual(again.added, [])
            self.assertEqual(election, (directory / "election.toml").read_text())

    def test_nothing_posted_yet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_watch_settings(write_watch(Path(tmp))).watches[0]
            empty = "<h2>June 9, 2026 - Primary Election</h2>"
            year_page = maine.YEAR_PAGE.format(year=2026)
            web = FakeWeb(
                {
                    maine.RESULTS_PAGE: Response(maine.RESULTS_PAGE, empty.encode()),
                    year_page: Response(year_page, empty.encode()),
                }
            )
            result = fetch_snapshots(config, web, datetime(2026, 11, 4, 20, 7, tzinfo=EASTERN), log=lambda _: None)
            self.assertEqual(result.skipped, ["maine: no results posted for this election yet"])

    def test_window_is_daily_for_35_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_watch_settings(write_watch(Path(tmp))).watches[0]
        window = watch_window(config, date(2026, 11, 3), "general", has_state=True)
        self.assertEqual(window.start, datetime(2026, 11, 3, 20, 0, tzinfo=EASTERN))
        self.assertEqual(window.end, datetime(2026, 12, 9, 0, 0, tzinfo=EASTERN))
        self.assertTrue(is_due(config, window, datetime(2026, 11, 10, 20, 7, tzinfo=EASTERN)))
        self.assertFalse(is_due(config, window, datetime(2026, 11, 10, 20, 37, tzinfo=EASTERN)))
        self.assertFalse(is_due(config, window, datetime(2026, 11, 10, 9, 7, tzinfo=EASTERN)))


class MaineHttpTests(unittest.TestCase):
    def test_browser_user_agent_is_sent_to_maine_only(self) -> None:
        sent = {}

        class Opener:
            def open(self, request, timeout):
                sent[request.full_url] = request.get_header("User-agent")
                raise http.urllib.error.URLError("offline")

        original = http._opener
        http._opener = lambda host: Opener()
        try:
            for url in (maine.RESULTS_PAGE, "https://results.votewa.gov/x"):
                with self.assertRaises(Exception):
                    http.fetch(url)
        finally:
            http._opener = original
        self.assertEqual(sent[maine.RESULTS_PAGE], http.BROWSER_USER_AGENT)
        self.assertEqual(sent["https://results.votewa.gov/x"], http.USER_AGENT)


if __name__ == "__main__":
    unittest.main()
