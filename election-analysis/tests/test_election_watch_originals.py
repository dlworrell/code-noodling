# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:9b2d7e41-5c0a-4f68-8e3b-1d6a4c9f2b70

"""Every downloaded file is kept unmodified for 30 days, then archived (offline).

Owner retention rule, 2026-11-03: "save all downloaded files and archive them
after 30 days". Cases:
- O1: a body is written byte for byte, named by its SHA-256, with a manifest
  entry whose archive date is 30 days after its first retrieval.
- O2: the same body fetched again (another URL) is written once, with both
  retrievals recorded.
- O3: the fetch wrapper returns the response unchanged; a failed request
  raises as before and keeps nothing.
- O4: a full Washington run keeps every body it downloaded, including the
  originals the repository stores only in derived or trimmed form, and adds
  nothing to the repository tree.
- O5: the CLI's --keep-originals wraps the fetch and writes the manifest.
- O6: the workflow passes ORIGINALS_DIR and uploads it as a 90-day artifact,
  and ci/watch.sh forwards it.
- O7 (#116): a failure to save an original warns and the fetch continues.
- O8 (#118): a second keeper on the same directory merges the manifest;
  extension and state-name fallbacks.
- D6-D10 (#117): the CLI success path forwards --base; oversize, 409 without
  "conflict", an empty directory, and a successful token response.
- D1-D5: the Dropbox archive (owner decision, 2026-11-03) puts each body and
  its header under /data/<state>/ in the app's own folder (the app has
  App folder access only, so this is /Apps/election-analyser/data/), never
  overwrites, treats an existing SHA-named file as already archived, reports
  real failures, and skips with a warning (exit 0) without credentials.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from election_reporting.model import ElectionReportError
from election_reporting.watch import __main__ as watch_cli
from election_reporting.watch import dropbox_archive
from election_reporting.watch.http import Response
from election_reporting.watch.originals import MANIFEST, OriginalsKeeper
from election_reporting.watch.run import fetch_snapshots

from test_election_watch import ELECTION_NIGHT, FakeWeb, load_watch_config, web, write_watch

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 11, 4, 4, 0, tzinfo=timezone.utc)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(directory: Path) -> dict:
    return json.loads((directory / MANIFEST).read_text(encoding="utf-8"))


class KeeperTests(unittest.TestCase):
    def test_o1_body_kept_byte_for_byte_with_30_day_archive_date(self) -> None:
        body = b"PK\x03\x04 workbook bytes \x00\xff"
        with tempfile.TemporaryDirectory() as directory:
            keep = OriginalsKeeper(Path(directory), clock=lambda: T0, state="washington")
            digest = keep.keep(
                "https://results.votewa.gov/cdn/results/1/Summary.xlsx",
                body,
                {"content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                 "last-modified": "Wed, 04 Nov 2026 03:55:00 GMT", "etag": '"abc"'},
            )
            keep.save()
            self.assertEqual(digest, sha(body))
            self.assertEqual((Path(directory) / "washington" / f"{digest}.xlsx").read_bytes(), body)
            header = json.loads((Path(directory) / "washington" / f"{digest}.header.json").read_text())
            self.assertEqual(header["sha256"], digest)
            self.assertEqual(header["state"], "washington")
            self.assertEqual(header["first_retrieval"]["url"], "https://results.votewa.gov/cdn/results/1/Summary.xlsx")
            self.assertEqual(header["archive_after"], (T0 + timedelta(days=30)).isoformat(timespec="seconds"))
            entry = manifest(Path(directory))["files"][0]
            self.assertEqual(entry["bytes"], len(body))
            self.assertEqual(entry["first_retrieved_at"], "2026-11-04T04:00:00+00:00")
            self.assertEqual(entry["archive_after"], (T0 + timedelta(days=30)).isoformat(timespec="seconds"))
            self.assertEqual(entry["retrievals"][0]["last_modified"], "Wed, 04 Nov 2026 03:55:00 GMT")
            self.assertEqual(entry["retrievals"][0]["etag"], '"abc"')

    def test_o2_same_body_written_once_with_every_retrieval(self) -> None:
        moments = iter([T0, T0 + timedelta(minutes=30)])
        with tempfile.TemporaryDirectory() as directory:
            keep = OriginalsKeeper(Path(directory), clock=lambda: next(moments), state="washington")
            keep.keep("https://kingcounty.gov/a.csv", b"same", {})
            keep.keep("https://kingcounty.gov/a.csv?rev=2", b"same", {})
            keep.save()
            files = sorted(p.relative_to(directory).as_posix() for p in Path(directory).rglob("*") if p.is_file())
            self.assertEqual(files, sorted([MANIFEST, f"washington/{sha(b'same')}.csv",
                                            f"washington/{sha(b'same')}.header.json"]))
            entry = manifest(Path(directory))["files"][0]
            self.assertEqual([r["url"] for r in entry["retrievals"]],
                             ["https://kingcounty.gov/a.csv", "https://kingcounty.gov/a.csv?rev=2"])
            # The 30-day clock starts at the first retrieval, not the latest.
            self.assertEqual(entry["first_retrieved_at"], "2026-11-04T04:00:00+00:00")

    def test_o3_wrapper_is_transparent_and_failures_keep_nothing(self) -> None:
        page = "https://www.maine.gov/sos/results"
        fake = FakeWeb({page: Response(page, b"<html>results</html>", {"content-type": "text/html"})})
        with tempfile.TemporaryDirectory() as directory:
            keep = OriginalsKeeper(Path(directory), clock=lambda: T0)
            fetch = keep.recording(fake)
            self.assertIs(fetch(page), fake.routes[page])
            with self.assertRaises(ElectionReportError):
                fetch("https://www.maine.gov/sos/missing")
            keep.save()
            self.assertEqual([e["retrievals"][0]["url"] for e in manifest(Path(directory))["files"]], [page])
            self.assertEqual(fake.caps[page], None)  # the size cap argument still reaches the fetch

    def test_o4_full_run_keeps_every_download_outside_the_repository_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = root / "originals"
            config = load_watch_config(write_watch(root))
            fake = web(root)
            keep = OriginalsKeeper(originals, clock=lambda: T0, state="washington")
            result = fetch_snapshots(config, keep.recording(fake), ELECTION_NIGHT, log=lambda _: None)
            keep.save()
            self.assertEqual(result.errors, [])
            served = {sha(fake.routes[url].body) for url in fake.requests}
            kept = {entry["sha256"] for entry in manifest(originals)["files"]}
            self.assertEqual(kept, served)
            for entry in manifest(originals)["files"]:
                self.assertEqual(sha((originals / entry["file"]).read_bytes()), entry["sha256"])
                self.assertTrue(entry["file"].startswith("washington/"))
                self.assertTrue((originals / entry["header"]).is_file())
            # VoteWA JSON is stored trimmed in the repository; the original is kept here untrimmed.
            votewa = next(url for url in fake.requests if url.endswith("/data"))
            self.assertIn(sha(fake.routes[votewa].body), kept)
            election = root / "elections" / "washington" / "2026-11-03"
            self.assertFalse(any("originals" in p.parts for p in election.rglob("*")))

    def test_o5_cli_keep_originals_wraps_the_fetch_and_writes_the_manifest(self) -> None:
        page = "https://www.maine.gov/sos/results"
        fake = FakeWeb({page: Response(page, b"<html>results</html>", {"content-type": "text/html"})})

        def fetch_one(config, fetch, now, force=False):
            fetch(page)
            return mock.Mock(added=[], errors=[], skipped=["ok"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            watch_toml = write_watch(root)
            originals = root / "kept"
            with mock.patch.object(watch_cli, "http_fetch", fake), \
                    mock.patch.object(watch_cli, "fetch_snapshots", fetch_one):
                status = watch_cli.main(["fetch", "--config", str(watch_toml), "--keep-originals", str(originals)])
            self.assertEqual(status, 0)
            state = load_watch_config(watch_toml).state_id
            self.assertEqual((originals / state / f"{sha(b'<html>results</html>')}.html").read_bytes(),
                             b"<html>results</html>")
            self.assertEqual(len(manifest(originals)["files"]), 1)


class KeeperRobustnessTests(unittest.TestCase):
    def test_o7_failure_to_keep_never_stops_the_fetch(self) -> None:
        page = "https://www.maine.gov/sos/results"
        fake = FakeWeb({page: Response(page, b"<html>results</html>", {"content-type": "text/html"})})
        with tempfile.TemporaryDirectory() as directory:
            keep = OriginalsKeeper(Path(directory), clock=lambda: T0, state="maine")
            with mock.patch.object(keep, "keep", side_effect=OSError("No space left on device")), \
                    mock.patch("builtins.print") as printed:
                response = keep.recording(fake)(page)
            self.assertIs(response, fake.routes[page])
            self.assertEqual(len(keep.failures), 1)
            self.assertIn("No space left", keep.failures[0])
            self.assertIn("::warning::", printed.call_args[0][0])

    def test_o8_second_keeper_merges_and_fallbacks_hold(self) -> None:
        from election_reporting.watch.originals import _extension, _safe_state
        with tempfile.TemporaryDirectory() as directory:
            first = OriginalsKeeper(Path(directory), clock=lambda: T0, state="alaska")
            first.keep("https://www.elections.alaska.gov/r.csv", b"rows", {})
            first.save()
            second = OriginalsKeeper(Path(directory), clock=lambda: T0 + timedelta(hours=1), state="alaska")
            second.keep("https://www.elections.alaska.gov/r.csv", b"rows", {})
            second.save()
            entry = manifest(Path(directory))["files"][0]
            self.assertEqual(len(entry["retrievals"]), 2)
            self.assertEqual(entry["first_retrieved_at"], "2026-11-04T04:00:00+00:00")
        cases = [
            ("https://x.gov/a", "text/csv; charset=utf-8", "csv"),
            ("https://x.gov/page.htm", "", "html"),
            ("https://x.gov/file.XLSX?rev=2", "", "xlsx"),
            ("https://x.gov/data", "application/octet-stream", "bin"),
        ]
        for url, content_type, expected in cases:
            self.assertEqual(_extension(url, content_type), expected, url)
        self.assertEqual(_safe_state("Washington"), "washington")
        self.assertEqual(_safe_state("../etc"), "etc")
        self.assertEqual(_safe_state(""), "unknown")


class FakeDropbox:
    """Records uploads; answers 409 conflict for paths already present and
    serves get_metadata with Dropbox's content hash of the stored bytes."""

    def __init__(self, existing=None, fail=(), folders=()):
        self.files: dict[str, bytes] = dict(existing or {})
        self.folders = set(folders)
        self.fail = set(fail)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, data, headers):
        if url == dropbox_archive.TOKEN_URL:
            return 200, b'{"access_token": "short-lived"}'
        if url == dropbox_archive.METADATA_URL:
            path = json.loads(data)["path"]
            if path in self.folders:
                return 200, json.dumps({".tag": "folder"}).encode()
            if path in self.files:
                return 200, json.dumps({".tag": "file",
                                        "content_hash": dropbox_archive.content_hash(self.files[path])}).encode()
            return 409, b'{"error_summary": "path/not_found/"}'
        argument = json.loads(headers["Dropbox-API-Arg"])
        self.calls.append((argument["path"], argument))
        if argument["path"] in self.fail:
            return 500, b"server error"
        if argument["path"] in self.files or argument["path"] in self.folders:
            return 409, b'{"error_summary": "path/conflict/file/"}'
        self.files[argument["path"]] = data
        return 200, b"{}"


def kept_run(directory: Path) -> OriginalsKeeper:
    keep = OriginalsKeeper(directory, clock=lambda: T0, state="maine")
    keep.keep("https://www.maine.gov/sos/a.xlsx", b"workbook", {})
    keep.state = "alaska"
    keep.keep("https://www.elections.alaska.gov/r.csv", b"precincts", {})
    keep.save()
    return keep


class DropboxArchiveTests(unittest.TestCase):
    BASE = "/data"  # relative to the App folder

    def test_d1_bodies_and_headers_go_to_state_folders_by_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            kept_run(Path(directory))
            box = FakeDropbox()
            status = dropbox_archive.archive(Path(directory), "41-1", "token", post=box, log=lambda _: None)
            self.assertEqual(status, 0)
            self.assertEqual(sorted(box.files), sorted([
                f"{self.BASE}/alaska/{sha(b'precincts')}.csv",
                f"{self.BASE}/alaska/{sha(b'precincts')}.header.json",
                f"{self.BASE}/maine/{sha(b'workbook')}.xlsx",
                f"{self.BASE}/maine/{sha(b'workbook')}.header.json",
                f"{self.BASE}/manifests/41-1.json",
            ]))
            self.assertEqual(box.files[f"{self.BASE}/maine/{sha(b'workbook')}.xlsx"], b"workbook")

    def test_d0_default_base_stays_inside_the_app_folder(self) -> None:
        # App folder apps address paths relative to /Apps/<app name>/; the
        # default must not name any location outside that folder.
        self.assertEqual(dropbox_archive.BASE, "/data")

    def test_d2_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            kept_run(Path(directory))
            box = FakeDropbox()
            dropbox_archive.archive(Path(directory), "41-1", "token", post=box, log=lambda _: None)
            self.assertTrue(all(arg["mode"] == "add" and arg["autorename"] is False for _, arg in box.calls))

    def test_d3_conflict_is_success_only_when_content_hash_matches(self) -> None:
        body_path = f"{self.BASE}/maine/{sha(b'workbook')}.xlsx"
        header_path = f"{self.BASE}/maine/{sha(b'workbook')}.header.json"
        with tempfile.TemporaryDirectory() as directory:
            kept_run(Path(directory))
            # Same bytes already archived by an earlier run (with that run's header): verified,
            # and this run's header is not uploaded over the first one.
            box = FakeDropbox(existing={body_path: b"workbook", header_path: b"{earlier header}"})
            lines: list[str] = []
            self.assertEqual(dropbox_archive.archive(Path(directory), "41-1", "token", post=box, log=lines.append), 0)
            self.assertIn(f"already archived (verified): {body_path}", lines)
            self.assertIn(f"first header kept: {header_path}", lines)
            self.assertEqual(box.files[header_path], b"{earlier header}")
            # Different bytes under the same name, or a folder in the way: failure, never "archived".
            for box in (FakeDropbox(existing={body_path: b""}), FakeDropbox(folders=[body_path])):
                lines = []
                self.assertEqual(dropbox_archive.archive(Path(directory), "41-1", "token", post=box, log=lines.append), 1)
                self.assertTrue(any(line.startswith("::error::") and body_path in line for line in lines))
                self.assertFalse(any(line.startswith("already archived") and body_path in line for line in lines))

    def test_d3c_missing_header_of_archived_body_is_uploaded_and_retry_repairs(self) -> None:
        # Em's 10:57 negative case: body already archived with a verified hash, header absent.
        body_path = f"{self.BASE}/maine/{sha(b'workbook')}.xlsx"
        header_path = f"{self.BASE}/maine/{sha(b'workbook')}.header.json"
        with tempfile.TemporaryDirectory() as directory:
            kept_run(Path(directory))
            box = FakeDropbox(existing={body_path: b"workbook"})
            lines: list[str] = []
            status = dropbox_archive.archive(Path(directory), "41-1", "token", post=box, log=lines.append)
            self.assertEqual(status, 0)
            self.assertIn(header_path, box.files)  # uploaded, not skipped
            self.assertIn(f"archived: {header_path}", lines)
            # A folder where the header should be is a failure, not "kept".
            box = FakeDropbox(existing={body_path: b"workbook"}, folders=[header_path])
            lines = []
            self.assertEqual(dropbox_archive.archive(Path(directory), "41-1", "token", post=box, log=lines.append), 1)
            self.assertTrue(any(line.startswith("::error::") and header_path in line for line in lines))

    def test_d3d_manifest_save_failure_never_fails_the_fetch(self) -> None:
        page = "https://www.maine.gov/sos/results"
        fake = FakeWeb({page: Response(page, b"<html>x</html>", {"content-type": "text/html"})})

        def fetch_one(config, fetch, now, force=False):
            fetch(page)
            return mock.Mock(added=[], errors=[], skipped=["ok"])

        with tempfile.TemporaryDirectory() as directory:
            watch_toml = write_watch(Path(directory))
            with mock.patch.object(watch_cli, "http_fetch", fake), \
                    mock.patch.object(watch_cli, "fetch_snapshots", fetch_one), \
                    mock.patch.object(OriginalsKeeper, "save", side_effect=OSError("disk full")), \
                    mock.patch("builtins.print") as printed:
                status = watch_cli.main(["fetch", "--config", str(watch_toml),
                                         "--keep-originals", str(Path(directory) / "kept")])
            self.assertEqual(status, 0)
            self.assertTrue(any("could not write the originals manifest" in str(c.args[0]) for c in printed.call_args_list))

    def test_d3e_header_note_does_not_claim_archival(self) -> None:
        from election_reporting.watch.originals import RETENTION_NOTE
        self.assertIn("archived only when the archive step uploads it", RETENTION_NOTE)

    def test_d3b_transport_error_fails_one_file_and_the_rest_continue(self) -> None:
        import urllib.error
        failing = f"{self.BASE}/alaska/{sha(b'precincts')}.csv"
        box = FakeDropbox()

        def flaky(url, data, headers):
            if "Dropbox-API-Arg" in headers and json.loads(headers["Dropbox-API-Arg"])["path"] == failing:
                raise urllib.error.URLError("timed out")
            return box(url, data, headers)

        with tempfile.TemporaryDirectory() as directory:
            kept_run(Path(directory))
            lines: list[str] = []
            with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
                status, body = dropbox_archive._post("https://content.dropboxapi.com/x", b"", {})
            self.assertEqual(status, dropbox_archive.TRANSPORT_ERROR)
            self.assertIn(b"transport error", body)

            def transport(url, data, headers):
                try:
                    return flaky(url, data, headers)
                except urllib.error.URLError as error:
                    return dropbox_archive.TRANSPORT_ERROR, f"transport error: {error.reason}".encode()

            status = dropbox_archive.archive(Path(directory), "41-1", "token", post=transport, log=lines.append)
            self.assertEqual(status, 1)
            self.assertIn(f"{self.BASE}/maine/{sha(b'workbook')}.xlsx", box.files)
            self.assertIn(f"{self.BASE}/manifests/41-1.json", box.files)

    def test_d4_real_failure_is_reported_and_others_continue(self) -> None:
        failing = f"{self.BASE}/alaska/{sha(b'precincts')}.csv"
        with tempfile.TemporaryDirectory() as directory:
            kept_run(Path(directory))
            box = FakeDropbox(fail=[failing])
            lines: list[str] = []
            status = dropbox_archive.archive(Path(directory), "41-1", "token", post=box, log=lines.append)
            self.assertEqual(status, 1)
            self.assertTrue(any(line.startswith("::error::") and failing in line for line in lines))
            self.assertIn(f"{self.BASE}/maine/{sha(b'workbook')}.xlsx", box.files)

    def test_d5_missing_credentials_warn_and_exit_zero_without_leaking(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict("os.environ", {"DROPBOX_APP_KEY": "k"}, clear=True), \
                mock.patch("builtins.print") as printed:
            self.assertEqual(dropbox_archive.main([directory, "--label", "x"]), 0)
            self.assertIn("NOT archived", printed.call_args[0][0])
        with self.assertRaises(RuntimeError) as failure:
            dropbox_archive.access_token("k", "s", "secret-refresh", post=lambda *_: (400, b"secret-refresh"))
        self.assertNotIn("secret-refresh", str(failure.exception))


class DropboxEdgeTests(unittest.TestCase):
    def test_d6_cli_success_path_forwards_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            kept_run(Path(directory))
            box = FakeDropbox()
            env = {"DROPBOX_APP_KEY": "k", "DROPBOX_APP_SECRET": "s", "DROPBOX_REFRESH_TOKEN": "r"}
            with mock.patch.dict("os.environ", env, clear=True), \
                    mock.patch.object(dropbox_archive, "_post", box), mock.patch("builtins.print"):
                status = dropbox_archive.main([directory, "--label", "7-1", "--base", "/elsewhere"])
            self.assertEqual(status, 0)
            self.assertTrue(box.files)
            self.assertTrue(all(path.startswith("/elsewhere/") for path in box.files))

    def test_d7_oversize_is_refused(self) -> None:
        self.assertEqual(dropbox_archive.content_hash(b""), hashlib.sha256(b"").hexdigest())
        with mock.patch.object(dropbox_archive, "MAX_SINGLE_UPLOAD", 3):
            with self.assertRaises(RuntimeError):
                dropbox_archive.upload("t", "/data/x", b"four", post=FakeDropbox())

    def test_d8_409_without_conflict_is_a_failure(self) -> None:
        with self.assertRaises(RuntimeError):
            dropbox_archive.upload("t", "/data/x", b"a", post=lambda *_: (409, b'{"error_summary": "path/insufficient_space/"}'))

    def test_d9_empty_directory_has_nothing_to_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lines: list[str] = []
            self.assertEqual(dropbox_archive.archive(Path(directory), "x", "t", post=FakeDropbox(), log=lines.append), 0)
            self.assertIn("No downloaded originals", lines[0])

    def test_d10_token_response_is_parsed(self) -> None:
        token = dropbox_archive.access_token("k", "s", "r", post=lambda *_: (200, b'{"access_token": "abc"}'))
        self.assertEqual(token, "abc")


class WorkflowContractTests(unittest.TestCase):
    def test_o6_workflow_uploads_originals_for_90_days_and_wrapper_forwards_them(self) -> None:
        workflow = (ROOT / ".github/workflows/election-source-watch.yml").read_text(encoding="utf-8")
        self.assertIn("ORIGINALS_DIR: ${{ runner.temp }}/election-originals", workflow)
        self.assertIn("path: ${{ runner.temp }}/election-originals/", workflow)
        self.assertIn("retention-days: 90", workflow)
        self.assertIn("if: ${{ !cancelled() }}", workflow.split("Keep downloaded originals", 1)[1].split("uses:", 1)[0])
        wrapper = (ROOT / "election-analysis/ci/watch.sh").read_text(encoding="utf-8")
        self.assertIn('args+=(--keep-originals "$ORIGINALS_DIR")', wrapper)
        # The Dropbox archive runs after publication, never blocks it, and takes secrets from the environment.
        archive = workflow.split("Archive originals to Dropbox", 1)[1]
        self.assertLess(workflow.index("Publish verified commit"), workflow.index("Archive originals to Dropbox"))
        for secret in ("DROPBOX_APP_KEY", "DROPBOX_APP_SECRET", "DROPBOX_REFRESH_TOKEN"):
            self.assertIn(f"${{{{ secrets.{secret} }}}}", archive)
        self.assertIn("election-analysis/ci/watch.sh archive", archive)
        self.assertIn("python3 -m election_reporting.watch.dropbox_archive", wrapper)


if __name__ == "__main__":
    unittest.main()
