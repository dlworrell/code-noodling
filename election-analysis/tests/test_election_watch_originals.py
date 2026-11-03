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
            keep = OriginalsKeeper(Path(directory), clock=lambda: T0)
            digest = keep.keep(
                "https://results.votewa.gov/cdn/results/1/Summary.xlsx",
                body,
                {"content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                 "last-modified": "Wed, 04 Nov 2026 03:55:00 GMT", "etag": '"abc"'},
            )
            keep.save()
            self.assertEqual(digest, sha(body))
            self.assertEqual((Path(directory) / f"{digest}.xlsx").read_bytes(), body)
            entry = manifest(Path(directory))["files"][0]
            self.assertEqual(entry["bytes"], len(body))
            self.assertEqual(entry["first_retrieved_at"], "2026-11-04T04:00:00+00:00")
            self.assertEqual(entry["archive_after"], (T0 + timedelta(days=30)).isoformat(timespec="seconds"))
            self.assertEqual(entry["retrievals"][0]["last_modified"], "Wed, 04 Nov 2026 03:55:00 GMT")
            self.assertEqual(entry["retrievals"][0]["etag"], '"abc"')

    def test_o2_same_body_written_once_with_every_retrieval(self) -> None:
        moments = iter([T0, T0 + timedelta(minutes=30)])
        with tempfile.TemporaryDirectory() as directory:
            keep = OriginalsKeeper(Path(directory), clock=lambda: next(moments))
            keep.keep("https://kingcounty.gov/a.csv", b"same", {})
            keep.keep("https://kingcounty.gov/a.csv?rev=2", b"same", {})
            keep.save()
            files = sorted(p.name for p in Path(directory).iterdir())
            self.assertEqual(files, sorted([MANIFEST, f"{sha(b'same')}.csv"]))
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
            keep = OriginalsKeeper(originals, clock=lambda: T0)
            result = fetch_snapshots(config, keep.recording(fake), ELECTION_NIGHT, log=lambda _: None)
            keep.save()
            self.assertEqual(result.errors, [])
            served = {sha(fake.routes[url].body) for url in fake.requests}
            kept = {entry["sha256"] for entry in manifest(originals)["files"]}
            self.assertEqual(kept, served)
            for entry in manifest(originals)["files"]:
                self.assertEqual(sha((originals / entry["file"]).read_bytes()), entry["sha256"])
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
            self.assertEqual((originals / f"{sha(b'<html>results</html>')}.html").read_bytes(),
                             b"<html>results</html>")
            self.assertEqual(len(manifest(originals)["files"]), 1)


class WorkflowContractTests(unittest.TestCase):
    def test_o6_workflow_uploads_originals_for_90_days_and_wrapper_forwards_them(self) -> None:
        workflow = (ROOT / ".github/workflows/election-source-watch.yml").read_text(encoding="utf-8")
        self.assertIn("ORIGINALS_DIR: ${{ runner.temp }}/election-originals", workflow)
        self.assertIn("path: ${{ runner.temp }}/election-originals/", workflow)
        self.assertIn("retention-days: 90", workflow)
        self.assertIn("if: ${{ !cancelled() }}", workflow.split("Keep downloaded originals", 1)[1].split("uses:", 1)[0])
        wrapper = (ROOT / "election-analysis/ci/watch.sh").read_text(encoding="utf-8")
        self.assertIn('args+=(--keep-originals "$ORIGINALS_DIR")', wrapper)


if __name__ == "__main__":
    unittest.main()
