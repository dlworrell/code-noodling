# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:3f6b0d2e-8c41-4a57-9e1d-6a2b7c5f0e93

"""Keep every downloaded file, byte for byte, for 30 days and then archive it.

Purpose. Donovan's retention rule (2026-11-03): every file the watcher
downloads is saved unmodified and retained for 30 days, then archived. Some
adapters store only a derived or trimmed file in the repository (the
Washington "Summary and Precinct Results" workbook, VoteWA JSON), and none of
them stores the discovery pages. This module saves the original response body
of every successful download, so any published number can later be re-derived
from the exact bytes the official source served.

Where the files go. ``OriginalsKeeper`` writes into a directory outside the
repository tree (the workflow passes ``$RUNNER_TEMP/election-originals``).
The workflow uploads that directory as a GitHub Actions artifact with 90-day
retention. The files are deliberately not committed through the snapshot pull
request: a 21 MB workbook, base64-encoded in a GraphQL ``createCommitOnBranch``
request, could make the snapshot publication itself fail on election night.

Layout of the directory:
- ``<sha256>.<ext>``: one file per distinct body, named by its SHA-256, so the
  name proves the content. A body downloaded again (any URL, any run) is not
  written twice.
- ``MANIFEST.json``: one entry per distinct body, holding its SHA-256, byte
  length, file name, the first retrieval time, the date 30 days later after
  which it is due for archiving (``archive_after``), and every retrieval in
  this run (URL, time, and the ``Last-Modified``, ``ETag`` and
  ``Content-Type`` headers the source sent).

The 30-day clock starts at the body's first retrieval in the run that saved
it. Archiving itself (moving files older than 30 days to a long-term store
before the artifact expires at 90 days) needs an archive destination, which is
an owner decision not yet made. Until then the artifact is the retained copy.

Limits:
- Only successful downloads are kept. A request that fails or is refused
  (oversize, wrong host) has no body to keep; the failure is still reported
  in the run's errors and coverage.
- The artifact is one per workflow run. Retention is therefore per run: each
  run's artifact holds the bodies that run downloaded.
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .store import sha256

MANIFEST = "MANIFEST.json"
RETAIN = timedelta(days=30)  # Donovan, 2026-11-03: retain 30 days, then archive

# File extensions by declared content type, then by URL suffix. The extension
# is a convenience for whoever opens the file; the SHA-256 is the identity.
_TYPES = {
    "text/html": "html",
    "text/csv": "csv",
    "application/json": "json",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/zip": "zip",
    "text/plain": "txt",
}
_SUFFIXES = ("xlsx", "csv", "json", "zip", "html", "htm", "txt", "xml", "pdf")


def _extension(url: str, content_type: str) -> str:
    """A file extension for a body: from Content-Type, else the URL path, else ``bin``."""
    declared = content_type.split(";", 1)[0].strip().lower()
    if declared in _TYPES:
        return _TYPES[declared]
    path = urllib.parse.urlsplit(url).path.lower()
    for suffix in _SUFFIXES:
        if path.endswith("." + suffix):
            return "html" if suffix == "htm" else suffix
    return "bin"


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


@dataclass
class OriginalsKeeper:
    """Saves each downloaded body once and records every retrieval of it.

    Inputs: ``directory``, where files and ``MANIFEST.json`` are written (it is
    created if missing); ``clock``, which returns the current time (injectable
    for tests).

    Output: files in ``directory``. ``save()`` writes the manifest; call it
    once at the end of the run (the CLI does this even when a state fails).
    """

    directory: Path
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    entries: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        manifest = self.directory / MANIFEST
        if manifest.exists():  # a second command in the same run adds to it
            for entry in json.loads(manifest.read_text(encoding="utf-8")).get("files", []):
                self.entries[entry["sha256"]] = entry

    def keep(self, url: str, body: bytes, headers: dict[str, str]) -> str:
        """Save ``body`` unless already saved, record this retrieval, and return its SHA-256."""
        digest = sha256(body)
        now = self.clock()
        content_type = headers.get("content-type", "")
        retrieval = {
            "url": url,
            "retrieved_at": _iso(now),
            "last_modified": headers.get("last-modified", ""),
            "etag": headers.get("etag", ""),
            "content_type": content_type,
        }
        entry = self.entries.get(digest)
        if entry is None:
            name = f"{digest}.{_extension(url, content_type)}"
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / name).write_bytes(body)
            entry = {
                "sha256": digest,
                "file": name,
                "bytes": len(body),
                "first_retrieved_at": _iso(now),
                "archive_after": _iso(now + RETAIN),
                "retrievals": [],
            }
            self.entries[digest] = entry
        entry["retrievals"].append(retrieval)
        return digest

    def recording(self, fetch):
        """Wrap a fetch function so every successful response body is kept.

        The wrapper has the same signature and returns the same response, so
        adapters are unchanged. Exceptions pass through untouched; nothing is
        kept for a failed request.
        """

        def fetch_and_keep(url, *args, **kwargs):
            response = fetch(url, *args, **kwargs)
            self.keep(response.url or url, response.body, dict(response.headers))
            return response

        return fetch_and_keep

    def save(self) -> Path | None:
        """Write ``MANIFEST.json`` (sorted, deterministic). Returns its path, or None if nothing was kept."""
        if not self.entries:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = self.directory / MANIFEST
        payload = {
            "schema": "election-originals.v1",
            "retention": "retain 30 days from first retrieval, then archive (owner rule, 2026-11-03)",
            "files": sorted(self.entries.values(), key=lambda entry: entry["sha256"]),
        }
        manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return manifest
