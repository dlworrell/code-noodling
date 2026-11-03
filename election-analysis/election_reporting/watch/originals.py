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
The workflow then (1) uploads that directory as a GitHub Actions artifact
with 90-day retention, and (2) archives it to Dropbox under
``/repo/election-analyser/data/`` (owner decision, 2026-11-03; see
``dropbox_archive.py``). The files are deliberately not committed through the
snapshot pull request: a 21 MB workbook, base64-encoded in a GraphQL
``createCommitOnBranch`` request, could make the snapshot publication itself
fail on election night.

Layout of the directory, one folder per state (the owner asked for a
directory per state, each file stored with a header and a SHA stamp):
- ``<state>/<sha256>.<ext>``: the body exactly as served. The SHA-256 in the
  name is the stamp: it proves the content, and a body downloaded again (any
  URL, any run) is not written twice.
- ``<state>/<sha256>.header.json``: the header for that body, written when it
  is first saved: SHA-256, byte length, state, first retrieval (URL, time,
  ``Last-Modified``, ``ETag``, ``Content-Type``) and ``archive_after``. The
  header is a separate file so the original stays byte-identical.
- ``MANIFEST.json``: every body this run downloaded and every retrieval of
  it, including repeat downloads of bodies saved by earlier runs.

The 30-day clock starts at the body's first retrieval in the run that saved
it. ``archive_after`` marks when the retained copy is due for archiving.

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


def _safe_state(state: str) -> str:
    """A state identifier usable as one path component (lowercase letters, digits, hyphens)."""
    cleaned = "".join(ch for ch in state.lower() if ch.isalnum() or ch == "-")
    return cleaned or "unknown"


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
    state: str = "unknown"  # set by the caller before each state's fetch

    def __post_init__(self) -> None:
        manifest = self.directory / MANIFEST
        if manifest.exists():  # a second command in the same run adds to it
            for entry in json.loads(manifest.read_text(encoding="utf-8")).get("files", []):
                self.entries[f"{entry['state']}/{entry['sha256']}"] = entry

    def keep(self, url: str, body: bytes, headers: dict[str, str]) -> str:
        """Save ``body`` and its header unless already saved, record this retrieval, return its SHA-256.

        The body goes to ``<state>/<sha256>.<ext>`` exactly as served; its
        header goes beside it as ``<sha256>.header.json``. Both are written
        once per state and body.
        """
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
        state = _safe_state(self.state)
        key = f"{state}/{digest}"
        entry = self.entries.get(key)
        if entry is None:
            folder = self.directory / state
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{state}/{digest}.{_extension(url, content_type)}"
            (self.directory / name).write_bytes(body)
            entry = {
                "sha256": digest,
                "state": state,
                "file": name,
                "header": f"{state}/{digest}.header.json",
                "bytes": len(body),
                "first_retrieved_at": _iso(now),
                "archive_after": _iso(now + RETAIN),
                "retrievals": [],
            }
            header = {key_: entry[key_] for key_ in ("sha256", "state", "file", "bytes", "first_retrieved_at", "archive_after")}
            header["first_retrieval"] = retrieval
            header["retention"] = "retain 30 days from first retrieval, then archive (owner rule, 2026-11-03)"
            (self.directory / entry["header"]).write_text(
                json.dumps(header, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.entries[key] = entry
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
            "files": sorted(self.entries.values(), key=lambda entry: (entry["state"], entry["sha256"])),
        }
        manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return manifest
