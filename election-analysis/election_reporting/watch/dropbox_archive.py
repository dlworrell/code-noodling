# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:c41e8a27-6d93-4b5f-a0e2-7f18d3b96c54

"""Archive a run's downloaded originals to Dropbox.

Purpose. Owner decisions (2026-11-03): downloaded files go to Dropbox, one
directory per state, each file stored with a header and a SHA stamp, and the
Dropbox app gets access to its own data area only, not the whole Dropbox.
``originals.OriginalsKeeper`` already lays a run's files out that way; this
uploads them unchanged.

Access. The app is an **App folder** app: Dropbox confines it to
``/Apps/<app name>/`` and it cannot read or write anything else. Paths in API
calls are relative to that folder, so the default base ``/data`` lands in
``/Apps/<app name>/data/``; with the app named ``election-analyser`` that is
``/Apps/election-analyser/data/<state>/``.

Inputs:
- ``directory``: the run's originals directory (``ORIGINALS_DIR``).
- ``label``: a unique name for this run's manifest (the workflow passes
  ``<run id>-<attempt>``).
- Credentials, from the environment only and never printed:
  ``DROPBOX_APP_KEY``, ``DROPBOX_APP_SECRET``, ``DROPBOX_REFRESH_TOKEN``.
  The app is an App folder app with the ``files.content.write`` scope only.

Output, in Dropbox:
- ``<base>/<state>/<sha256>.<ext>`` and ``<base>/<state>/<sha256>.header.json``,
  uploaded in ``add`` mode without autorename. Nothing in Dropbox is ever
  overwritten. For a body, a file already present under its SHA-256 name has
  that exact content, so a conflict means "already archived". For a header,
  a conflict means the header from the body's first archived retrieval is
  kept and this run's header is not uploaded; this run's retrievals of that
  body are still recorded in the run manifest below.
- ``<base>/manifests/<label>.json``: this run's MANIFEST.json, including
  every retrieval of bodies archived by earlier runs.

Returns 0 when every file is archived or already present; 1 if any upload
failed (the failure is listed, and the workflow step shows red, but snapshot
publication has already run and is not affected). Returns 0 with a notice
when there is nothing to archive.

Limits: single-request uploads are limited to 150 MB per file by Dropbox; the
largest expected source (about 21 MB) is well under it, and a larger file is
reported as a failure rather than truncated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from .originals import MANIFEST

BASE = "/data"  # relative to the App folder: /Apps/<app name>/data
TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"
UPLOAD_URL = "https://content.dropboxapi.com/2/files/upload"
MAX_SINGLE_UPLOAD = 150 * 1024 * 1024

# (url, data, headers) -> (status, body); injectable for tests.
Post = Callable[[str, bytes, dict[str, str]], tuple[int, bytes]]


def _post(url: str, data: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            return reply.status, reply.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def access_token(app_key: str, app_secret: str, refresh_token: str, post: Post = _post) -> str:
    """A short-lived access token from the long-lived refresh token."""
    form = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": app_key,
        "client_secret": app_secret,
    }).encode("ascii")
    status, body = post(TOKEN_URL, form, {"Content-Type": "application/x-www-form-urlencoded"})
    if status != 200:
        # The response can echo request details; report only the status.
        raise RuntimeError(f"Dropbox token request failed with HTTP {status}")
    return json.loads(body)["access_token"]


def upload(token: str, path: str, data: bytes, post: Post = _post) -> str:
    """Upload ``data`` to ``path`` without overwriting. Returns 'archived' or 'already archived'."""
    if len(data) > MAX_SINGLE_UPLOAD:
        raise RuntimeError(f"{path}: {len(data):,} bytes exceeds the single-upload limit")
    argument = json.dumps({"path": path, "mode": "add", "autorename": False, "mute": True})
    status, body = post(UPLOAD_URL, data, {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
        "Dropbox-API-Arg": argument,
    })
    if status == 200:
        return "archived"
    if status == 409 and b"conflict" in body:
        return "already archived"
    summary = body[:200].decode("utf-8", errors="replace")
    raise RuntimeError(f"{path}: HTTP {status}: {summary}")


def archive(directory: Path, label: str, token: str, base: str = BASE,
            post: Post = _post, log: Callable[[str], None] = print) -> int:
    """Upload every original and header under ``directory``, then the run manifest."""
    if not (directory / MANIFEST).exists():
        log("::notice::No downloaded originals to archive.")
        return 0
    failures = 0
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != MANIFEST)
    for local in files:
        relative = local.relative_to(directory).as_posix()
        try:
            outcome = upload(token, f"{base}/{relative}", local.read_bytes(), post)
            log(f"{outcome}: {base}/{relative}")
        except RuntimeError as error:
            failures += 1
            log(f"::error::Dropbox archive failed: {error}")
    try:
        outcome = upload(token, f"{base}/manifests/{label}.json", (directory / MANIFEST).read_bytes(), post)
        log(f"{outcome}: {base}/manifests/{label}.json")
    except RuntimeError as error:
        failures += 1
        log(f"::error::Dropbox archive failed: {error}")
    log(f"Dropbox archive: {len(files)} file(s), {failures} failure(s).")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m election_reporting.watch.dropbox_archive")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--label", required=True, help="unique manifest name for this run")
    parser.add_argument("--base", default=BASE)
    args = parser.parse_args(argv)
    keys = ("DROPBOX_APP_KEY", "DROPBOX_APP_SECRET", "DROPBOX_REFRESH_TOKEN")
    if not all(os.environ.get(key) for key in keys):
        print("::warning::Dropbox archive skipped: DROPBOX_APP_KEY, DROPBOX_APP_SECRET and "
              "DROPBOX_REFRESH_TOKEN are not all configured. The 90-day artifact still holds the files.")
        return 0
    try:
        # _post is looked up here, at call time, so tests can substitute it.
        token = access_token(*(os.environ[key] for key in keys), post=_post)
    except (RuntimeError, KeyError, ValueError) as error:
        print(f"::error::{error}")
        return 1
    return archive(args.directory, args.label, token, args.base, post=_post)


if __name__ == "__main__":
    sys.exit(main())
