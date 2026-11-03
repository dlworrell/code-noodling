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
  The app is an App folder app with the ``files.content.write`` and
  ``files.metadata.read`` scopes, both confined to the app folder.

Output, in Dropbox:
- ``<base>/<state>/<sha256>.<ext>`` and ``<base>/<state>/<sha256>.header.json``,
  uploaded in ``add`` mode without autorename. Nothing in Dropbox is ever
  overwritten. On a conflict the stored file's Dropbox content hash is
  compared with the local bytes: equal means "already archived (verified)",
  anything else (different bytes, a folder, a failed check) is a failure. A
  header from an earlier run of the same body has different bytes, so it is
  never uploaded over it: when a body was already archived (verified), its
  header is skipped and the first archived header is kept; later retrievals
  are recorded in each run's manifest.
- ``<base>/manifests/<label>.json``: this run's MANIFEST.json, including
  every retrieval of bodies archived by earlier runs.

Status words, kept distinct: *captured* (saved by the keeper on the runner),
*backed up* (in the run's 90-day artifact), *archived* (uploaded to Dropbox,
or already there with a verified content hash). Without credentials the
step prints that the run was NOT archived and exits 0, so snapshot
publication is never blocked; such runs are not replayed to Dropbox later
automatically (a backfill from the artifacts is a separate follow-up).

Returns 0 when every file is archived or verified as already present; 1 if
any upload failed (the failure is listed, and the workflow step shows red, but snapshot
publication has already run and is not affected). Returns 0 with a notice
when there is nothing to archive.

Limits: single-request uploads are limited to 150 MB per file by Dropbox; the
largest expected source (about 21 MB) is well under it, and a larger file is
reported as a failure rather than truncated.
"""

from __future__ import annotations

import argparse
import hashlib
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
METADATA_URL = "https://api.dropboxapi.com/2/files/get_metadata"
MAX_SINGLE_UPLOAD = 150 * 1024 * 1024
BLOCK = 4 * 1024 * 1024  # Dropbox content_hash block size
TRANSPORT_ERROR = 0  # pseudo-status for a request that never got an HTTP answer

# (url, data, headers) -> (status, body); injectable for tests.
Post = Callable[[str, bytes, dict[str, str]], tuple[int, bytes]]


def _post(url: str, data: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
    """POST and return (status, body). Never raises for network trouble.

    An HTTP error returns its status and body. A request that gets no HTTP
    answer at all (DNS failure, refused connection, timeout, TLS error)
    returns ``TRANSPORT_ERROR`` with the reason, so the caller reports that
    one file as failed and carries on with the rest.
    """
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            return reply.status, reply.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        return TRANSPORT_ERROR, f"transport error: {reason}".encode("utf-8", errors="replace")


def content_hash(data: bytes) -> str:
    """Dropbox's ``content_hash``: SHA-256 over the SHA-256 of each 4 MiB block."""
    blocks = b"".join(hashlib.sha256(data[i:i + BLOCK]).digest() for i in range(0, len(data), BLOCK))
    return hashlib.sha256(blocks).hexdigest()


def remote_metadata(token: str, path: str, post: Post) -> dict | None:
    """Dropbox metadata for ``path``, or None if it cannot be read."""
    status, body = post(METADATA_URL, json.dumps({"path": path}).encode("utf-8"), {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    if status != 200:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def remote_matches(token: str, path: str, data: bytes, post: Post) -> bool:
    """True only if ``path`` in Dropbox is a file whose content hash equals ``data``'s.

    Needs the ``files.metadata.read`` scope (inside the app folder only).
    """
    status, body = post(METADATA_URL, json.dumps({"path": path}).encode("utf-8"), {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    if status != 200:
        return False
    try:
        metadata = json.loads(body)
    except ValueError:
        return False
    return metadata.get(".tag") == "file" and metadata.get("content_hash") == content_hash(data)


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


def upload(token: str, path: str, data: bytes, post: Post = _post, *, keep_existing_file: bool = False) -> str:
    """Upload ``data`` to ``path`` without overwriting.

    Returns 'archived', or 'already archived (verified)' when a file is
    already there and Dropbox reports the same content hash. A conflict that
    cannot be verified (different bytes, a folder in the way, or the check
    itself failing) raises: a SHA-256 in the file name is not proof of what
    is stored under it.

    ``keep_existing_file`` is for the header of a body that an earlier run
    already archived: a header that already exists there as a file is that
    first archived header and is kept ('first header kept'); a missing header
    is uploaded; a folder in the way or an unreadable check still raises.
    """
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
        if remote_matches(token, path, data, post):
            return "already archived (verified)"
        if keep_existing_file and (remote_metadata(token, path, post) or {}).get(".tag") == "file":
            return "first header kept"
        raise RuntimeError(f"{path}: already exists in Dropbox with different or unverifiable content")
    summary = body[:200].decode("utf-8", errors="replace")
    raise RuntimeError(f"{path}: HTTP {status}: {summary}")


def archive(directory: Path, label: str, token: str, base: str = BASE,
            post: Post = _post, log: Callable[[str], None] = print) -> int:
    """Upload every original, then its header, then the run manifest.

    Bodies go first. Every header is then uploaded. When its body was
    already archived (verified) by an earlier run, a header already present
    as a file is the first archived one and is kept; a header that is
    missing (for example after an interrupted earlier run) is uploaded now,
    so a retry repairs it. Each file is attempted even if an earlier one
    failed, and the run counts as archived only if every body, header and
    the manifest succeeded.
    """
    if not (directory / MANIFEST).exists():
        log("::notice::No downloaded originals to archive.")
        return 0
    failures = 0
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != MANIFEST)
    headers = [p for p in files if p.name.endswith(".header.json")]
    bodies = [p for p in files if not p.name.endswith(".header.json")]
    previously_archived: set[str] = set()  # "<state>/<sha256>" of bodies archived by an earlier run
    for local in bodies + headers:
        relative = local.relative_to(directory).as_posix()
        key = f"{local.parent.name}/{local.name.split('.', 1)[0]}"
        try:
            outcome = upload(token, f"{base}/{relative}", local.read_bytes(), post,
                             keep_existing_file=local in headers and key in previously_archived)
            log(f"{outcome}: {base}/{relative}")
            if local in bodies and outcome.startswith("already archived"):
                previously_archived.add(key)
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
        print("::warning::NOT archived to Dropbox: DROPBOX_APP_KEY, DROPBOX_APP_SECRET and "
              "DROPBOX_REFRESH_TOKEN are not all configured. This run's originals are captured and backed "
              "up only in its 90-day artifact; they are not replayed to Dropbox later automatically.")
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
