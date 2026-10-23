"""Minimal HTTPS GET client limited to official election-results hosts."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from datetime import datetime
from typing import Callable

from ..model import ElectionReportError

ALLOWED_HOSTS = frozenset(
    {
        "results.votewa.gov",
        "kingcounty.gov",
        "cdn.kingcounty.gov",
    }
)
USER_AGENT = (
    "code-noodling-election-watcher/1 (+https://github.com/dlworrell/code-noodling)"
)
MAX_BYTES = 25 * 1024 * 1024
TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Response:
    url: str
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def last_modified(self) -> datetime | None:
        value = self.headers.get("last-modified")
        if not value:
            return None
        try:
            return parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None


Fetch = Callable[[str], Response]


def check_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise ElectionReportError(f"refusing to fetch outside the official result hosts: {url}")
    return url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_NoRedirect)


def fetch(url: str) -> Response:
    """GET ``url`` from an allowed host, refusing oversized bodies."""
    check_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with _OPENER.open(request, timeout=TIMEOUT_SECONDS) as reply:
            body = reply.read(MAX_BYTES + 1)
            headers = {key.lower(): value for key, value in reply.headers.items()}
            final_url = reply.geturl()
    except urllib.error.HTTPError as exc:
        raise ElectionReportError(f"HTTP {exc.code} from {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ElectionReportError(f"could not fetch {url}: {exc}") from exc
    if len(body) > MAX_BYTES:
        raise ElectionReportError(f"{url} exceeds the {MAX_BYTES:,}-byte download limit")
    return Response(url=check_url(final_url), body=body, headers=headers)
