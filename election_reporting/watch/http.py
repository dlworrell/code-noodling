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
from .certs import context_for

ALLOWED_HOSTS = frozenset(
    {
        "results.votewa.gov",
        "kingcounty.gov",
        "cdn.kingcounty.gov",
        "www.elections.alaska.gov",
        "www.maine.gov",
    }
)
USER_AGENT = (
    "code-noodling-election-watcher/1 (+https://github.com/dlworrell/code-noodling)"
)
# www.maine.gov's firewall (AWS CloudFront) refuses every non-browser client,
# honestly identified ones included, so that host alone gets a browser
# User-Agent. Maine snapshots record this in SOURCES.json.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.6 Safari/605.1.15"
)
USER_AGENT_OVERRIDES = {"www.maine.gov": BROWSER_USER_AGENT}
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


_OPENERS: dict[str, urllib.request.OpenerDirector] = {}


def _opener(host: str) -> urllib.request.OpenerDirector:
    """One opener per host, so a pinned intermediate applies to its host only."""
    if host not in _OPENERS:
        handlers: list[urllib.request.BaseHandler] = [_NoRedirect]
        context = context_for(host)
        if context is not None:
            handlers.append(urllib.request.HTTPSHandler(context=context))
        _OPENERS[host] = urllib.request.build_opener(*handlers)
    return _OPENERS[host]


def fetch(url: str) -> Response:
    """GET ``url`` from an allowed host, refusing oversized bodies."""
    check_url(url)
    host = urllib.parse.urlsplit(url).hostname or ""
    agent = USER_AGENT_OVERRIDES.get(host, USER_AGENT)
    request = urllib.request.Request(url, headers={"User-Agent": agent})
    try:
        with _opener(host).open(request, timeout=TIMEOUT_SECONDS) as reply:
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
