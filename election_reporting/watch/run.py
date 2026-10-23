"""Plan an election from ``watch.toml`` and fetch new result snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from ..model import ElectionReportError
from . import alaska, csvformat, maine, sources
from .config import WatchConfig
from .http import Fetch
from .schedule import (
    NO_CERTIFICATION_DATE,
    Window,
    certification_date,
    election_kind,
    is_due,
    watch_window,
)
from .store import SOURCES_FILE, ElectionStore, SnapshotRecord, gzip_bytes, sha256

STATE_PRIORITY = 100
COUNTY_PRIORITY = 10


@dataclass(frozen=True)
class Plan:
    election: sources.ElectionRef
    kind: str
    places: tuple[sources.Place, ...]
    directory: Path
    window: Window
    state: str = "Washington"
    state_id: str = "washington"


def plan(config: WatchConfig, fetch: Fetch, now: datetime) -> Plan:
    """Resolve the selected places and election, and the checking window."""
    if config.state_id == "alaska":
        # election-data/elections/alaska -> election-data
        crosswalk = alaska.load_crosswalk(config.elections_root.parents[1])
        places = alaska.resolve_places(crosswalk, config.counties, config.all_counties)
        elections = alaska.list_elections(now.astimezone(config.timezone).date())
    elif config.state_id == "maine":
        places = maine.resolve_places(config.counties, config.all_counties)
        elections = maine.list_elections(now.astimezone(config.timezone).date())
    else:
        jurisdiction = sources.state_jurisdiction(fetch)
        places = sources.resolve_places(
            jurisdiction,
            config.state,
            config.counties,
            config.prefer_county_files,
            config.all_counties,
        )
        elections = sources.list_elections(jurisdiction)
    if not elections:
        raise ElectionReportError("the results site lists no elections")
    chosen = None
    for election in elections:
        kind = election_kind(election.name)
        window = watch_window(config, election.day, kind, has_state=True)
        if config.election == "next":
            if window.end >= now:
                chosen = (election, kind, window)
                break
        elif election.day.isoformat() == config.election:
            chosen = (election, kind, window)
            break
    if chosen is None:
        wanted = "an upcoming election" if config.election == "next" else config.election
        listed = ", ".join(e.day.isoformat() for e in elections)
        raise ElectionReportError(f"no election found for {wanted}; listed: {listed}")
    election, kind, window = chosen
    directory = config.elections_root / election.day.isoformat()
    return Plan(election, kind, tuple(places), directory, window, config.state, config.state_id)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def source_block(place: sources.Place, election_plan: Plan) -> str:
    scope = place.scope
    certified = certification_date(
        election_plan.election.day, election_plan.kind, scope, election_plan.state_id
    )
    jurisdiction = f"{place.name} State" if scope == "state" else f"{place.name}, {election_plan.state}"
    if election_plan.state_id in NO_CERTIFICATION_DATE:
        certification = "# certification_date: not published; add it when known\n"
    else:
        certification = f"certification_date = {certified.isoformat()}\n"
    return (
        "[[sources]]\n"
        f"id = {_toml_string(place.place_id)}\n"
        f"path = {_toml_string('input/' + place.place_id)}\n"
        f"jurisdiction = {_toml_string(jurisdiction)}\n"
        f'scope = "{scope}"\n'
        f"priority = {STATE_PRIORITY if scope == 'state' else COUNTY_PRIORITY}\n"
        + certification +
        "# expected_final_ballots = 0       # add a turnout forecast to enable\n"
        "# forecast_reliability = 50        # remaining-ballot estimates\n"
        '# forecast_basis = "..."\n'
    )


def write_election_config(election_plan: Plan) -> list[str]:
    """Create ``election.toml``, or append sources for newly selected places.

    Existing content is never rewritten, so hand-added forecasts and contest
    rules are kept. Returns the place ids that were added.
    """
    path = election_plan.directory / "election.toml"
    election = election_plan.election
    if path.exists():
        import tomllib

        with path.open("rb") as stream:
            existing = {row.get("id") for row in tomllib.load(stream).get("sources", [])}
        missing = [p for p in election_plan.places if p.place_id not in existing]
        if missing:
            with path.open("a", encoding="utf-8") as stream:
                for place in missing:
                    stream.write("\n" + source_block(place, election_plan))
        return [p.place_id for p in missing]
    title = election.name or f"Election of {election.day.isoformat()}"
    header = (
        "# Generated by `python -m election_reporting.watch plan` from\n"
        "# election-data/watch.toml. Safe to edit: later runs only append sources\n"
        "# for newly selected places.\n"
        "schema_version = 1\n\n"
        "[election]\n"
        f"title = {_toml_string(f'{title} ({election.day:%B} {election.day.day}, {election.day.year})')}\n\n"
        "[analysis]\n"
        'write_in_labels = ["write-in", "write in"]\n'
    )
    if election_plan.state_id == "alaska":
        header += (
            "# Alaska: primaries advance the top four; general elections are ranked choice.\n"
            f'default_rule = "{alaska.DEFAULT_RULES[election_plan.kind]}"\n'
        )
    elif election_plan.state_id == "maine":
        header += (
            "# Maine: plurality, except contests posted as ranked choice, which the\n"
            "# watcher adds below as [[contest_rules]] with rule \"rcv\".\n"
            'default_rule = "winner"\n'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = "".join("\n" + source_block(place, election_plan) for place in election_plan.places)
    path.write_text(header + blocks, encoding="utf-8")
    return [p.place_id for p in election_plan.places]


@dataclass
class FetchResult:
    due: bool
    added: list[SnapshotRecord] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _stamp(moment: datetime, config: WatchConfig) -> str:
    return moment.astimezone(config.timezone).strftime("%Y-%m-%d-%H%M")


def _free_stamp(store: ElectionStore, place: str, stamp: str) -> str:
    moment = datetime.strptime(stamp, "%Y-%m-%d-%H%M")
    while store.has_snapshot(place, moment.strftime("%Y-%m-%d-%H%M")):
        moment += timedelta(minutes=1)
    return moment.strftime("%Y-%m-%d-%H%M")


def _has_votes(csv_text: str) -> bool:
    import csv
    import io

    reader = csv.DictReader(io.StringIO(csv_text))
    return any(int(row["Votes"] or 0) > 0 for row in reader)


def _record(
    store: ElectionStore,
    config: WatchConfig,
    place: sources.Place,
    *,
    kind: str,
    moment: datetime,
    retrieved: datetime,
    source_url: str,
    source_timestamp: str,
    input_bytes: bytes,
    raw_bytes: bytes | None,
    raw_ext: str,
    result: FetchResult,
    note: str = "",
) -> None:
    digest = sha256(input_bytes)
    if any(r.place == place.place_id and r.input_sha256 == digest for r in store.records):
        result.skipped.append(f"{place.place_id}: unchanged")
        return
    stamp = _free_stamp(store, place.place_id, _stamp(moment, config))
    input_path = f"input/{place.place_id}/{stamp}.csv"
    raw_path = f"raw/{place.place_id}/{stamp}.{raw_ext}" if raw_bytes is not None else ""
    store.write_file(input_path, input_bytes)
    raw_digest = store.write_file(raw_path, raw_bytes) if raw_bytes is not None else ""
    record = SnapshotRecord(
        place=place.place_id,
        snapshot=stamp,
        kind=kind,
        source_url=source_url,
        retrieved_at=retrieved.astimezone(timezone.utc).isoformat(timespec="seconds"),
        source_timestamp=source_timestamp,
        input=input_path,
        input_sha256=digest,
        raw=raw_path,
        raw_sha256=raw_digest,
        note=note,
    )
    store.add(record)
    result.added.append(record)


def _fetch_state_xlsx(store, config, place, plan_, fetch, now, result, url) -> None:
    response = fetch(url)
    moment = response.last_modified or now
    text = csvformat.from_summary_xlsx(
        response.body, label=url.rsplit("/", 1)[-1], jurisdiction=place.name
    )
    if not _has_votes(text):
        result.skipped.append(f"{place.place_id}: no votes reported yet")
        return
    _record(
        store, config, place,
        kind="state-xlsx", moment=moment, retrieved=now, source_url=url,
        source_timestamp=response.headers.get("last-modified", ""),
        input_bytes=text.encode("utf-8"), raw_bytes=response.body, raw_ext="xlsx",
        result=result,
    )


def _fetch_votewa_json(store, config, place, plan_, fetch, now, result) -> None:
    data = sources.results_data(fetch, place, plan_.election)
    as_of = str((data.get("election") or {}).get("asOf") or "")
    try:
        moment = datetime.fromisoformat(as_of.replace("Z", "+00:00")) if as_of else now
    except ValueError:
        moment = now
    text = csvformat.from_votewa_data(
        data,
        jurisdiction=place.name,
        county=place.name if place.scope == "county" else "",
    )
    if not _has_votes(text):
        result.skipped.append(f"{place.place_id}: no votes reported yet")
        return
    url = f"{sources.VOTEWA_API}/elections/{place.votewa}/{plan_.election.public_id}/data"
    _record(
        store, config, place,
        kind="votewa-json", moment=moment, retrieved=now, source_url=url,
        source_timestamp=as_of, input_bytes=text.encode("utf-8"),
        raw_bytes=gzip_bytes(csvformat.trim_votewa_data(data)), raw_ext="json.gz",
        result=result,
        note="Ballots are votes plus any reported overvotes and undervotes; "
        "the results API does not report ballots cast.",
    )


def _fetch_king_county(store, config, place, plan_, fetch, now, result) -> None:
    page = sources.king_county_page(fetch, plan_.election.day)
    if page is None:
        result.skipped.append(f"{place.place_id}: no county results page yet")
        return
    files = [f for f in sources.king_county_files(fetch, page) if f.day >= plan_.election.day]
    if not files:
        result.skipped.append(f"{place.place_id}: no county result files yet")
        return
    seen_urls = {r.source_url.split("?", 1)[0] for r in store.records if r.place == place.place_id}
    for item in files:
        if item.url.split("?", 1)[0] in seen_urls:
            continue
        response = fetch(item.url)
        local = (response.last_modified or now).astimezone(config.timezone)
        if local.date() != item.day:
            local = datetime.combine(item.day, datetime.max.time().replace(second=0, microsecond=0), config.timezone)
        _record(
            store, config, place,
            kind="county-csv", moment=local, retrieved=now, source_url=item.url,
            source_timestamp=response.headers.get("last-modified", ""),
            input_bytes=response.body, raw_bytes=None, raw_ext="",
            result=result,
            note="Official King County Elections export, stored unmodified."
            + (" Final certified results." if item.final else ""),
        )


def _fetch_alaska(store, config, plan_, fetch, now, result) -> None:
    """One statewide precinct file, converted to statewide and borough inputs."""
    url = alaska.results_file_url(fetch, plan_.election)
    if url is None:
        result.skipped.append("alaska: no precinct results file posted yet")
        return
    response = fetch(url)
    modified = response.last_modified
    # The Division reuses one results path per year, so before new results are
    # posted the link can still serve an earlier election's file (or the
    # election day's zero report, posted before polls close).
    if modified is not None and modified < plan_.window.start:
        result.skipped.append(
            f"alaska: results file last modified {modified:%Y-%m-%d %H:%M %Z}, "
            "before polls closed; no results for this election yet"
        )
        return
    crosswalk = alaska.load_crosswalk(config.elections_root.parents[1])
    outputs = alaska.convert(response.body, crosswalk, list(plan_.places))
    if not _has_votes(outputs.get("alaska", "ContestOrder,Votes\n")):
        result.skipped.append("alaska: no votes reported yet")
        return
    moment = modified or now
    raw = gzip_bytes(response.body)
    for place in plan_.places:
        text = outputs.get(place.place_id)
        if text is None:
            result.skipped.append(f"{place.place_id}: no precincts in the results file")
            continue
        state = place.scope == "state"
        _record(
            store, config, place,
            kind="alaska-precinct-csv", moment=moment, retrieved=now, source_url=url,
            source_timestamp=response.headers.get("last-modified", ""),
            input_bytes=text.encode("utf-8"),
            raw_bytes=raw if state else None, raw_ext="csv.gz" if state else "",
            result=result,
            note="" if state else alaska.DISTRICT_LEVEL_NOTE,
        )


def append_contest_rules(directory: Path, rules: dict[str, str]) -> list[str]:
    """Append [[contest_rules]] for contests that election.toml does not name yet."""
    import tomllib

    path = directory / "election.toml"
    with path.open("rb") as stream:
        named = {row.get("contest_id") for row in tomllib.load(stream).get("contest_rules", [])}
    missing = [contest_id for contest_id in sorted(rules) if contest_id not in named]
    if missing:
        with path.open("a", encoding="utf-8") as stream:
            for contest_id in missing:
                stream.write(
                    f"\n[[contest_rules]]\ncontest_id = {_toml_string(contest_id)}\n"
                    f'rule = "{rules[contest_id]}"\n'
                )
    return missing


def _zip_workbooks(files: list[tuple[str, str, bytes]]) -> bytes:
    """A reproducible ZIP of the source workbooks plus a SOURCES.txt manifest."""
    import io
    import zipfile

    buffer = io.BytesIO()
    manifest = ["url\tlast-modified\tsha256"]
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        def entry(name: str) -> zipfile.ZipInfo:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            return info

        for index, (url, modified, body) in enumerate(files, start=1):
            name = f"{index:02d}-" + url.rsplit("/", 1)[-1].replace("%20", "_")
            archive.writestr(entry(name), body)
            manifest.append(f"{url}\t{modified}\t{sha256(body)}")
        archive.writestr(entry("SOURCES.txt"), "\n".join(manifest) + "\n")
    return buffer.getvalue()


def _fetch_maine(store, config, plan_, fetch, now, result) -> None:
    """Every office workbook posted for the election, combined into one snapshot."""
    links = maine.result_links(fetch, plan_.election)
    if not links:
        result.skipped.append("maine: no results posted for this election yet")
        return
    contests: list[maine.Contest] = []
    files: list[tuple[str, str, bytes]] = []
    newest = None
    for link in links:
        try:
            response = fetch(link.url)
            found, problems = maine.parse_workbook(
                response.body, label=link.label, rcv=link.rcv, url=link.url
            )
        except ElectionReportError as error:
            result.errors.append(f"maine: {link.label}: {error}")
            continue
        result.errors.extend(f"maine: {problem}" for problem in problems)
        contests.extend(found)
        files.append((link.url, response.headers.get("last-modified", ""), response.body))
        if response.last_modified and (newest is None or response.last_modified > newest):
            newest = response.last_modified
    seen: set[str] = set()
    unique = []
    for contest in contests:
        if contest.contest_id in seen:
            result.errors.append(f"maine: contest {contest.name!r} is in more than one workbook; first kept")
            continue
        seen.add(contest.contest_id)
        unique.append(contest)
    outputs = maine.convert(unique, list(plan_.places))
    if "maine" not in outputs:
        result.skipped.append("maine: no readable results yet")
        return
    added = append_contest_rules(plan_.directory, {c.contest_id: "rcv" for c in unique if c.rcv})
    if added:
        result.skipped.append(f"maine: election.toml: ranked-choice rules added for {', '.join(added)}")
    raw = _zip_workbooks(files)
    for place in plan_.places:
        text = outputs.get(place.place_id)
        if text is None:
            result.skipped.append(f"{place.place_id}: no results for this county yet")
            continue
        state = place.scope == "state"
        _record(
            store, config, place,
            kind="maine-office-workbooks", moment=newest or now, retrieved=now,
            source_url=maine.RESULTS_PAGE, source_timestamp=newest.isoformat() if newest else "",
            input_bytes=text.encode("utf-8"),
            raw_bytes=raw if state else None, raw_ext="zip" if state else "",
            result=result,
            note=maine.STATE_NOTE if state else maine.NOTE,
        )


def fetch_snapshots(
    config: WatchConfig,
    fetch: Fetch,
    now: datetime,
    *,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> FetchResult:
    election_plan = plan(config, fetch, now)
    window = election_plan.window
    due = force or is_due(config, window, now)
    result = FetchResult(due=due)
    log(
        f"Election {election_plan.election.day} ({election_plan.election.name}); "
        f"window {window.start:%Y-%m-%d %H:%M} to {window.end:%Y-%m-%d %H:%M} {config.timezone.key}"
    )
    if not due:
        log("Not due; nothing fetched.")
        return result
    added = write_election_config(election_plan)
    if added:
        log(f"election.toml: added sources {', '.join(added)}")
    store = ElectionStore(election_plan.directory)
    store.election = {
        "date": election_plan.election.day.isoformat(),
        "name": election_plan.election.name,
        "kind": election_plan.kind,
        "places": [p.place_id for p in election_plan.places],
    }
    adapters = {"alaska": _fetch_alaska, "maine": _fetch_maine}
    if config.state_id in adapters:
        try:
            adapters[config.state_id](store, config, election_plan, fetch, now, result)
        except ElectionReportError as error:
            result.errors.append(f"{config.state_id}: {error}")
    for place in election_plan.places if config.state_id not in adapters else ():
        try:
            if place.county_files:
                _fetch_king_county(store, config, place, election_plan, fetch, now, result)
            elif place.scope == "state":
                metadata = sources.election_metadata(fetch, place, election_plan.election)
                url = sources.report_url(metadata)
                if url and "votewa-json" not in store.kinds(place.place_id):
                    _fetch_state_xlsx(store, config, place, election_plan, fetch, now, result, url)
                else:
                    _fetch_votewa_json(store, config, place, election_plan, fetch, now, result)
            else:
                _fetch_votewa_json(store, config, place, election_plan, fetch, now, result)
        except ElectionReportError as error:
            result.errors.append(f"{place.place_id}: {error}")
    if result.added or store.path.exists():
        store.save()
    for record in result.added:
        log(f"new snapshot {record.place} {record.snapshot} ({record.kind})")
    for line in result.skipped:
        log(f"skipped {line}")
    for line in result.errors:
        log(f"error {line}")
    return result


def carry_forward(source_root: Path, target_root: Path, log: Callable[[str], None] = print) -> int:
    """Copy snapshots recorded under ``source_root`` but missing from ``target_root``.

    Used by the scheduled workflow: snapshots waiting in an unmerged automation
    pull request are copied onto a fresh checkout of the default branch before
    the next fetch, because report files that the state overwrites cannot be
    downloaded again. Only snapshot files and their records move; everything
    else on the default branch (for example a hand-edited ``election.toml``)
    is left as it is.
    """
    copied = 0
    if not source_root.is_dir():
        return 0
    # Election directories are <state>/<date>/; find them by their records.
    for source_dir in sorted(path.parent for path in source_root.rglob(SOURCES_FILE)):
        relative = source_dir.relative_to(source_root)
        if ".." in relative.parts:
            continue
        carried = ElectionStore(source_dir)
        if not carried.records:
            continue
        target = ElectionStore(target_root / relative)
        before = copied
        if not target.election:
            target.election = carried.election
        for record in carried.records:
            if target.has_snapshot(record.place, record.snapshot):
                continue
            for stored, digest in ((record.input, record.input_sha256), (record.raw, record.raw_sha256)):
                if not stored:
                    continue
                data = (source_dir / stored).read_bytes()
                if sha256(data) != digest:
                    raise ElectionReportError(f"{source_dir / stored} does not match its recorded SHA-256")
                target.write_file(stored, data)
            target.add(record)
            copied += 1
            log(f"carried forward {relative.as_posix()} {record.place} {record.snapshot}")
        if copied > before:
            target.save()
    return copied
