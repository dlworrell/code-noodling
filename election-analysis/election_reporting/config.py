# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:e828ccbe-b036-442b-8472-4dfd1a6ab4c0

from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path

from .model import AreaConfig, ElectionConfig, ElectionReportError, RemainingBallots, SourceConfig
from .stages import aware_time, parse_status, parse_status_records


SUPPORTED_RULES = {
    "winner",
    "top_two",
    "top_four",
    "rcv",
    "wa_judicial",
    "automatic",
    "ignore",
}
SUPPORTED_SCOPES = {"county", "state", "district", "local", "other"}


def _required_string(table: dict[str, object], key: str, location: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ElectionReportError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _optional_date(value: object, location: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ElectionReportError(f"{location} must be YYYY-MM-DD") from exc
    raise ElectionReportError(f"{location} must be an ISO date")


def _optional_positive_int(value: object, location: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ElectionReportError(f"{location} must be a positive integer")
    return value


def _score(value: object, location: str, default: int) -> int:
    if value is None:
        return default
    if type(value) is not int or not 0 <= value <= 100:
        raise ElectionReportError(f"{location} must be an integer from 0 through 100")
    return value


def _optional_string(value: object, location: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ElectionReportError(f"{location} must be a non-empty string")
    return value.strip()


AREA_LEVELS = ("county", "precinct")


def _areas(raw: dict, base: Path, source_ids: set[str]) -> tuple[AreaConfig, ...]:
    """[[areas]]: county and precinct breakdown snapshots, kept apart from inputs."""
    rows = raw.get("areas", [])
    if not isinstance(rows, list):
        raise ElectionReportError("[[areas]] must be an array of tables")
    areas = []
    seen = set()
    for index, row in enumerate(rows, start=1):
        location = f"areas[{index}]"
        if not isinstance(row, dict):
            raise ElectionReportError(f"{location} must be a TOML table")
        source_id = _required_string(row, "source", location)
        if source_id not in source_ids:
            raise ElectionReportError(f"{location}.source {source_id!r} is not a configured source")
        level = _required_string(row, "level", location).lower()
        if level not in AREA_LEVELS:
            raise ElectionReportError(f"{location}.level must be one of: {', '.join(AREA_LEVELS)}")
        if level in seen:
            raise ElectionReportError(f"{location}: only one [[areas]] table per level")
        seen.add(level)
        relative = Path(_required_string(row, "path", location))
        if relative.is_absolute() or ".." in relative.parts:
            raise ElectionReportError(f"{location}.path must stay inside the election directory")
        areas.append(AreaConfig(source_id=source_id, level=level, path=base / relative))
    return tuple(areas)


def _remaining_ballots(raw: dict) -> tuple[RemainingBallots, ...]:
    """[[remaining_ballots]]: per-county ballots still to count, as published."""
    rows = raw.get("remaining_ballots", [])
    if not isinstance(rows, list):
        raise ElectionReportError("[[remaining_ballots]] must be an array of tables")
    counts = []
    for index, row in enumerate(rows, start=1):
        location = f"remaining_ballots[{index}]"
        if not isinstance(row, dict):
            raise ElectionReportError(f"{location} must be a TOML table")
        ballots = row.get("ballots")
        if type(ballots) is not int or ballots < 0:
            raise ElectionReportError(f"{location}.ballots must be a non-negative whole number")
        as_of = _optional_date(row.get("as_of"), f"{location}.as_of")
        if as_of is None:
            raise ElectionReportError(f"{location}.as_of is required")
        counts.append(
            RemainingBallots(
                county=_required_string(row, "county", location),
                ballots=ballots,
                as_of=as_of,
                basis=_required_string(row, "basis", location),
            )
        )
    return tuple(counts)


def load_config(path: Path) -> ElectionConfig:
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ElectionReportError(f"configuration file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ElectionReportError(f"invalid TOML in {path}: {exc}") from exc

    schema = raw.get("schema_version")
    if type(schema) is not int or schema not in {1, 2}:
        raise ElectionReportError("schema_version must be 1 or 2")

    election = raw.get("election")
    if not isinstance(election, dict):
        raise ElectionReportError("[election] is required")
    title = _required_string(election, "title", "election")
    if schema == 1 and (raw.get("result_status") or any(
        key in election for key in ("status", "report_mode", "forecast_target", "information_time")
    )):
        raise ElectionReportError("lifecycle/stage fields require schema_version = 2")
    report_mode = election.get("report_mode", "descriptive")
    forecast_target = election.get("forecast_target", "none")
    if report_mode not in ("descriptive", "historical", "current"):
        raise ElectionReportError("election.report_mode must be descriptive, historical or current")
    if forecast_target not in ("none", "outstanding_ballots", "recount", "litigation"):
        raise ElectionReportError("election.forecast_target is unsupported")
    information_time = aware_time(election.get("information_time"), "election.information_time")
    election_status = parse_status(election.get("status", {}), "election.status")

    source_rows = raw.get("sources")
    if not isinstance(source_rows, list) or not source_rows:
        raise ElectionReportError("at least one [[sources]] table is required")

    base = path.parent
    sources: list[SourceConfig] = []
    source_ids: set[str] = set()
    for index, row in enumerate(source_rows, start=1):
        location = f"sources[{index}]"
        if not isinstance(row, dict):
            raise ElectionReportError(f"{location} must be a TOML table")
        source_id = _required_string(row, "id", location)
        if source_id in source_ids:
            raise ElectionReportError(f"duplicate source id: {source_id}")
        source_ids.add(source_id)
        relative_path = Path(_required_string(row, "path", location))
        scope = str(row.get("scope", "other")).strip().lower()
        if scope not in SUPPORTED_SCOPES:
            raise ElectionReportError(
                f"{location}.scope must be one of: {', '.join(sorted(SUPPORTED_SCOPES))}"
            )
        priority = row.get("priority", 0)
        if type(priority) is not int:
            raise ElectionReportError(f"{location}.priority must be an integer")
        expected_final_ballots = _optional_positive_int(
            row.get("expected_final_ballots"),
            f"{location}.expected_final_ballots",
        )
        default_forecast_reliability = 50 if expected_final_ballots else 0
        sources.append(
            SourceConfig(
                source_id=source_id,
                path=base / relative_path,
                jurisdiction=_required_string(row, "jurisdiction", location),
                scope=scope,
                priority=priority,
                certification_date=_optional_date(
                    row.get("certification_date"),
                    f"{location}.certification_date",
                ),
                expected_final_ballots=expected_final_ballots,
                forecast_reliability=_score(
                    row.get("forecast_reliability"),
                    f"{location}.forecast_reliability",
                    default_forecast_reliability,
                ),
                forecast_basis=_optional_string(
                    row.get("forecast_basis"),
                    f"{location}.forecast_basis",
                    "Configured expected final ballot count."
                    if expected_final_ballots
                    else "No remaining-ballot forecast is configured.",
                ),
            )
        )

    analysis = raw.get("analysis", {})
    if not isinstance(analysis, dict):
        raise ElectionReportError("[analysis] must be a TOML table")
    raw_labels = analysis.get("write_in_labels", ["write-in", "write in"])
    if not isinstance(raw_labels, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_labels
    ):
        raise ElectionReportError("analysis.write_in_labels must be a string array")
    write_in_labels = frozenset(item.strip().casefold() for item in raw_labels)
    # Rule for candidate contests that no [[contest_rules]] entry names. Unset
    # keeps the Washington inference (top-two primaries, judicial majorities).
    default_rule = analysis.get("default_rule")
    if default_rule is not None:
        if not isinstance(default_rule, str) or default_rule.strip().lower() not in SUPPORTED_RULES:
            raise ElectionReportError(
                f"analysis.default_rule must be one of: {', '.join(sorted(SUPPORTED_RULES))}"
            )
        default_rule = default_rule.strip().lower()

    overrides: dict[str, str] = {}
    override_rows = raw.get("contest_rules", [])
    if not isinstance(override_rows, list):
        raise ElectionReportError("[[contest_rules]] must be an array of tables")
    for index, row in enumerate(override_rows, start=1):
        location = f"contest_rules[{index}]"
        if not isinstance(row, dict):
            raise ElectionReportError(f"{location} must be a TOML table")
        contest_id = _required_string(row, "contest_id", location)
        rule = _required_string(row, "rule", location).lower()
        if rule not in SUPPORTED_RULES:
            raise ElectionReportError(
                f"{location}.rule must be one of: {', '.join(sorted(SUPPORTED_RULES))}"
            )
        source_id = str(row.get("source_id", "*")).strip() or "*"
        overrides[f"{source_id}:{contest_id}"] = rule

    areas = _areas(raw, base, {source.source_id for source in sources})
    remaining = _remaining_ballots(raw)

    return ElectionConfig(
        title=title,
        sources=tuple(sources),
        areas=areas,
        remaining_ballots=remaining,
        write_in_labels=write_in_labels,
        rule_overrides=overrides,
        default_rule=default_rule,
        report_mode=report_mode,
        forecast_target=forecast_target,
        information_time=information_time,
        election_status=election_status,
        result_statuses=parse_status_records(raw.get("result_status", []), source_ids),
    )
