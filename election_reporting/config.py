from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path

from .model import ElectionConfig, ElectionReportError, SourceConfig


SUPPORTED_RULES = {"winner", "top_two", "wa_judicial", "automatic", "ignore"}
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


def load_config(path: Path) -> ElectionConfig:
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ElectionReportError(f"configuration file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ElectionReportError(f"invalid TOML in {path}: {exc}") from exc

    if raw.get("schema_version") != 1:
        raise ElectionReportError("schema_version must be 1")

    election = raw.get("election")
    if not isinstance(election, dict):
        raise ElectionReportError("[election] is required")
    title = _required_string(election, "title", "election")

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
                expected_final_ballots=_optional_positive_int(
                    row.get("expected_final_ballots"),
                    f"{location}.expected_final_ballots",
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

    return ElectionConfig(
        title=title,
        sources=tuple(sources),
        write_in_labels=write_in_labels,
        rule_overrides=overrides,
    )
