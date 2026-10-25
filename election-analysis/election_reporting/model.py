# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:8e606fc0-6856-4226-8736-e7e867f23f4c

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path


class ElectionReportError(RuntimeError):
    """Raised for an actionable input, configuration, or publication error."""


@dataclass(frozen=True)
class ChoiceResult:
    name: str
    votes: int
    order: int


@dataclass(frozen=True)
class ResultStatus:
    """Dated authority evidence, separate from a scheduled certification date."""

    lifecycle: str = "unknown"
    tabulation_stage: str = "unknown"
    authority_url: str = ""
    evidence_as_of: datetime | None = None
    result_version: str = ""
    round_id: str = ""


@dataclass(frozen=True)
class ResultStatusRecord:
    source_id: str
    snapshot_sha256: str
    contest_id: str
    status: ResultStatus


@dataclass(frozen=True)
class ContestResult:
    contest_id: str
    name: str
    order: float
    ballots: int
    reported_ballots: int
    overvotes: int | None
    undervotes: int | None
    choices: tuple[ChoiceResult, ...]
    key: str
    # Counties the contest spans, when the source reports them (the source
    # watcher's normalized CSV does; official exports do not).
    counties: tuple[str, ...] = ()
    result_status: ResultStatus = field(default_factory=ResultStatus)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Snapshot:
    source_id: str
    path: Path
    timestamp: datetime
    sha256: str
    contests: dict[str, ContestResult]
    source_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    path: Path
    jurisdiction: str
    scope: str
    priority: int
    certification_date: date | None = None
    expected_final_ballots: int | None = None
    forecast_reliability: int = 0
    forecast_basis: str = "No remaining-ballot forecast is configured."


@dataclass(frozen=True)
class AreaConfig:
    """A directory of county or precinct breakdown snapshots for one source."""

    source_id: str
    level: str  # "county" or "precinct"
    path: Path


@dataclass(frozen=True)
class RemainingBallots:
    """Ballots a county still has to count, as published (e.g. a ballot status report)."""

    county: str
    ballots: int
    as_of: date
    basis: str


@dataclass(frozen=True)
class ElectionConfig:
    title: str
    sources: tuple[SourceConfig, ...]
    write_in_labels: frozenset[str]
    rule_overrides: dict[str, str] = field(default_factory=dict)
    default_rule: str | None = None
    areas: tuple[AreaConfig, ...] = ()
    remaining_ballots: tuple[RemainingBallots, ...] = ()
    report_mode: str = "descriptive"
    forecast_target: str = "none"
    information_time: datetime | None = None
    election_status: ResultStatus = field(default_factory=ResultStatus)
    result_statuses: tuple[ResultStatusRecord, ...] = ()
    directory: Path | None = None
    audit_contract: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    flip_band: str
    change_probability: float | None
    probability_model: str
    reliability_score: int
    reliability_grade: str
    reliability_rationale: str
    required_share: float | None
    estimated_remaining_votes: int | None
    latest_batch_share: float | None
    latest_batch_votes: int | None
    lead_changed: bool
    rationale: str


@dataclass(frozen=True)
class DecisionAnalysis:
    kind: str
    current_state: str
    current_side: str
    change_side: str
    margin: int
    risk: RiskAssessment
    margin_history: tuple[tuple[str, int], ...]
    event: dict[str, object] = field(default_factory=dict)
    remaining_pool: dict[str, object] = field(default_factory=dict)
    numerical_evidence: dict[str, object] = field(default_factory=dict)
    batch_evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RaceAnalysis:
    contest_id: str
    contest: str
    contest_order: float
    source_id: str
    jurisdiction: str
    snapshot: str
    ballots: int
    reported_ballots: int
    overvotes: int | None
    undervotes: int | None
    valid_votes: int
    rule: str
    controlling: bool
    scope_note: str
    choices: tuple[ChoiceResult, ...]
    decisions: tuple[DecisionAnalysis, ...]
    overall_risk: str
    result_status: ResultStatus = field(default_factory=ResultStatus)
    forecast_eligible: bool = False
    forecast_reason: str = "Lifecycle and tabulation stage have not been established."
    excluded_history: tuple[str, ...] = ()
    canonical_id: str = ""
    data_status: str = "unknown"
    evidence: dict[str, object] = field(default_factory=dict)
    scientific_eligibility: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ElectionAnalysis:
    title: str
    as_of: str
    races: tuple[RaceAnalysis, ...]
    snapshots: tuple[Snapshot, ...]
    areas: object | None = None  # areas.AreaAnalysis
    report_mode: str = "descriptive"
    forecast_target: str = "none"
    information_time: str | None = None
    election_status: ResultStatus = field(default_factory=ResultStatus)
    official_versions: tuple[dict, ...] = ()
    generated_at: str = ""
    source_summary: dict[str, object] = field(default_factory=dict)
    coverage_summary: dict[str, object] = field(default_factory=dict)
