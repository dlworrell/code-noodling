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
class ContestResult:
    contest_id: str
    name: str
    order: float
    ballots: int
    choices: tuple[ChoiceResult, ...]
    key: str


@dataclass(frozen=True)
class Snapshot:
    source_id: str
    path: Path
    timestamp: datetime
    sha256: str
    contests: dict[str, ContestResult]


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    path: Path
    jurisdiction: str
    scope: str
    priority: int
    certification_date: date | None = None
    expected_final_ballots: int | None = None


@dataclass(frozen=True)
class ElectionConfig:
    title: str
    sources: tuple[SourceConfig, ...]
    write_in_labels: frozenset[str]
    rule_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    flip_band: str
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


@dataclass(frozen=True)
class RaceAnalysis:
    contest_id: str
    contest: str
    contest_order: float
    source_id: str
    jurisdiction: str
    snapshot: str
    ballots: int
    valid_votes: int
    rule: str
    controlling: bool
    scope_note: str
    choices: tuple[ChoiceResult, ...]
    decisions: tuple[DecisionAnalysis, ...]
    overall_risk: str


@dataclass(frozen=True)
class ElectionAnalysis:
    title: str
    as_of: str
    races: tuple[RaceAnalysis, ...]
    snapshots: tuple[Snapshot, ...]
