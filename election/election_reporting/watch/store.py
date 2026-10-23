"""One election's acquired snapshots and their provenance record.

Layout under ``election/data/elections/<state>/<YYYY-MM-DD>/``:

- ``election.toml``: the analyzer configuration for this election;
- ``input/<place>/<YYYY-MM-DD-HHMM>.csv``: plain-text CSV the analyzer reads;
- ``raw/<place>/<YYYY-MM-DD-HHMM>.<ext>``: the original download, when it is
  not itself the input (XLSX workbooks, and API JSON trimmed and gzipped);
- ``SOURCES.json``: for every snapshot, the source URL, retrieval time,
  source timestamp and SHA-256 of each stored file.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SOURCES_FILE = "SOURCES.json"


@dataclass(frozen=True)
class SnapshotRecord:
    place: str
    snapshot: str  # YYYY-MM-DD-HHMM, local time of the source's own timestamp
    kind: str  # "county-csv", "state-xlsx" or "votewa-json"
    source_url: str
    retrieved_at: str  # ISO 8601, UTC
    source_timestamp: str  # ISO 8601 as reported by the source, or ""
    input: str  # path relative to the election directory
    input_sha256: str
    raw: str = ""
    raw_sha256: str = ""
    note: str = ""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gzip_bytes(data: bytes) -> bytes:
    """Deterministic gzip: no file name and a fixed timestamp."""
    return gzip.compress(data, compresslevel=9, mtime=0)


class ElectionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / SOURCES_FILE
        self.records: list[SnapshotRecord] = []
        self.election: dict = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.election = data.get("election", {})
            self.records = [SnapshotRecord(**item) for item in data.get("snapshots", [])]

    def latest(self, place: str) -> SnapshotRecord | None:
        matching = [record for record in self.records if record.place == place]
        return max(matching, key=lambda record: record.snapshot) if matching else None

    def kinds(self, place: str) -> set[str]:
        return {record.kind for record in self.records if record.place == place}

    def has_snapshot(self, place: str, snapshot: str) -> bool:
        return any(r.place == place and r.snapshot == snapshot for r in self.records)

    def write_file(self, relative: str, data: bytes) -> str:
        target = self.directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return sha256(data)

    def add(self, record: SnapshotRecord) -> None:
        self.records.append(record)

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "election": self.election,
            "snapshots": [
                asdict(record)
                for record in sorted(self.records, key=lambda r: (r.place, r.snapshot))
            ],
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
