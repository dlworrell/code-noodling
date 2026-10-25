# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:b42e937f-e4a7-4beb-bd58-f5b7fc3e1e65

"""Qualify the OC RLA inventory without inventing chronology or vote counts."""
from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter

from .model import ElectionReportError

COLUMNS = ("Container", "Tabulator", "Batch Name", "Number of Ballots")


def qualify_manifest(raw, expected_sha256):
    if not isinstance(raw, bytes) or len(raw) > 2_000_000 or b"\x00" in raw:
        raise ElectionReportError("ballot manifest byte/type budget invalid")
    sha = hashlib.sha256(raw).hexdigest()
    if sha != expected_sha256:
        raise ElectionReportError("manifest differs from publisher's SHA-256")
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ElectionReportError("manifest columns differ from recovered format")
        rows, keys, groups, tabulators = 0, set(), Counter(), set()
        for row in reader:
            if None in row or any(not isinstance(v, str) or not v or len(v) > 128 for v in row.values()):
                raise ElectionReportError("invalid manifest row shape/label")
            if row["Container"] not in {"in-person", "mail"}:
                raise ElectionReportError("unknown manifest container category")
            key = tuple(row[name] for name in COLUMNS[:3])
            if key in keys or not row["Number of Ballots"].isascii() or not row["Number of Ballots"].isdigit():
                raise ElectionReportError("duplicate batch identity or non-integer inventory")
            count = int(row["Number of Ballots"])
            if not 0 < count <= 10000:
                raise ElectionReportError("manifest batch count outside supported budget")
            keys.add(key);tabulators.add(row["Tabulator"]);groups[row["Container"]] += count;rows += 1
            if rows > 100000:
                raise ElectionReportError("manifest row budget exceeded")
        if not rows:
            raise ElectionReportError("empty manifest")
    except (UnicodeError, csv.Error) as exc:
        raise ElectionReportError("invalid UTF-8/CSV ballot manifest") from exc
    return {"schema_version": 1, "sha256": sha, "source_columns": list(COLUMNS), "rows": rows,
            "unique_composite_batch_ids": len(keys), "tabulators": sorted(tabulators),
            "source_count_column": "Number of Ballots", "source_count_sum": sum(groups.values()), "counts_by_container": dict(groups),
            "source_unit_qualification": "The associated official RLA report calls this a ballot-sheet manifest and describes ballot cards audited. Its row sum must not be substituted for unique voters, counted voter ballots or contest pair votes.",
            "publication_or_count_clock": None, "candidate_votes": None, "correction_partition": None,
            "new_ballot_likelihood_eligible": False,
            "missing_fields": ["historical snapshot/publication time", "per-candidate and contest vote totals", "new-ballot versus correction/recount/transfer lineage", "outstanding acceptance/geographic categories"],
            "qualification": "Publisher-hash-verified final audit inventory. Batch identifiers alone do not establish dated new-ballot increments or forecast eligibility."}
