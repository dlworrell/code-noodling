# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:83e327cd-d7f5-416a-9424-b077a42cf0c3

"""Ingest separate, dated official round/recount records; never subtract stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .evidence import timestamp
from .model import ElectionReportError


def validate_versions(payload):
    if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
        raise ElectionReportError("result versions require schema 1 records")
    keys = set()
    for record in payload["records"]:
        key = tuple(record.get(k) for k in ("canonical_id", "result_version", "round_id"))
        if not all(key) or key in keys:
            raise ElectionReportError("missing/duplicate official result identity/version/round")
        keys.add(key)
        if not record.get("authority_url", "").startswith("https://") or not timestamp(record.get("evidence_as_of")):
            raise ElectionReportError("official version requires dated HTTPS authority")
        if record.get("tabulation_stage") not in {"raw_first_rankings", "rcv_round", "final_round", "recount", "plurality_count"}:
            raise ElectionReportError("unsupported result version stage")
        if record.get("coverage_status") == "unresolved":
            if not record.get("finding"):
                raise ElectionReportError("unresolved official coverage needs a finding")
            continue
        values = record.get("votes", {})
        if not values or any(type(v) is not int or v < 0 for v in values.values()):
            raise ElectionReportError("official votes must be nonnegative integer totals")
        for k in ("continuing", "blanks", "exhausted", "overvotes", "ballots"):
            if type(record.get(k)) is not int or record[k] < 0:
                raise ElectionReportError(f"official version {k} must be a nonnegative integer")
        if sum(values.values()) != record["continuing"] or sum(record[k] for k in ("continuing", "blanks", "exhausted", "overvotes")) != record["ballots"]:
            raise ElectionReportError("official version ballot conservation failed")
    return tuple(payload["records"])


def load_versions(config):
    selector = config.audit_contract.get("official_versions")
    if not selector:
        return ()
    path = Path(selector["path"])
    if path.is_absolute() or ".." in path.parts or not config.directory:
        raise ElectionReportError("official versions path must remain inside election directory")
    path = config.directory / path
    raw = path.read_bytes()
    if len(raw) > 8_000_000 or hashlib.sha256(raw).hexdigest() != selector.get("sha256"):
        raise ElectionReportError("official result version bytes do not match reviewed hash/budget")
    return validate_versions(json.loads(raw))
