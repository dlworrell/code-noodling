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
    if payload.get("schema_version") not in {1, 2} or not isinstance(payload.get("records"), list):
        raise ElectionReportError("result versions require schema 1/2 records")
    keys = set()
    records = []
    for record in payload["records"]:
        record = dict(record)
        key = tuple(record.get(k) for k in ("canonical_id", "result_version", "round_id"))
        if not all(key) or key in keys:
            raise ElectionReportError("missing/duplicate official result identity/version/round")
        keys.add(key)
        if not record.get("authority_url", "").startswith("https://") or not timestamp(record.get("evidence_as_of")):
            raise ElectionReportError("official version requires dated HTTPS authority")
        if record.get("tabulation_stage") not in {"raw_first_rankings", "rcv_round", "final_round", "recount", "plurality_count"}:
            raise ElectionReportError("unsupported result version stage")
        if payload["schema_version"] == 2:
            required = {"source_published_at", "count_updated_at", "certification_status"}
            if not required <= record.keys() or not record["certification_status"]:
                raise ElectionReportError("version 2 requires explicit publication/count clocks and certification status")
            if any(record[k] is not None and not timestamp(record[k]) for k in ("source_published_at", "count_updated_at")):
                raise ElectionReportError("version publication/count time needs an offset or null")
        records.append(record)
        if record.get("coverage_status") == "unresolved":
            if not record.get("finding"):
                raise ElectionReportError("unresolved official coverage needs a finding")
            continue
        values = record.get("votes", {})
        if not values or any(type(v) is not int or v < 0 for v in values.values()):
            raise ElectionReportError("official votes must be nonnegative integer totals")
        if record.get("coverage_status") == "partial_accounting":
            if (not record.get("finding") or type(record.get("continuing")) is not int
                    or sum(values.values()) != record["continuing"]
                    or type(record.get("ballots")) is not int or record["ballots"] < record["continuing"]
                    or record.get("noncontinuing_total") != record["ballots"] - record["continuing"]):
                raise ElectionReportError("partial official accounting must conserve known combined totals and retain its finding")
            components = [record.get(k) for k in ("blanks", "exhausted", "overvotes")]
            if any(v is not None and (type(v) is not int or v < 0) for v in components):
                raise ElectionReportError("unknown partial accounting categories must be null")
            if sum(v for v in components if v is not None) > record["noncontinuing_total"]:
                raise ElectionReportError("partial accounting exceeds known noncontinuing total")
            continue
        for k in ("continuing", "blanks", "exhausted", "overvotes", "ballots"):
            if type(record.get(k)) is not int or record[k] < 0:
                raise ElectionReportError(f"official version {k} must be a nonnegative integer")
        if sum(values.values()) != record["continuing"] or sum(record[k] for k in ("continuing", "blanks", "exhausted", "overvotes")) != record["ballots"]:
            raise ElectionReportError("official version ballot conservation failed")
    selections = payload.get("selections", [])
    selected = set()
    for selection in selections:
        key = tuple(selection.get(k) for k in ("canonical_id", "result_version", "round_id"))
        if key not in keys or key[0] in selected or not str(selection.get("authority_url", "")).startswith("https://") or not selection.get("reason"):
            raise ElectionReportError("official selection requires a unique existing version and reviewed authority/reason")
        chosen = next(r for r in records if tuple(r[k] for k in ("canonical_id", "result_version", "round_id")) == key)
        if chosen.get("coverage_status") == "unresolved" or chosen.get("tabulation_stage") in {"raw_first_rankings", "rcv_round"}:
            raise ElectionReportError("unresolved/raw/intermediate stage cannot be selected as current final reference")
        previous = selection.get("supersedes", [])
        if not isinstance(previous, list) or not previous or len(set(previous)) != len(previous):
            raise ElectionReportError("official selection requires distinct superseded version lineage")
        for version in previous:
            if version == key[1] or not any(r["canonical_id"] == key[0] and r["result_version"] == version for r in records):
                raise ElectionReportError("official superseded lineage is missing or self-referential")
        selected.add(key[0])
        chosen["selection"] = {**selection, "status": "authoritative_vote_totals_with_partial_accounting" if chosen.get("coverage_status") == "partial_accounting" else "selected_official_reference",
                               "usage": "retrospective descriptive reference; never a pre-outcome forecast input"}
    return tuple(records)


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
