# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:99a55549-721f-437d-acf1-d25ff2861e65

"""EA-02/03/04/08/09/11/14: reviewable evidence, never inferred authority.

The versioned contract is specified in ADR-004. Missing evidence is a reason to
withhold a scenario, not an instruction to invent a date, identity or zero pool.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .model import ElectionConfig, ElectionReportError, Snapshot


def timestamp(value):
    if not value or str(value).upper() == "UNKNOWN":
        return None
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except ValueError:
        try:
            result = parsedate_to_datetime(str(value))
        except (ValueError, TypeError, OverflowError):
            return None
    return result.astimezone(timezone.utc) if result.tzinfo and result.utcoffset() is not None else None


def iso(value):
    result = timestamp(value)
    return result.isoformat() if result else None


def validate_contract(raw, source_ids):
    if not isinstance(raw, dict):
        raise ElectionReportError("[audit] must be a table")
    if raw and raw.get("schema_version") != 1:
        raise ElectionReportError("audit.schema_version must be 1")
    age = raw.get("max_source_age_hours", 48)
    if type(age) not in (int, float) or not math.isfinite(age) or age <= 0:
        raise ElectionReportError("audit.max_source_age_hours must be positive and finite")
    keys = set()
    selectors = set()
    controlling_ids = set()
    for name in ("identities", "source_versions", "area_versions", "remaining_pools", "rules"):
        rows = raw.get(name, [])
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ElectionReportError(f"audit.{name} must be an array of tables")
        for row in rows:
            selector = (name, row.get("canonical_id")) if name in {"rules", "remaining_pools"} else (name, row.get("source_id"), row.get("snapshot_sha256")) if name in {"source_versions", "area_versions"} else None
            if selector is not None:
                if selector in selectors:
                    raise ElectionReportError(f"duplicate audit.{name} selector")
                selectors.add(selector)
            if name in {"identities", "source_versions"} and row.get("source_id") not in source_ids:
                raise ElectionReportError(f"audit.{name}.source_id is not configured")
            if name == "identities":
                key = (row.get("source_id"), row.get("contest_id"))
                if key in keys:
                    raise ElectionReportError("duplicate reviewed source/contest identity")
                keys.add(key)
                if row.get("role") == "controlling":
                    if row.get("canonical_id") in controlling_ids:
                        raise ElectionReportError("a canonical contest may have only one reviewed controlling source identity")
                    controlling_ids.add(row.get("canonical_id"))
                aliases = row.get("area_aliases", [])
                if not isinstance(aliases, list) or not all(isinstance(a,dict) for a in aliases):
                    raise ElectionReportError("reviewed area aliases must be objects")
                alias_keys = [(a.get("contest_id"),a.get("name"),a.get("jurisdiction")) for a in aliases]
                if len(set(alias_keys)) != len(alias_keys):
                    raise ElectionReportError("duplicate reviewed area alias would count geography twice")
                required = ("canonical_id", "jurisdiction_id", "office", "source_contest_name", "authority_url")
                if not raw.get("election_id") or any(not isinstance(row.get(k), str) or not row[k] for k in required):
                    raise ElectionReportError("reviewed identity requires election, jurisdiction, office, authority and canonical IDs")
                if not isinstance(row.get("jurisdictions"), list) or len(set(row["jurisdictions"])) != len(row["jurisdictions"]):
                    raise ElectionReportError("reviewed identity requires distinct jurisdiction members")
                if not row["authority_url"].startswith("https://"):
                    raise ElectionReportError("reviewed identity requires an HTTPS authority")
                if row.get("role") not in {"controlling", "partial"}:
                    raise ElectionReportError("reviewed identity role must be controlling or partial")
            if name == "remaining_pools":
                for k in ("ballots", "low", "base", "high"):
                    if k in row and (type(row[k]) is not int or row[k] < 0):
                        raise ElectionReportError(f"remaining pool {k} must be a nonnegative integer")
                for k in ("contest_fraction", "pair_fraction"):
                    if k in row and (type(row[k]) not in (int, float) or not math.isfinite(row[k]) or not 0 <= row[k] <= 1):
                        raise ElectionReportError(f"remaining pool {k} must be a fraction from 0 to 1")
    return raw


def hydrate_snapshots(config: ElectionConfig, source, snapshots):
    records = []
    observations = []
    if config.directory and (path := config.directory / "SOURCES.json").exists():
        if path.stat().st_size > 16_000_000:
            raise ElectionReportError("SOURCES.json exceeds the metadata size budget")
        ledger = json.loads(path.read_text())
        records = ledger.get("snapshots", [])
        observations = ledger.get("observations", [])
    overrides = config.audit_contract.get("source_versions", [])
    output = []
    for snapshot in snapshots:
        matches = [r for r in records if r.get("place") == source.source_id and r.get("input_sha256") == snapshot.sha256]
        known = matches[0] if matches else {}
        explicit = [r for r in overrides if r.get("source_id") == source.source_id and r.get("snapshot_sha256") == snapshot.sha256]
        if len(explicit) > 1:
            raise ElectionReportError("duplicate source version evidence")
        facts = {
            "source_published_at": iso(known.get("source_timestamp")),
            "count_updated_at": None,
            "retrieved_at": iso(known.get("retrieved_at")),
            "source_url": known.get("source_url"),
            "timestamp_kind": "server_last_modified" if known.get("source_timestamp") else "unknown",
            "legacy_filename_label": snapshot.timestamp.isoformat(),
            "legacy_filename_timezone": "unknown",
            "publication_id": "",
            "coverage_complete": False,
            "coverage_findings": ["Source coverage has not been independently established."],
        }
        if explicit:
            facts.update(explicit[0])
            for name in ("source_published_at", "count_updated_at", "retrieved_at"):
                facts[name] = iso(facts.get(name))
        # Same bytes cannot acquire a later count update merely by retrieval.
        retrievals = sorted({iso(r.get("retrieved_at")) for r in matches + [o for o in observations if o.get("place") == source.source_id and o.get("input_sha256") == snapshot.sha256]} - {None})
        if retrievals:
            facts["retrieved_at"] = retrievals[-1]
        output.append(replace(snapshot, source_metadata=facts))
    return tuple(output)


def identity(config, source, contest):
    rows = [r for r in config.audit_contract.get("identities", [])
            if r["source_id"] == source.source_id and r.get("contest_id") == contest.contest_id and r.get("source_contest_name") == contest.name]
    if rows:
        row = dict(rows[0])
        row["reviewed"] = True
        row["election_id"] = config.audit_contract.get("election_id")
        row["complete"] = bool(row.get("jurisdictions")) and row.get("coverage_complete") is True
        return row
    return {"canonical_id": f"unreviewed:{config.audit_contract.get('election_id', 'unknown')}:{source.source_id}:{contest.contest_id}:{contest.key}",
            "reviewed": False, "complete": False, "role": "partial", "jurisdictions": [],
            "finding": "No reviewed election/jurisdiction/office/source-ID crosswalk."}


def source_facts(snapshot, contest, information_time):
    facts = dict(snapshot.source_metadata)
    supplemental = contest.metadata.get("source_evidence", {})
    facts.update(supplemental)
    facts["coverage_complete"] = snapshot.source_metadata.get("coverage_complete") is True and supplemental.get("coverage_complete", True) is True
    moment = timestamp(facts.get("count_updated_at") or facts.get("source_published_at"))
    facts["freshness_basis"] = "explicit count-update time" if facts.get("count_updated_at") else "source publication proxy; count-update time unknown"
    facts["source_age_hours"] = ((information_time - moment).total_seconds() / 3600
                                 if information_time and moment else None)
    facts["snapshot_sha256"] = snapshot.sha256
    facts["geography"] = contest.metadata.get("geography", {})
    if facts["geography"].get("approximate"):
        facts["coverage_complete"] = False
    return facts


def load_geography(config):
    from .areas import area_files, read_area_file
    result = []
    for area in config.areas:
        if area.level != "county" or not (files := area_files(area)):
            continue
        path = files[-1]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        versions = [r for r in config.audit_contract.get("area_versions", [])
                    if r.get("source_id") == area.source_id and r.get("snapshot_sha256") == digest]
        result.append((read_area_file(path, area.source_id, "county"), versions[0] if len(versions) == 1 else {}))
    return result


def geographic_comparison(contest, ident, facts, geography):
    """Only reviewed ID/name/jurisdiction aliases may join headline/detail votes."""
    aliases = ident.get("area_aliases", [])
    if not ident.get("reviewed") or not aliases:
        return {"status": "not_joined", "reason": "No reviewed area identity aliases."}
    for snapshot, version in geography:
        aggregate = Counter()
        found = []
        for alias in aliases:
            item = snapshot.contests.get((alias.get("contest_id"), alias.get("name")))
            votes = item.areas.get(alias.get("jurisdiction")) if item else None
            if votes is None:
                continue
            found.append(alias["jurisdiction"])
            aggregate.update({name.casefold(): count for name, count in votes.items()})
        complete = len(found) == len(aliases) and set(found) == set(ident.get("jurisdictions", []))
        if not found:
            continue
        headline = {choice.name.casefold():choice.votes for choice in contest.choices}
        named = {choice.name.casefold():choice.name for choice in contest.choices}
        geography_choices = set(aggregate) - {"ballots cast", "over votes", "under votes"}
        differences = {named.get(name,name): {"headline": headline.get(name,0), "geography": aggregate.get(name,0)}
                       for name in sorted(set(headline)|geography_choices) if headline.get(name,0) != aggregate.get(name,0)}
        same = bool(facts.get("publication_id")) and facts.get("publication_id") == version.get("publication_id")
        return {"status": "contradictory" if differences and same and complete else "source_vintage_conflict" if differences else "reconciled" if same and complete else "incompatible_or_partial",
                "same_publication": same, "complete_membership": complete, "jurisdictions": found,
                "differences": differences, "publication_id": version.get("publication_id"),
                "area_source_label": snapshot.stamp, "arithmetic_sum": dict(aggregate)}
    return {"status": "missing", "reason": "Reviewed area aliases are missing from the supplied geography."}


def reviewed_rule(config, canonical_id, inferred):
    rows = [r for r in config.audit_contract.get("rules", []) if r.get("canonical_id") == canonical_id]
    if not rows:
        return {"rule": inferred, "supported": False, "authority_url": None,
                "reason": "Inferred rule is descriptive only; governing office/election rule is unreviewed."}
    rule = dict(rows[0])
    kind = rule.get("rule")
    supported = kind in {"winner", "top_two", "top_four", "wa_judicial"}
    offices = {r.get("office") for r in config.audit_contract.get("identities", [])
               if r.get("canonical_id") == canonical_id}
    scope_matches = (len(offices) == 1 and bool(rule.get("office"))
                     and offices == {rule["office"]}
                     and bool(rule.get("election_type"))
                     and rule["election_type"] == config.audit_contract.get("election_type"))
    supported = supported and scope_matches
    if row_positions := rule.get("positions"):
        supported = supported and row_positions == {"winner": 1, "top_two": 2, "top_four": 4, "wa_judicial": 1}.get(kind)
    else:
        supported = False
    if rule.get("passage_denominator", 2) != 2 or rule.get("passage_numerator", 1) != 1:
        supported = False
    if rule.get("validation_requirements") or not str(rule.get("authority_url", "")).startswith("https://"):
        supported = False
    rule.update(supported=supported, scope_matches=scope_matches,
                reason="Reviewed office/election binary-boundary rule" if supported else
                "Office/election rule scope is missing or mismatched" if not scope_matches else
                "Unsupported passage/validation/RCV rule")
    return rule


def remaining_pool(config, canonical_id, jurisdictions, source_time):
    rows = [r for r in config.audit_contract.get("remaining_pools", []) if r.get("canonical_id") == canonical_id]
    unknown = {"status": "unknown", "kind": "unavailable", "ballots": None, "reason": "No contemporaneous remaining-count evidence or explicit scenario."}
    if not rows:
        return unknown
    if len(rows) != 1:
        return {**unknown, "reason": "Conflicting remaining pools require review."}
    row = dict(rows[0])
    as_of = timestamp(row.get("as_of"))
    info = config.information_time
    if not as_of or not info or as_of > info or (source_time and as_of < source_time):
        return {**unknown, "reason": "Remaining pool time is missing, stale or after information time."}
    if (info-as_of).total_seconds() > 3600 * config.audit_contract.get("max_source_age_hours", 48):
        return {**unknown, "reason": "Remaining pool exceeds the declared freshness limit."}
    if set(row.get("jurisdictions", [])) != set(jurisdictions) or not row.get("coverage_complete"):
        return {**unknown, "reason": "Remaining pool jurisdiction/category coverage is incomplete or mismatched."}
    if row.get("kind") == "official":
        if not row.get("authority_url") or "ballots" not in row or not row.get("categories"):
            return {**unknown, "reason": "Official pool lacks authority/count/category evidence."}
        categories = row["categories"]
        if (not isinstance(categories, list) or not all(isinstance(c,dict) and isinstance(c.get("category"),str) and type(c.get("ballots")) is int and c["ballots"] >= 0 for c in categories)
                or len({c["category"] for c in categories}) != len(categories)
                or sum(c["ballots"] for c in categories) != row["ballots"]):
            return {**unknown, "reason": "Official category counts are invalid, duplicate, or do not sum to the pool."}
        if row.get("ballot_unit") not in {"accepted_uncounted", "returned_potentially_countable"} or not row.get("acceptance_assumption"):
            return {**unknown, "reason": "Official pool must distinguish returned/accepted/countable ballots and document acceptance/cure assumptions."}
        row["status"] = "confirmed_zero" if row["ballots"] == 0 else "official_outstanding"
    elif row.get("kind") == "scenario":
        if not row.get("basis") or not row.get("category_basis") or not all(k in row for k in ("low", "base", "high")) or not row["low"] <= row["base"] <= row["high"]:
            return {**unknown, "reason": "Scenario must declare ordered low/base/high counts, basis and ballot-category assumptions."}
        row["ballots"] = row["base"]
        row["status"] = "conditional_scenario"
    else:
        return unknown
    if row["ballots"] and ("contest_fraction" not in row or "pair_fraction" not in row or not row.get("participation_basis")):
        return {**unknown, "reason": "Contest/pair participation mapping is not specified."}
    row["as_of"] = as_of.isoformat()
    row["interpretation"] = "Scenario sensitivity is not a confidence interval or calibrated forecast."
    return row


def scientific_gate(config, context, lifecycle_ok, lifecycle_reason):
    facts, ident, rule, pool = (context[k] for k in ("source", "identity", "rule", "remaining_pool"))
    age = facts.get("source_age_hours")
    checks = {
        "lifecycle_and_stage": lifecycle_ok,
        "source_time": age is not None and 0 <= age <= config.audit_contract.get("max_source_age_hours", 48),
        "identity": ident.get("reviewed") is True,
        "complete_scope": ident.get("complete") is True and ident.get("role") == "controlling",
        "source_coverage": facts.get("coverage_complete") is True,
        "provenance": bool(facts.get("snapshot_sha256") and facts.get("source_url") and facts.get("retrieved_at")),
        "geography_compatibility": context.get("geography_comparison", {}).get("status") not in {"contradictory", "source_vintage_conflict", "missing", "incompatible_or_partial"},
        "noncontradictory_data": context["data_status"] == "reported",
        "supported_rule": rule.get("supported") is True,
        "remaining_pool": pool.get("status") in {"official_outstanding", "confirmed_zero", "conditional_scenario"},
        "model_status": "experimental_uncalibrated",
    }
    reasons = [name for name, value in checks.items() if value is False]
    if not lifecycle_ok:
        reasons.append(lifecycle_reason)
    if pool.get("status") == "unknown":
        reasons.append(pool["reason"])
    return {"schema_version": 1, "eligible": not reasons,
            "publication_policy": "experimental_conditional_scenarios_only", "validated_forecast_eligible": False,
            "validation_finding": "No frozen held-out calibration evidence; experimental scenario numbers are not validated forecasts.", "checks": checks,
            "reasons": reasons, "calibration_status": "unvalidated", "rubric_status": "unvalidated_heuristic"}


def summarize_sources(races):
    vintages = Counter((r.source_id, str(r.evidence.get("source", {}).get("source_published_at")), r.snapshot) for r in races)
    return {"schema_version": 1, "vintages": [{"source_id": s, "source_published_at": None if p == "None" else p,
            "legacy_filename_label": label, "records": count} for (s,p,label),count in sorted(vintages.items())],
            "missing_source_time": sum(r.evidence.get("source", {}).get("source_age_hours") is None for r in races),
            "missing_coverage": sum(not r.evidence.get("source", {}).get("coverage_complete") for r in races)}
