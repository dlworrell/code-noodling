# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:cf52b7b1-2509-4bc9-984c-c19941b49154

"""Bounded archival readers, separate from live-source acquisition and forecasts.

PRM-0026: preserve source format, units, encoding and unsupported evidence.
Reading a dated count does not establish then-available outstanding ballots.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict

from .model import ElectionReportError
from .watch import csvformat


def _rows(raw, required, *, max_bytes, max_rows, cp1252=False):
    if not raw or len(raw) > max_bytes:
        raise ElectionReportError("archive is empty or exceeds its byte budget")
    encoding = "utf-8-sig"
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        if not cp1252:
            raise ElectionReportError("archive is not UTF-8") from exc
        encoding = "cp1252"
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as alternate:
            raise ElectionReportError("archive encoding is unsupported") from alternate
    if "\x00" in text:
        raise ElectionReportError("archive contains NUL bytes")
    try:
        reader = csv.DictReader(io.StringIO(text))
        header = reader.fieldnames or []
        if len(set(header)) != len(header) or not set(required) <= set(header):
            raise ElectionReportError("archive has duplicate or missing required columns")
        output = []
        for line, row in enumerate(reader, 2):
            if len(output) >= max_rows or None in row or any(v is None or len(v) > 4096 for v in row.values()):
                raise ElectionReportError(f"archive row {line} exceeds shape/row/field budget")
            output.append(row)
    except csv.Error as exc:
        raise ElectionReportError("invalid archive CSV") from exc
    if not output:
        raise ElectionReportError("archive has no data rows")
    return output, encoding


def _integer(value):
    if not re.fullmatch(r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)", value):
        raise ElectionReportError("archive count is not a nonnegative integer")
    result = int(value.replace(",", ""))
    if result > 1_000_000_000:
        raise ElectionReportError("archive count exceeds numeric budget")
    return result


KING_COLUMNS = ("GEMS Contest ID", "Contest Sort Seq", "District Type", "District Type Subheading",
                "District Name", "Ballot Title", "Ballots Counted for District", "Registered Voters for District",
                "Candidate Sort Seq", "Ballot Response", "Votes")


def king_snapshot(raw):
    """King's 2022–2025 GEMS archive, including explicit Windows-1252 files.

    District turnout ballots remain district ballots; missing contest-level
    under/overvotes are never filled with zero. Composite IDs do not assert a
    reviewed cross-election or cross-county identity.
    """
    rows, encoding = _rows(raw, KING_COLUMNS, max_bytes=8_000_000, max_rows=50_000, cp1252=True)
    contests = {}
    for row in rows:
        identity = tuple(row[k].strip() for k in ("GEMS Contest ID", "District Type", "District Type Subheading", "District Name", "Ballot Title"))
        if not identity[0] or not identity[-1] or not row["Ballot Response"].strip():
            raise ElectionReportError("King archive lacks contest/choice identity")
        key = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
        ballots = _integer(row["Ballots Counted for District"])
        registered = _integer(row["Registered Voters for District"])
        item = contests.setdefault(key, {"source_identity": list(identity), "district_ballots": ballots,
            "registered_voters": registered, "votes": {}, "contest_undervotes": None, "contest_overvotes": None,
            "identity_reviewed": False, "whole_contest_coverage": False})
        choice = row["Ballot Response"].strip()
        if choice in item["votes"] or ballots != item["district_ballots"] or registered != item["registered_voters"]:
            raise ElectionReportError("King archive repeats a choice or contradicts district accounting")
        item["votes"][choice] = _integer(row["Votes"])
    return {"schema_version": 1, "format": "king_gems_archive_v1", "encoding": encoding,
            "sha256": hashlib.sha256(raw).hexdigest(), "contests": contests,
            "qualification": "Dated descriptive count; district ballots are not valid contest votes or outstanding ballots; publication clock and governing rule require separate evidence."}


MEDSL_COLUMNS = ("precinct", "office", "candidate", "district", "mode", "votes", "party_detailed", "year", "state", "stage", "magnitude")
MEDSL_MODES = {"ELECTION DAY", "ABSENTEE", "EARLY VOTING", "QUESTION", "FEDERAL OVERSEAS"}


def alaska_medsl(raw, crosswalk, places):
    """Convert the frozen 2024 derivative, preserving every supplied mode once.

    MEDSL pads precinct IDs (001-600 -> 01-600) and has zero-filled repetitions.
    At most one positive contribution per precinct/office/district/mode/choice
    is allowed. District early-vote rows mislabeled ELECTION DAY cannot enter
    borough slices. Judicial retention questions are separated by judge.
    """
    rows, encoding = _rows(raw, MEDSL_COLUMNS, max_bytes=64_000_000, max_rows=250_000)
    unique = {}
    zero_repeats = 0
    for row in rows:
        if row["year"] != "2024" or row["state"] != "ALASKA" or row["stage"] != "GEN" or row["magnitude"] != "1" or row["mode"] not in MEDSL_MODES:
            raise ElectionReportError("unsupported MEDSL election or ballot mode")
        if any(not row[k].strip() for k in ("precinct", "office", "candidate", "district")):
            raise ElectionReportError("MEDSL row lacks identity")
        key = tuple(row[k] for k in ("precinct", "office", "district", "mode", "candidate"))
        value = _integer(row["votes"])
        if key in unique:
            previous, previous_value = unique[key]
            if previous_value and value or previous["party_detailed"] != row["party_detailed"]:
                raise ElectionReportError("ambiguous repeated positive MEDSL contribution")
            zero_repeats += 1
            if previous_value:
                continue
        unique[key] = row, value
    totals = {p.place_id: defaultdict(Counter) for p in places}
    parties = {}
    modes = {p.place_id: Counter() for p in places}
    state = next((p for p in places if p.scope == "state"), None)
    if state is None:
        raise ElectionReportError("MEDSL conversion requires a statewide target")
    conflicts = set()
    for row, value in unique.values():
        name, office, district = row["candidate"], row["office"], row["district"]
        if office.startswith("RETENTION"):
            judge, separator, name = name.rpartition(" - ")
            if not separator or name not in {"YES", "NO", "UNDERVOTES", "OVERVOTES"}:
                raise ElectionReportError("unsupported MEDSL judicial choice")
            title = f"{office} {district}: {judge}"
        else:
            title = office if district == "STATEWIDE" else f"{office} District {district}"
        parties[title, name] = row["party_detailed"]
        precinct = row["precinct"]
        borough = None
        if row["mode"] == "ELECTION DAY" and re.fullmatch(r"\d{3}-\d{3}", precinct):
            normalized = f"{int(precinct[:3]):02d}-{precinct[4:]}"
            borough = crosswalk.precincts.get(normalized)
            if value and borough is None:
                raise ElectionReportError(f"MEDSL precinct missing from crosswalk: {normalized}")
        elif value and row["mode"] == "ELECTION DAY":
            conflicts.add(precinct)
        targets = [state] + [p for p in places if p.scope != "state" and p.name == borough]
        for place in targets:
            totals[place.place_id][title][name] += value
            modes[place.place_id][row["mode"]] += value
    outputs = {}
    for place in places:
        normalized_rows = []
        for order, (title, tally) in enumerate(sorted(totals[place.place_id].items()), 1):
            under, over = tally.get("UNDERVOTES", 0), tally.get("OVERVOTES", 0)
            choices = {n: v for n, v in tally.items() if n not in {"UNDERVOTES", "OVERVOTES"}}
            ballots = sum(choices.values()) + under + over
            for position, (name, value) in enumerate(sorted(choices.items()), 1):
                normalized_rows.append(csvformat.Row(order, title, re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"),
                    position, name, ballots, value, over, under, parties[title, name], place.name,
                    place.name if place.scope != "state" else "", (), "WRITE" in name.upper(),
                    {"geography": {**crosswalk.vintage, "approximate": place.scope != "state",
                        "label": "all supplied statewide categories (MEDSL derivative)" if place.scope == "state" else "election-day precinct subset / approximate borough assignment",
                        "whole_contest_coverage": False,
                        "split_precincts": list(crosswalk.split_precincts),
                        "included_categories": sorted(modes[place.place_id]),
                        "excluded_categories": [] if place.scope == "state" else sorted(MEDSL_MODES - {"ELECTION DAY"}) + ["district-level rows mislabeled ELECTION DAY"],
                        "reconciliation_policy": "Expected geographic subset; borough/state difference is not an arithmetic error.",
                        "source": "frozen derivative; official mode completeness not independently established"},
                     "archive_format": "medsl_alaska_2024_v1"}))
        if normalized_rows:
            outputs[place.place_id] = csvformat.render(normalized_rows)
    evidence = {"schema_version": 1, "format": "medsl_alaska_2024_v1", "encoding": encoding,
                "input_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows), "zero_filled_repetitions": zero_repeats,
                "mode_contributions": {p: dict(c) for p, c in modes.items()}, "district_rows_mislabeled_election_day": sorted(conflicts),
                "qualification": "Mode contributions count contest selections, not distinct voters. All supplied categories retained statewide; derivative provenance and approximate geography remain explicit."}
    return outputs, evidence


def main():
    import argparse
    from datetime import datetime, timezone
    from pathlib import Path
    from .watch.alaska import load_crosswalk, resolve_places
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medsl-alaska",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    crosswalk=load_crosswalk(args.data_root)
    outputs,evidence=alaska_medsl(args.medsl_alaska.read_bytes(),crosswalk,resolve_places(crosswalk,(),True))
    observed=datetime.now(timezone.utc)
    args.output.mkdir(parents=True,exist_ok=True)
    evidence["executed_at"]=observed.isoformat()
    evidence["normalized_filename_qualification"]="Reconstruction observation date, not a 2024 count-update timestamp."
    evidence["outputs"]=[]
    for name,text in outputs.items():
        path=args.output/name/f"{observed.date().isoformat()}.csv"
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        evidence["outputs"].append({"place_id":name,"path":str(path.relative_to(args.output)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.output/"coverage.json").write_text(json.dumps(evidence,indent=2)+"\n")


if __name__=="__main__":
    main()
