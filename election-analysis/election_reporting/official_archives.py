# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:7194eda3-20a7-4014-9b30-24e11df9fe39

"""Bounded readers for original Orange County count PDFs and official exports.

PRM-0026: preserve source clocks, amended forms, voting-method units and
ambiguous field semantics. Parsing never establishes forecast eligibility.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

from .model import ElectionReportError

INTEGER = r"[0-9]+(?:,[0-9]{3})*"
PERCENT = r"[0-9]+(?:\.[0-9]+)?%"
NUMERIC_ROW = re.compile(rf"^(.*?)\s+({INTEGER})\s+({PERCENT})\s+({INTEGER})\s+({PERCENT})\s+({INTEGER})\s+({PERCENT})\s*$")
DISTRICT_ROW = re.compile(rf"^\s*({INTEGER})\s+({INTEGER})\s+({PERCENT})\s+({INTEGER})\s+({INTEGER})\s+({PERCENT})\s*$")
PARTIES = {"DEM", "REP", "AI", "GRN", "LIB", "P/F", "AIP", "NPP"}
PACIFIC = ZoneInfo("America/Los_Angeles")
SUMMARY_COLUMNS = ("Contest Name", "Candidate Name", "Party", "Vote-by-Mail Ballots Votes",
                   "Vote Center Ballots Votes", "Total Votes", "Number Of Precincts",
                   "Precincts Reported", "Ballots Cast", "Under Votes", "Vote For")
POOL_FIELDS = {"voting_location":"undefined_3", "mail_received_by_election_day":"undefined_4",
               "mail_received_after_election_day":"1_2", "provisional":"2_2",
               "conditional_registration_provisional":"undefined_5", "other":"undefined_6"}


def integer(value):
    if not isinstance(value, str) or not re.fullmatch(INTEGER, value.strip()):
        raise ElectionReportError("official count must be a nonnegative integer")
    result = int(value.strip().replace(",", ""))
    if result > 1_000_000_000:
        raise ElectionReportError("official count exceeds numeric budget")
    return result


def compact(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def choice_key(name, title, party="", *, summary=False):
    """Only explicit report presentation markers; never fuzzy name merging."""
    name = compact(name)
    if summary:
        if party and name.endswith(f" ({party})"):
            name = name[:-(len(party)+3)]
        if name.startswith("*"):
            name = name[1:]
    if title == "PRESIDENT AND VICE PRESIDENT":
        name = name.replace("/", " ")
    return compact(name)


def _pdf_reader(raw):
    if not isinstance(raw, bytes) or not raw.startswith(b"%PDF-") or len(raw) > 8_000_000:
        raise ElectionReportError("count/form PDF signature or byte budget invalid")
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw), strict=True)
        if reader.is_encrypted or not 0 < len(reader.pages) <= 100:
            raise ElectionReportError("encrypted or over-budget count/form PDF")
        if sum(len(v) for v in reader.xref.values()) > 20_000:
            raise ElectionReportError("PDF object budget exceeded")
        return reader
    except (ImportError, ElectionReportError):
        raise
    except Exception as exc:
        raise ElectionReportError("invalid count/form PDF") from exc


def _metadata(reader):
    info = reader.metadata or {}
    return {"created":str(info.get("/CreationDate", "")), "modified":str(info.get("/ModDate", "")),
            "producer":str(info.get("/Producer", "")),
            "qualification":"Embedded producer metadata, not an authenticated publication clock."}


def read_count_pdf(raw):
    reader = _pdf_reader(raw)
    pages = []
    clocks = set()
    text_budget = 0
    for number, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text(extraction_mode="layout")
        except Exception as exc:
            raise ElectionReportError("PDF text extraction failed") from exc
        text_budget += len(text)
        if text_budget > 16_000_000 or len(text) > 2_000_000:
            raise ElectionReportError("PDF extracted-text budget exceeded")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not any("Cumulative Results Report" in s and "Orange County" in s for s in lines):
            raise ElectionReportError("unsupported cumulative PDF publisher/header")
        end = next((i for i,s in enumerate(lines) if re.search(rf"\bPage {number}\b",s)), None)
        if end is None:
            raise ElectionReportError("cumulative PDF page numbering is incomplete")
        header = "\n".join(lines[:end+1])
        day = re.search(r"Run Date\s+(\d{2}/\d{2}/\d{4})",header)
        clock = re.search(r"Run Time\s+(\d{1,2}:\d{2} [AP]M)",header)
        if day is None or clock is None or "Presidential General Election" not in header or "11/5/2024" not in header:
            raise ElectionReportError("unsupported election or missing PDF count clock")
        clocks.add(datetime.strptime(day[1]+" "+clock[1],"%m/%d/%Y %I:%M %p").replace(tzinfo=PACIFIC).isoformat())
        pages.extend(lines[end+1:])
    if len(clocks) != 1:
        raise ElectionReportError("mixed count clocks in one cumulative PDF")
    result = parse_count_lines(pages)
    return {"schema_version":1, "format":"oc_cumulative_pdf_2024_v1", "sha256":hashlib.sha256(raw).hexdigest(),
            "count_updated_at":next(iter(clocks)), "metadata":_metadata(reader), "pages":len(reader.pages),
            "contests":result, "source_ballots":max(c["ballots"] for c in result.values()),
            "qualification":"Publisher's cumulative count, not a disjoint batch or official outstanding pool."}


def parse_count_lines(lines):
    """PDF layout parser, separately testable with small malformed captures."""
    contests, title_lines, current, state, ended = {}, [], None, "title", False
    for line in lines:
        if len(line)>4096 or "\x00" in line:
            raise ElectionReportError("invalid/oversized PDF text field")
        if ended:
            raise ElectionReportError("unexpected data after cumulative end marker")
        if line == "*** End of report ***":
            if state!="title" or title_lines:
                raise ElectionReportError("premature cumulative end marker")
            ended=True
            continue
        if state == "title":
            if line.startswith("Precincts"):
                display = compact(" ".join(title_lines))
                title, separator, rule = display.rpartition(" - Vote for ")
                if not separator:
                    title, rule = display, "unknown measure/passage rule"
                if not title or title in contests:
                    raise ElectionReportError("missing or repeated cumulative contest identity")
                vote_for = 1 if rule in {"One", "One Party"} else None
                if rule.startswith("no more than "):
                    vote_for = {"Two":2,"Three":3,"Four":4,"Five":5}.get(rule.removeprefix("no more than "))
                    if vote_for is None:
                        raise ElectionReportError("unsupported multi-choice ballot instruction")
                current = {"title":title,"source_ballot_instruction":rule,"vote_for":vote_for,"choices":{},
                           "under_votes":None,"over_votes":None}
                title_lines=[];state="district"
            else:
                title_lines.append(line)
        elif state == "district":
            match = DISTRICT_ROW.fullmatch(line)
            if match:
                current.update(precincts_counted=integer(match[1]),precincts=integer(match[2]),
                               ballots=integer(match[4]),registered_voters=integer(match[5]))
            elif line.startswith("Choice"):
                if "ballots" not in current or "Vote Center" not in line or "Vote by Mail" not in line:
                    raise ElectionReportError("missing cumulative district/method columns")
                state="choices"
            elif not (line.startswith("Counted") or line=="Voters"):
                raise ElectionReportError("unsupported cumulative district row")
        else:
            if line.startswith("Choice") and "Vote Center" in line and "Vote by Mail" in line:
                continue
            match = NUMERIC_ROW.fullmatch(line)
            if match:
                prefix = match[1].strip()
                vc, mail, total = (integer(match[i]) for i in (2,4,6))
                if vc+mail != total or any(float(match[i][:-1])>100 for i in (3,5,7)):
                    raise ElectionReportError("cumulative voting-method accounting invalid")
                if prefix == "Cast Votes:":
                    values=list(current["choices"].values())
                    if not values or any(sum(c[key] for c in values)!=value for key,value in (("vote_center",vc),("mail",mail),("votes",total))):
                        raise ElectionReportError("cumulative candidate totals do not conserve cast votes")
                    if current["vote_for"] and total>current["ballots"]*current["vote_for"]:
                        raise ElectionReportError("valid votes exceed ballot opportunities")
                    current["cast_votes"] = total
                    contests[current["title"]] = current
                    if len(contests)>1000:
                        raise ElectionReportError("contest budget exceeded")
                    current=None;state="title"
                else:
                    party = ""
                    pieces = prefix.rsplit(None,1)
                    if len(pieces)==2 and pieces[1] in PARTIES:
                        prefix,party=pieces
                    if prefix in current["choices"]:
                        raise ElectionReportError("repeated cumulative choice")
                    current["choices"][prefix] = {"name":prefix,"party":party,"vote_center":vc,"mail":mail,"votes":total}
            elif current["choices"] and "%" not in line and not line.startswith(("Choice", "Cast Votes:")):
                # Wrapped candidate/ticket label immediately after its numeric row.
                if current["title"] != "PRESIDENT AND VICE PRESIDENT" or any(ch.isdigit() for ch in line):
                    raise ElectionReportError("unsupported wrapped or malformed choice row")
                old = next(reversed(current["choices"]))
                item=current["choices"].pop(old);item["name"]=compact(old+" "+line)
                if item["name"] in current["choices"]:
                    raise ElectionReportError("ambiguous wrapped candidate identity")
                current["choices"][item["name"]]=item
            else:
                raise ElectionReportError("unparsed cumulative choice row")
    if state!="title" or title_lines or not contests or not ended:
        raise ElectionReportError("truncated/unaccounted cumulative report")
    return contests


def parse_pool_fields(fields):
    required={"County Name","Update Report date and time","undefined","1","2","undefined_7","undefined_8",*POOL_FIELDS.values(),"11Next vote results expected date and time"}
    if not required<=set(fields) or fields["County Name"]!="Orange" or fields["Update Report date and time"]!="/On":
        raise ElectionReportError("unsupported official pool form or missing fields")
    label=fields["undefined"].strip()
    amended=label.endswith("AMENDED")
    clean=label.removesuffix("AMENDED").strip()
    try:
        count_time=datetime.strptime(clean,"%B %d, %Y %I:%M %p").replace(tzinfo=PACIFIC)
        next_time=datetime.strptime(fields["11Next vote results expected date and time"].strip(),"%B %d, %Y %I:%M %p").replace(tzinfo=PACIFIC)
    except ValueError as exc:
        raise ElectionReportError("unsupported pool form timestamp") from exc
    if count_time.year!=2024 or count_time.month!=11 or next_time<count_time:
        raise ElectionReportError("pool form chronology/election differs")
    categories={name:integer(fields[key]) for name,key in POOL_FIELDS.items()}
    total=integer(fields["undefined_7"])
    if sum(categories.values())!=total:
        raise ElectionReportError("official unprocessed category total does not conserve")
    return {"schema_version":1,"format":"ca_unprocessed_form_oc_2024_v1","county":"Orange",
            "reported_at":count_time.isoformat(),"amended":amended,"source_timestamp_label":label,
            "processed_ballots":integer(fields["1"]),"source_field_2":integer(fields["2"]),
            "source_field_2_label":"Total number of processed ballots from the last report",
            "source_field_2_interpretation":"Ambiguous label; never automatically a latest-batch count. Reconcile with publisher's dated release.",
            "unprocessed_categories":categories,"estimated_unprocessed":total,
            "challenged_left_to_cure":integer(fields["undefined_8"]),"challenged_overlap_with_total":"unknown; reported separately, never silently added",
            "next_results_expected":next_time.isoformat(),
            "qualification":"Unprocessed estimates include eligibility/acceptance uncertainty and omit not-yet-received ballots. Not a guaranteed final countable pool."}


def read_pool_pdf(raw):
    reader=_pdf_reader(raw)
    if len(reader.pages)!=1 or "November 5, 2024, General Election" not in reader.pages[0].extract_text():
        raise ElectionReportError("unsupported unprocessed PDF election/page count")
    fields={name:str(item.get("/V", "")) for name,item in (reader.get_fields() or {}).items()}
    if any(len(value)>4096 for value in fields.values()):
        raise ElectionReportError("pool PDF field budget exceeded")
    # Check page widgets as well as canonical form fields. Do not repair source PDFs.
    for ref in reader.pages[0].get("/Annots",[]):
        widget=ref.get_object()
        if widget.get("/Subtype")!="/Widget":
            continue
        field=widget.get("/Parent",widget)
        field=field.get_object() if hasattr(field,"get_object") else field
        name=str(field.get("/T", ""))
        if name in fields and str(field.get("/V", ""))!=fields[name]:
            raise ElectionReportError("widget and canonical form values disagree")
    result=parse_pool_fields(fields)
    return {**result,"sha256":hashlib.sha256(raw).hexdigest(),"metadata":_metadata(reader)}


def read_final_summary(raw):
    if len(raw)>8_000_000 or b"\x00" in raw:
        raise ElectionReportError("final summary byte budget or NUL invalid")
    try:
        lines=raw.decode("utf-8-sig").splitlines()
    except UnicodeError as exc:
        raise ElectionReportError("summary must be UTF-8") from exc
    if lines[:2] != ["Official Election Results","Orange"]:
        raise ElectionReportError("official summary status/county mismatch")
    reader=csv.reader(lines[2:])
    if tuple(next(reader,[]))!=SUMMARY_COLUMNS:
        raise ElectionReportError("unsupported/duplicate official summary columns")
    rows=list(reader)
    if not rows or len(rows)>20_000 or rows[0][:1]!=["Total Precincts Reported"] or len(rows[0])!=2:
        raise ElectionReportError("missing summary precinct total or row budget")
    total_precincts=integer(rows[0][1]);contests={}
    for values in rows[1:]:
        if len(values)!=len(SUMMARY_COLUMNS) or any(len(s)>4096 for s in values):
            raise ElectionReportError("invalid official summary row shape")
        row=dict(zip(SUMMARY_COLUMNS,values));title=compact(row["Contest Name"])
        name=choice_key(row["Candidate Name"],title,row["Party"],summary=True)
        if not title or not name:
            raise ElectionReportError("summary contest/choice identity missing")
        accounting={"ballots":integer(row["Ballots Cast"]),"under_votes":integer(row["Under Votes"]),
                    "precincts":integer(row["Number Of Precincts"]),"precincts_counted":integer(row["Precincts Reported"]),"vote_for":integer(row["Vote For"])}
        contest=contests.setdefault(title,{"title":title,**accounting,"over_votes":None,"choices":{}})
        if any(contest[k]!=v for k,v in accounting.items()) or name in contest["choices"]:
            raise ElectionReportError("summary duplicate choice or conflicting district totals")
        vc,mail,total=(integer(row[k]) for k in ("Vote Center Ballots Votes","Vote-by-Mail Ballots Votes","Total Votes"))
        if vc+mail!=total or not 1<=accounting["vote_for"]<=10:
            raise ElectionReportError("summary voting-method/ballot instruction invalid")
        contest["choices"][name]={"name":name,"source_label":row["Candidate Name"],"party":row["Party"],"vote_center":vc,"mail":mail,"votes":total}
    return {"schema_version":1,"format":"oc_enr_final_summary_csv_v1","sha256":hashlib.sha256(raw).hexdigest(),
            "total_precincts":total_precincts,"source_ballots":max(c["ballots"] for c in contests.values()),"contests":contests,
            "qualification":"Official vendor export linked from registrar archive; original certification and later recount references must remain separate."}
