# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:f205d255-349d-42c2-811a-32d627946360

"""Two-phase, frozen-model retrospective *scenario* experiment.

This does not relax validation.evaluate's prospective contract, manufacture
official outstanding ballots, or classify tally deltas as new-ballot batches.
Prediction generation cannot read the reserved certified-result files. Scoring
is a separate command over the already sealed predictions and later outcomes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .archives import king_snapshot
from .evidence import timestamp
from .model import ElectionReportError
from .numerics import MAX_REMAINING, beta_binomial_tail, posterior, threshold
from .validation import protocol_hash

SCOPE = Path(__file__).resolve().parents[1]
MODE = "retrospective_conditional_scenarios"
BASELINES = ("fixed_half", "cumulative_share", "untempered_beta_binomial",
             "scenario_margin_lock", "verified_latest_batch_only", "official_pool_margin_lock")
QUALIFICATION = ("Reconstructed after the outcomes under a frozen recipe; not predictions published then. "
                 "Turnout-derived pools and participation are assumptions, not official outstanding counts. "
                 "Scores compare realized tallies under hypothetical pools, not calibration conditional on those pools being realized. "
                 "No verified new-ballot batches; no production eligibility or calibration claim.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ElectionReportError("experiment path must stay inside its declared root")
    return root / path


def load_protocol(path, scope=SCOPE):
    protocol = json.loads(Path(path).read_text())
    if (protocol.get("schema_version") != 2 or protocol.get("mode") != MODE
            or not timestamp(protocol.get("frozen_at")) or protocol.get("used_for_tuning") is not False
            or set(protocol.get("baselines", [])) != set(BASELINES)):
        raise ElectionReportError("explicit frozen retrospective protocol required")
    if protocol.get("training_elections") != [] or protocol.get("training_jurisdictions") != []:
        raise ElectionReportError("this no-fitting experiment holds the entire jurisdiction out")
    if (protocol.get("turnout_offsets") != {"low":-.05, "base":0, "high":.05}
            or protocol.get("max_archive_age_hours") != 48
            or protocol.get("event_scope") != "selected_pair_strict_reversal"
            or protocol.get("model_id") != "disjoint_tempered_beta_binomial_v2"):
        raise ElectionReportError("unsupported experiment recipe")
    elections = protocol.get("elections", [])
    if not elections or len({r["election_id"] for r in elections}) != len(elections):
        raise ElectionReportError("unique held-out elections required")
    for name, sha in protocol.get("model_files", {}).items():
        if digest(_relative(scope, name)) != sha:
            raise ElectionReportError("frozen model/harness bytes changed; new protocol required")
    if not protocol.get("model_files"):
        raise ElectionReportError("experiment requires frozen model hashes")
    return protocol


def binomial_tail(n, p, first):
    """Bounded full-support normalized recurrence; no normal approximation."""
    if type(n) is not int or n < 0 or not math.isfinite(p) or not 0 <= p <= 1:
        raise ElectionReportError("invalid binomial baseline inputs")
    if first <= 0:
        return 1.0
    if first > n or p == 0:
        return 0.0
    if p == 1:
        return 1.0
    if n > MAX_REMAINING:
        return None
    logs, value = [0.0], 0.0
    odds = math.log(p) - math.log1p(-p)
    for k in range(n):
        value += math.log(n-k) - math.log(k+1) + odds
        logs.append(value)
    maximum = max(logs)
    denominator = math.fsum(math.exp(v-maximum) for v in logs)
    return min(1.0, max(0.0, math.fsum(math.exp(v-maximum) for v in logs[first:])/denominator))


def snapshot_clock(spec, election, max_age_hours):
    """Conservative publisher-archive availability bound, not a count clock."""
    if spec["file"] == election["outcome_file"] or "final" in spec["file"].casefold():
        return None, "reserved_certified_outcome"
    match = re.search(r"(20\d{6})", spec["file"])
    if not match:
        return None, "missing_dated_archive_label"
    try:
        report_day = datetime.strptime(match[1], "%Y%m%d").date()
        modified = parsedate_to_datetime(spec["last_modified"]).astimezone(timezone.utc)
    except (ValueError, TypeError, KeyError):
        return None, "unknown_publication_proxy"
    zone = ZoneInfo("America/Los_Angeles")
    lower = datetime.combine(report_day, time(), zone)
    scheduled = datetime.combine(report_day+timedelta(days=1), time(16,30), zone)
    information = max(scheduled, modified)
    outcome_lower = timestamp(election["certification_time_lower_bound"])
    if modified < lower or information >= outcome_lower:
        return None, "publication_proxy_after_outcome_or_before_archive_day"
    if (information-lower).total_seconds() > max_age_hours*3600:
        return None, "stale_archive_or_late_migration"
    return {"information_time": information.isoformat(), "archive_report_day": report_day.isoformat(),
            "publication_proxy": modified.isoformat(), "count_updated_at": None,
            "availability_qualification": "Daily archive promised by next day 16:30 Pacific; later HTTP Last-Modified retained as a conservative proxy, not authenticated historical capture or count-update time."}, None


def scoped_candidate(identity):
    """Explicit reviewed jurisdictions; never name-merge county/state slices."""
    _, kind, _, district, title = identity
    county_districts = {f"{prefix}Council District No. {i}" for prefix in ("Metropolitan King County ", "County ") for i in range(1,10)}
    city_districts = {f"City of Seattle Council District No. {i}" for i in range(1,8)}
    scope = ((kind == "County" and (district == "King County" or district in county_districts))
             or (kind == "City" and (district == "City of Seattle" or district in city_districts)))
    if not scope:
        return False
    return not any(word in title.casefold() for word in ("proposition", "amendment", "initiative", "advisory", "question"))


class Corpus:
    def __init__(self, fixtures, protocol):
        self.root = Path(fixtures)
        self.manifest_path = self.root/"king-archives-manifest.json"
        if digest(self.manifest_path) != protocol["corpus_manifest_sha256"]:
            raise ElectionReportError("corpus manifest differs from frozen recipe")
        self.manifest = json.loads(self.manifest_path.read_text())
        archive = self.root/"king-archives.zip"
        if digest(archive) != self.manifest["archive_sha256"]:
            raise ElectionReportError("archive hash mismatch")
        self.zip = zipfile.ZipFile(archive)
        self.specs = {s["file"]: s for s in self.manifest["files"]}
        if len(self.specs) != len(self.manifest["files"]) or len(self.specs) > 100:
            raise ElectionReportError("duplicate or over-budget archive inventory")
        if (len(self.zip.infolist()) != len(self.specs) or set(self.zip.namelist()) != set(self.specs)
                or sum(i.file_size for i in self.zip.infolist()) > 32_000_000):
            raise ElectionReportError("archive member inventory/size mismatch")
        self.reads = []

    def read(self, name, allowed):
        if name not in allowed or name not in self.specs:
            raise ElectionReportError("artifact forbidden in this prediction/scoring phase")
        info = self.zip.getinfo(name)
        if info.file_size > 8_000_000:
            raise ElectionReportError("archive member exceeds budget")
        raw = self.zip.read(name)
        if hashlib.sha256(raw).hexdigest() != self.specs[name]["sha256"]:
            raise ElectionReportError("archive member hash mismatch")
        self.reads.append(name)
        return king_snapshot(raw)

    def close(self):
        self.zip.close()


def build_predictions(protocol, fixtures, forecast_sources):
    generated = datetime.now(timezone.utc).isoformat()
    if timestamp(generated) < timestamp(protocol["frozen_at"]):
        raise ElectionReportError("prediction computation precedes protocol freeze")
    if digest(forecast_sources) != protocol["forecast_sources_sha256"]:
        raise ElectionReportError("forecast-source facts differ from frozen recipe")
    forecasts = {r["election_id"]:r for r in json.loads(Path(forecast_sources).read_text())["sources"]}
    elections = {r["election_id"]:r for r in protocol["elections"]}
    corpus = Corpus(fixtures, protocol)
    rows, ledger, seen_bytes = [], [], set()
    reserved = {r["outcome_file"] for r in elections.values()}
    allowed = set(corpus.specs) - reserved
    try:
        # Some 2025 filenames have extra hyphens. Use their dated report label,
        # so a duplicate later snapshot cannot displace its earlier availability.
        def archive_order(spec):
            match = re.search(r"(20\d{6})", spec["file"])
            return spec["election_id"], match[1] if match else "z", spec["file"]
        for spec in sorted(corpus.specs.values(), key=archive_order):
            file_record = {"file":spec["file"], "election_id":spec["election_id"], "source_contests":spec["contests"],
                           "input_sha256":spec["sha256"], "excluded_contests":{}, "candidates":[]}
            ledger.append(file_record)
            election = elections.get(spec["election_id"])
            if election is None:
                file_record["file_exclusion"] = "outside_frozen_elections; 2022 migrated bytes not historical publication evidence"
                continue
            clock, reason = snapshot_clock(spec, election, protocol["max_archive_age_hours"])
            if reason:
                file_record["file_exclusion"] = reason
                continue
            if spec["sha256"] in seen_bytes:
                file_record["file_exclusion"] = "identical_snapshot_bytes_do_not_advance_count_clock"
                continue
            seen_bytes.add(spec["sha256"])
            forecast = forecasts[spec["election_id"]]
            if not timestamp(forecast["source_published_at"]) < timestamp(clock["information_time"]):
                raise ElectionReportError("turnout assumption was published after information time")
            snapshot = corpus.read(spec["file"], allowed)
            contests = snapshot["contests"]
            if len(contests) != spec["contests"]:
                raise ElectionReportError("archive contest inventory changed")
            source_ballots = max(c["district_ballots"] for c in contests.values())
            registered = max(c["registered_voters"] for c in contests.values())
            for key, contest in sorted(contests.items()):
                canonical = spec["election_id"]+":"+hashlib.sha256(key.encode()).hexdigest()[:16]
                def exclude(why):
                    file_record["excluded_contests"].setdefault(why,[]).append(canonical)
                if not scoped_candidate(contest["source_identity"]):
                    exclude("unreviewed_whole_jurisdiction_or_passage_rule")
                    continue
                names = sorted(((name,votes) for name,votes in contest["votes"].items()
                                if name.casefold().replace("-","").replace(" ","") not in {"writein","writeins","writeinvotes"}),
                               key=lambda c:(-c[1],c[0]))
                if len(names) < 2 or names[0][1] == names[1][1] or not source_ballots:
                    exclude("uncontested_tied_or_no_reported_data")
                    continue
                current, change = names[:2]
                margin = current[1]-change[1]
                case = {"canonical_id":canonical,"source_identity":contest["source_identity"],"scenario_exclusions":{}}
                file_record["candidates"].append(case)
                pair_fraction = (current[1]+change[1])/source_ballots
                if not 0 <= pair_fraction <= 1:
                    raise ElectionReportError("pair participation exceeds county ballot denominator")
                for label, offset in protocol["turnout_offsets"].items():
                    rate = forecast["forecast_turnout_fraction"]+offset
                    assumed_final = round(registered*rate)
                    remainder = assumed_final-source_ballots
                    if remainder <= 0:
                        case["scenario_exclusions"][label] = "turnout_target_reached_is_not_official_zero"
                        continue
                    n = round(remainder*pair_fraction)
                    if n <= 0 or n > MAX_REMAINING:
                        case["scenario_exclusions"][label] = "no_positive_supported_pair_pool_or_numerical_budget"
                        continue
                    first = threshold(n,margin,False)
                    alpha,beta = posterior(change[1],current[1])  # no verified new-ballot batches
                    probability = beta_binomial_tail(n,alpha,beta,first)
                    elapsed = (timestamp(clock["information_time"]).astimezone(ZoneInfo("America/Los_Angeles")).date()
                               - date.fromisoformat(election["election_day"])).days
                    rows.append({"election_id":spec["election_id"],"canonical_id":canonical,"source_identity":contest["source_identity"],
                        "source_key":key,"input_file":spec["file"],"input_sha256":spec["sha256"],"source_url":spec["url"],
                        **clock,"retrieved_at":spec["retrieved_at"],"prediction_generated_at":generated,
                        "scenario":label,"assumed_turnout":rate,"assumed_final_source_ballots":assumed_final,
                        "turnout_forecast_authority":forecast["authority_url"],"turnout_forecast_published_at":forecast["source_published_at"],
                        "source_ballots":source_ballots,"remaining_source_ballots":remainder,"pair_participation":pair_fraction,
                        "remaining_pair_votes":n,"current_side":current[0],"change_side":change[0],"current_votes":current[1],"change_votes":change[1],
                        "event_scope":"selected_pair_strict_reversal","probability":probability,
                        "baselines":{"fixed_half":.5,"cumulative_share":binomial_tail(n,change[1]/(current[1]+change[1]),first),
                            "untempered_beta_binomial":beta_binomial_tail(n,1+change[1],1+current[1],first),
                            "scenario_margin_lock":0.0 if first>n else .5,
                            "verified_latest_batch_only":None,"official_pool_margin_lock":None},
                        "posterior":{"alpha":alpha,"beta":beta,"verified_latest_pair_votes":0,"delta_classification":"unknown"},
                        "state":"Washington","contest_type":contest["source_identity"][1],"jurisdiction_cluster":"King County, Washington",
                        "counting_stage":"initial" if elapsed<=2 else "middle" if elapsed<=7 else "late",
                        "missing_data_stratum":"turnout_scenario; acceptance/category and batch provenance unavailable",
                        "operationally_eligible":False,"qualification":QUALIFICATION})
            accounted = sum(len(v) for v in file_record["excluded_contests"].values())+len(file_record["candidates"])
            if accounted != spec["contests"]:
                raise ElectionReportError("inclusion/exclusion ledger lost source records")
        if set(corpus.reads)&reserved:
            raise ElectionReportError("certified outcomes leaked into prediction generation")
        return {"schema_version":2,"mode":MODE,"protocol_sha256":protocol_hash(protocol),"generated_at":generated,
                "rows":rows,"inclusion_exclusion_ledger":ledger,"read_artifacts":corpus.reads,
                "reserved_outcomes_read":[],"qualification":QUALIFICATION}
    finally:
        corpus.close()


def _metrics(rows, name):
    def value(row):
        return row["probability"] if name=="model" else row["baselines"][name]
    if not rows or any(value(r) is None for r in rows):
        return {"status":"NOT RUN","reason":"No verified new-ballot data" if name=="verified_latest_batch_only" else "No official contemporaneous outstanding pool" if name=="official_pool_margin_lock" else "No eligible rows","rows":0}
    elections = defaultdict(lambda:defaultdict(list))
    for row in rows:
        elections[row["election_id"]][row["canonical_id"]].append(row)
    weighted=[]
    for contests in elections.values():
        for snapshots in contests.values():
            for row in snapshots:
                weight=1/(len(elections)*len(contests)*len(snapshots))
                weighted.append((row,weight))
    def losses(row):
        p,y=value(row),row["outcome"]
        q=min(1-1e-12,max(1e-12,p))
        return (p-y)**2, -(y*math.log(q)+(1-y)*math.log1p(-q))
    return {"status":"scored_reconstructed_scenarios","rows":len(rows),"elections":len(elections),
            "brier":math.fsum(losses(r)[0]*w for r,w in weighted),"log_loss":math.fsum(losses(r)[1]*w for r,w in weighted),
            "weights":"equal elections, equal contests within election, equal snapshots within contest",
            "calibration_bins":[{"low":k/10,"high":(k+1)/10,"rows":len(selected),
                "predicted":math.fsum(value(r)*w for r,w in selected)/mass if mass else None,
                "observed":math.fsum(r["outcome"]*w for r,w in selected)/mass if mass else None}
                for k in range(10)
                for selected in [[(r,w) for r,w in weighted if k/10<=value(r)<(k+1)/10 or k==9 and value(r)==1]]
                for mass in [math.fsum(w for r,w in selected)]]}


def score_predictions(protocol, predictions_path, fixtures, expected_predictions_sha256):
    raw=Path(predictions_path).read_bytes()
    if len(raw)>16_000_000:
        raise ElectionReportError("predictions exceed experiment size budget")
    if hashlib.sha256(raw).hexdigest() != expected_predictions_sha256:
        raise ElectionReportError("predictions changed after their pre-scoring seal")
    predictions=json.loads(raw)
    frozen=timestamp(protocol["frozen_at"])
    if predictions.get("mode")!=MODE or predictions.get("protocol_sha256")!=protocol_hash(protocol):
        raise ElectionReportError("predictions do not match frozen retrospective protocol")
    generated=timestamp(predictions.get("generated_at"))
    if not generated or generated<frozen or predictions.get("reserved_outcomes_read")!=[]:
        raise ElectionReportError("unsealed/early predictions or outcome leakage")
    elections={r["election_id"]:r for r in protocol["elections"]}
    corpus=Corpus(fixtures,protocol)
    reserved={r["outcome_file"] for r in elections.values()}
    if set(predictions.get("read_artifacts",[]))&reserved:
        raise ElectionReportError("prediction phase read a reserved outcome")
    output=[];keys=set()
    try:
        outcomes={e:corpus.read(spec["outcome_file"],reserved) for e,spec in elections.items()}
        for row in predictions["rows"]:
            election=elections[row["election_id"]]
            information=timestamp(row.get("information_time"))
            lower=timestamp(election["certification_time_lower_bound"])
            if not information or not information<lower<frozen or timestamp(row.get("prediction_generated_at"))!=generated:
                raise ElectionReportError("invalid historical availability/outcome/freeze chronology")
            if (row["input_file"] in reserved or row["input_file"] not in predictions["read_artifacts"]
                    or corpus.specs[row["input_file"]]["sha256"]!=row["input_sha256"]
                    or corpus.specs[row["input_file"]]["election_id"]!=row["election_id"]):
                raise ElectionReportError("forecast input lineage invalid")
            clock, reason = snapshot_clock(corpus.specs[row["input_file"]], election, protocol["max_archive_age_hours"])
            canonical = row["election_id"]+":"+hashlib.sha256(row["source_key"].encode()).hexdigest()[:16]
            if (reason or clock["information_time"] != row["information_time"]
                    or row["canonical_id"] != canonical or json.loads(row["source_key"]) != row["source_identity"]
                    or not scoped_candidate(row["source_identity"]) or row["scenario"] not in protocol["turnout_offsets"]
                    or row.get("event_scope") != "selected_pair_strict_reversal"
                    or row.get("operationally_eligible") is not False):
                raise ElectionReportError("forecast scope or frozen availability recipe differs")
            key=(row["canonical_id"],row["input_sha256"],row["scenario"])
            if key in keys:
                raise ElectionReportError("duplicate reconstructed scenario")
            keys.add(key)
            probabilities=[row["probability"],*row["baselines"].values()]
            if (row["probability"] is None or set(row["baselines"])!=set(BASELINES)
                    or any(row["baselines"][name] is None for name in BASELINES[:4])
                    or any(row["baselines"][name] is not None for name in BASELINES[4:])
                    or any(p is not None and (type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1) for p in probabilities)):
                raise ElectionReportError("invalid model/baseline probabilities")
            final=outcomes[row["election_id"]]["contests"].get(row["source_key"])
            if final is None or row["current_side"] not in final["votes"] or row["change_side"] not in final["votes"]:
                raise ElectionReportError("certified selected-pair identity does not reconcile")
            output.append({**row,"outcome":int(final["votes"][row["change_side"]]>final["votes"][row["current_side"]]),
                           "outcome_source":election["outcome_authority"],"outcome_file":election["outcome_file"],
                           "outcome_file_sha256":corpus.specs[election["outcome_file"]]["sha256"]})
        base=[r for r in output if r["scenario"]=="base"]
        scores={name:_metrics(base,name) for name in ("model",*BASELINES)}
        strata={field:{value:_metrics([r for r in base if r[field]==value],"model") for value in sorted({r[field] for r in base})}
                for field in ("election_id","counting_stage","contest_type","state","missing_data_stratum")}
        file_ledger=predictions["inclusion_exclusion_ledger"]
        return {"schema_version":2,"mode":MODE,"status":"exploratory only; original EA-13 validation remains incomplete",
                "protocol_sha256":protocol_hash(protocol),"sealed_predictions_sha256":hashlib.sha256(raw).hexdigest(),
                "scored_base_rows":len(base),"all_scenario_rows":len(output),"unique_contests":len({r["canonical_id"] for r in base}),
                "election_groups":len({r["election_id"] for r in base}),
                "jurisdiction_clusters":len({r["jurisdiction_cluster"] for r in base}),"scores":scores,"strata":strata,
                "scenario_sensitivity":{label:_metrics([r for r in output if r["scenario"]==label],"model") for label in protocol["turnout_offsets"]},
                "uncertainty":{"interval":None,"reason":"All elections share one county; three years are not three independent geographic samples. No defensible dependence-aware confidence interval from one jurisdiction cluster."},
                "coverage":{"predictive_intervals":"NOT RUN: model supplies binary-event probabilities, not outcome intervals",
                            "source_files":len(file_ledger),"source_contest_observations":sum(r["source_contests"] for r in file_ledger),
                            "file_exclusions":dict(Counter(r["file_exclusion"] for r in file_ledger if r.get("file_exclusion")))},
                "inclusion_exclusion_ledger":file_ledger,"scored_rows":output,"outcome_artifacts_read":corpus.reads,
                "production_eligible":False,"calibration_established":False,"qualification":QUALIFICATION}
    finally:
        corpus.close()


def plot_scores(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
    bins=report["scores"]["model"].get("calibration_bins",[])
    active=[b for b in bins if b["rows"]]
    axes[0].plot([0,1],[0,1],color="#9ca3af",linestyle="--",linewidth=1)
    axes[0].scatter([b["predicted"] for b in active],[b["observed"] for b in active],s=45,color="#2563eb")
    axes[0].set(xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel="Mean conditional scenario probability",ylabel="Observed selected-pair reversal frequency",title="Retrospective calibration diagnostic")
    available=[(n,s["brier"]) for n,s in report["scores"].items() if s["status"]!="NOT RUN"]
    axes[1].barh([n.replace("_"," ") for n,_ in available],[s for _,s in available],color="#2563eb")
    axes[1].set(xlabel="Brier score (lower is better)",title="Equal election / contest / snapshot weights")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x",alpha=.15)
    fig.suptitle("Frozen-model turnout scenarios · exploratory; unvalidated",fontsize=13,fontweight="bold")
    fig.text(.5,-.03,f"{report['scored_base_rows']} base rows · {report['election_groups']} elections · one county · no verified new-ballot or official-pool baseline",ha="center",fontsize=9)
    fig.savefig(path,dpi=160,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("predict","score"))
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--fixtures",type=Path,required=True)
    parser.add_argument("--forecast-sources",type=Path)
    parser.add_argument("--predictions",type=Path)
    parser.add_argument("--predictions-sha256",help="Recorded before the separate scoring command")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--plot",type=Path)
    args=parser.parse_args();protocol=load_protocol(args.protocol)
    if args.phase=="predict":
        if not args.forecast_sources:
            parser.error("prediction generation requires --forecast-sources")
        result=build_predictions(protocol,args.fixtures,args.forecast_sources)
    else:
        if not args.predictions or not args.predictions_sha256:
            parser.error("scoring requires sealed --predictions and --predictions-sha256")
        result=score_predictions(protocol,args.predictions,args.fixtures,args.predictions_sha256)
        if args.plot:
            plot_scores(result,args.plot)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"phase":args.phase,"rows":len(result.get("rows",[])) if args.phase=="predict" else result["scored_base_rows"],"status":result.get("status","predictions sealed before outcome access")}))


if __name__=="__main__":
    main()
