# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:81986db1-8a70-47ca-a007-1f1ffae0cbfb

"""Reconstruct the audit's fixed-input sensitivities; never used by publication.

Legacy overlap and normal-tail formulas are comparison subjects only. The
audited original is read statically; no original script is executed here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .config import load_config
from .ingest import read_snapshot
from .model import ElectionReportError
from .numerics import beta_binomial_tail, posterior, posterior_details, threshold


def _tail(n, margin, alpha, beta, *, legacy_normal=False):
    first=threshold(n,margin,False)
    if n==0 or first>n:
        return 0.0
    if legacy_normal and n>5000:
        shape=alpha+beta
        mean=n*alpha/shape
        variance=n*alpha*beta*(shape+n)/(shape*shape*(shape+1))
        return .5*math.erfc((first-.5-mean)/math.sqrt(2*variance))
    return beta_binomial_tail(n,alpha,beta,first)


def _legacy_shapes(change,current,share=None,batch=None):
    total=change+current; weight=min(total,300); p=change/total
    alpha,beta=1+p*weight,1+(1-p)*weight
    if share is not None and batch:
        alpha+=share*min(batch,200);beta+=(1-share)*min(batch,200)
    return alpha,beta


def reconstruct(election_directory: Path):
    config=load_config(election_directory/"election.toml")
    sources={s.source_id:s for s in config.sources}
    king_path=sources["king-county"].path/"2026-08-06-1600.csv"
    earlier=sources["washington"].path/"2026-08-08-0018.csv"
    later=sources["washington"].path/"2026-10-14-1100.csv"
    king=read_snapshot(king_path,"king-county")
    assessor=next(c for c in king.contests.values() if c.contest_id=="17")
    votes={c.name:c.votes for c in assessor.choices}
    current,change=votes["Dominique M Scarimbolo"],votes["Christopher Roberts"]
    if (current,change,assessor.ballots)!=(73546,72333,409531):
        raise ElectionReportError("fixed Assessor source differs from the audited case")
    maximum=max(c.ballots for c in king.contests.values())
    assessor_rows=[]
    for final in (409531,534500,654552,750000):
        # Reconstruct the audited assumption, not an official outstanding count.
        contest_remaining=max(0,round(assessor.ballots*(final/maximum-1)))
        remaining=round(contest_remaining*(current+change)/assessor.ballots)
        alpha,beta=_legacy_shapes(change,current)
        assessor_rows.append({"assumed_final_source_ballots":final,"hypothetical_remaining_pair_votes":remaining,
            "legacy_normal_probability":_tail(remaining,current-change,alpha,beta,legacy_normal=True),
            "stable_finite_tail_probability":_tail(remaining,current-change,alpha,beta),
            "legacy_zero_row_is_not_official_zero":remaining==0})
    def ld(path):
        contest=next(c for c in read_snapshot(path,"washington").contests.values() if c.contest_id=="169896")
        return {c.name:c.votes for c in contest.choices}
    previous,latest=ld(earlier),ld(later)
    lead="Michael Alvarez Shepard";challenger="Eamonn Collins"
    current,change=latest[lead],latest[challenger]
    batch=(current-previous[lead])+(change-previous[challenger]);share=(change-previous[challenger])/batch
    if (current,change,batch)!=(13125,13121,209):
        raise ElectionReportError("fixed LD42 source differs from the audited case")
    ld_rows=[]
    for label,p,count in (("cumulative only",None,None),("latest share -2 percentage points",share-.02,batch),
                           ("legacy overlapping components",share,batch),("latest share +2 percentage points",share+.02,batch)):
        alpha,beta=_legacy_shapes(change,current,p,count)
        ld_rows.append({"scenario":label,"latest_share":p,"conditional_probability":_tail(2800,4,alpha,beta)})
    alternatives={
        "cumulative_only":_legacy_shapes(change,current),
        "latest_only":(1+share*min(batch,200),1+(1-share)*min(batch,200)),
        "legacy_overlap":_legacy_shapes(change,current,share,batch),
        "disjoint_v2":posterior(change,current,share,batch),
        "untempered_cumulative":(1+change,1+current),
    }
    return {"schema_version":1,"status":"diagnostic sensitivity; not calibration or a forecast",
        "fixed_information_time":config.information_time.isoformat(),
        "clock_qualification":"Post-event audit observation time; never claimed as an August pre-outcome forecast time. Each table holds the named historical count fixed.",
        "original_revision":"99c03904a7258994bcbc25c2e73ede722d00c461",
        "comparison_provenance":"Independent reconstruction of the statically inspected original formula, compared with rounded tables in the October 23 audit. Original program not executed.",
        "inputs":[{"path":str(p.relative_to(election_directory)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in (king_path,earlier,later)],
        "assessor":{"snapshot_label":"2026-08-06-1600","current":73546,"challenger":72333,"margin":1213,
            "source_ballots":maximum,"pair_participation":145879/409531,"participation_qualification":"Audited source-wide scaling and fixed observed pair share, not measured future participation.","rows":assessor_rows},
        "ld42":{"current":current,"challenger":change,"margin":4,"hypothetical_remaining_pair_votes":2800,"latest_pair_votes":batch,"latest_pair_share":share,
            "rows":ld_rows,"alternative_constructions":[{"construction":name,"alpha":a,"beta":b,"conditional_probability":_tail(2800,4,a,b)} for name,(a,b) in alternatives.items()],
            "disjoint_effective_weights":posterior_details(change,current,share,batch)},
        "interpretation":"Neither table is a confidence interval. All assumptions are hypothetical, hand-set and unvalidated. Actual primary forecasts remain withheld because the count/recount completed and other evidence gates apply. No construction is selected by its fit to these examples."}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--election",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(reconstruct(args.election),indent=2)+"\n")


if __name__=="__main__":
    main()
