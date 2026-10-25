# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:258e62a9-c5b4-4897-b5c3-b66e1edf32f7

"""Independent audit oracles and the named adversarial source regressions."""
import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timezone
from fractions import Fraction
from pathlib import Path

from election_reporting.analysis import analyze_election, _tempered_beta_binomial_probability
from election_reporting.config import load_config
from election_reporting.evidence import hydrate_snapshots, timestamp, identity
from election_reporting.ingest import read_snapshot
from election_reporting.model import ElectionReportError
from election_reporting.numerics import beta_binomial_tail, posterior, threshold, ERROR_POLICY
from election_reporting.publish import analysis_payload, facebook_summary, build_edom
from election_reporting.rcv import allocate_patterns
from election_reporting.validation import evaluate, protocol_hash
from election_reporting.versions import validate_versions
from election_reporting.watch.maine import governing_rcv
from election_reporting.watch.store import ElectionStore
from test_election_reporting import result_rows, counting_config

FIXTURES = Path(__file__).parent / "fixtures"
SCOPE = Path(__file__).resolve().parents[1]


class NumericalAuditTests(unittest.TestCase):
    def test_1080_integer_thresholds_and_rational_distribution_oracles(self):
        checks = 0
        for n in range(1, 19):
            for margin in range(6):
                for tie_changes in (False, True):
                    for alpha, beta in [(1,1),(1,2),(2,1),(2,3),(4,2)]:
                        first = threshold(n, margin, tie_changes)
                        expected = Fraction(0)
                        for k in range(n+1):
                            # Direct final-margin inequality independent of threshold formula.
                            changes = 2*k-n >= margin if tie_changes else 2*k-n > margin
                            if changes:
                                expected += Fraction(math.comb(n,k)*math.factorial(k+alpha-1)*math.factorial(n-k+beta-1)*math.factorial(alpha+beta-1), math.factorial(n+alpha+beta-1)*math.factorial(alpha-1)*math.factorial(beta-1))
                        self.assertAlmostEqual(beta_binomial_tail(n,alpha,beta,first),float(expected),places=12)
                        checks += 1
        self.assertEqual(checks,1080)

    def test_large_skewed_and_boundary_tails_against_independent_oracle(self):
        for case in json.loads((FIXTURES/"beta-binomial-oracles.json").read_text())["cases"]:
            actual = beta_binomial_tail(case["n"],case["alpha"],case["beta"],case["first"])
            tolerance = max(ERROR_POLICY["absolute_tolerance"],abs(case["tail"])*ERROR_POLICY["relative_tolerance"])
            self.assertAlmostEqual(actual,case["tail"],delta=tolerance)
        actual,_ = _tempered_beta_binomial_probability(0,1,5001,1,None,None,False)
        self.assertAlmostEqual(actual,0.24985007,places=7)
        self.assertIsNone(beta_binomial_tail(200001,1,2,100000))

    def test_latest_batch_is_not_reused_as_historical_likelihood(self):
        # History 40/60 followed by a disjoint batch 10/90: posterior 51/151.
        self.assertEqual(posterior(50,150,.1,100),(51,151))
        self.assertEqual(posterior(50,150), (51,151))
        # Caps distinguish the disjoint model at larger sample sizes.
        self.assertEqual(posterior(500,1500,.1,1000),(141,361))
        self.assertEqual(posterior(5,5,.9,100),posterior(5,5))  # impossible increment refused


class SourceAuditTests(unittest.TestCase):
    def test_pool_sensitivity_zero_unknown_tie_and_staleness(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"inputs").mkdir()
            path=root/"inputs/2026-08-06.csv"
            path.write_text(result_rows("Local","9",100,[("Yes",51),("No",49)]))
            cfg=root/"election.toml"
            cfg.write_text('schema_version=1\n[election]\ntitle="t"\n[[sources]]\nid="local"\npath="inputs"\njurisdiction="Local"\nscope="county"\npriority=1\nexpected_final_ballots=120\n')
            config=counting_config(cfg)
            pool=config.audit_contract["remaining_pools"][0]
            pool.update(low=10,base=20,high=40)
            race=analyze_election(config).races[0]
            sensitivity=race.decisions[0].numerical_evidence["remaining_pool_sensitivity"]
            self.assertEqual([r["remaining_ballots"] for r in sensitivity],[10,20,40])
            self.assertGreater(len({r["conditional_probability"] for r in sensitivity}),1)
            pool.update(kind="official",ballots=0,categories=[{"category":"accepted outstanding","ballots":0}],ballot_unit="accepted_uncounted",acceptance_assumption="No unaccepted or cure-pending ballots in this synthetic fixture",authority_url="https://example.test/pool")
            self.assertEqual(analyze_election(config).races[0].decisions[0].risk.change_probability,0)
            pool["as_of"]="2025-01-01T00:00:00Z"
            self.assertIsNone(analyze_election(config).races[0].decisions[0].risk.change_probability)
            path.write_text(result_rows("Local","9",100,[("Yes",50),("No",50)]))
            config=counting_config(cfg)
            config.audit_contract["remaining_pools"][0].update(kind="official",ballots=0,categories=[{"category":"accepted outstanding","ballots":0}],ballot_unit="accepted_uncounted",acceptance_assumption="No unaccepted or cure-pending ballots in this synthetic fixture",authority_url="https://example.test/pool")
            race=analyze_election(config).races[0]
            self.assertEqual(race.data_status,"unresolved_tie")
            self.assertIsNone(race.decisions[0].risk.change_probability)
            path.write_text(result_rows("Local","9",0,[("Yes",0),("No",0)]))
            race=analyze_election(counting_config(cfg)).races[0]
            self.assertEqual(race.data_status,"no_reported_results")
            self.assertIsNone(race.decisions[0].risk.change_probability)

    def test_named_actual_washington_findings_are_preserved(self):
        analysis = analyze_election(load_config(SCOPE/"data/elections/washington/2026-08-04/election.toml"))
        by_source = {(r.source_id,r.contest_id):r for r in analysis.races}
        self.assertEqual(len(analysis.races),298)  # explicit source representations, not guessed deduplication
        for cid in ("17","152"):
            self.assertEqual(by_source[("king-county",cid)].data_status,"source_vintage_conflict")
        south = by_source[("washington","7154")]
        self.assertEqual(south.data_status,"contradictory")
        self.assertEqual(sum(v for k,v in south.evidence["geography_comparison"]["arithmetic_sum"].items() if k in {"approved","rejected"}),1595)
        snoqualmie = [r for r in analysis.races if r.canonical_id.endswith("snoqualmie-pass-benefit-charge")]
        self.assertEqual(len(snoqualmie),2)
        self.assertEqual(sum(r.controlling for r in snoqualmie),1)
        self.assertEqual(next(r for r in snoqualmie if r.controlling).valid_votes,206)
        self.assertTrue(all(d.risk.change_probability is None for r in analysis.races for d in r.decisions))
        payload = analysis_payload(analysis)
        self.assertEqual(payload["schema_version"],4)
        self.assertGreater(len(payload["source_summary"]["vintages"]),1)
        self.assertIn("count_updated_at",str(payload))
        self.assertIn("contradictory",str(build_edom(analysis)))
        self.assertIn("uncalibrated",facebook_summary(analysis))

    def test_unknown_times_and_duplicate_retrieval_cannot_freshen_count(self):
        self.assertIsNone(timestamp("2024-11-06T12:00:00"))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); inputs=root/"inputs";inputs.mkdir()
            (inputs/"2026-08-06.csv").write_text(result_rows("Local","9",100,[("Yes",60),("No",40)]))
            cfg=root/"election.toml"
            cfg.write_text('schema_version=1\n[election]\ntitle="t"\n[[sources]]\nid="local"\npath="inputs"\njurisdiction="Local"\nscope="county"\npriority=1\n')
            config=counting_config(cfg); source=config.sources[0];snap=read_snapshot(inputs/"2026-08-06.csv","local")
            original=config.audit_contract["source_versions"][0]["source_published_at"]
            store=ElectionStore(root);store.observations=[{"place":"local","input_sha256":snap.sha256,"retrieved_at":"2026-08-08T15:00:00Z"}];store.save()
            [hydrated]=hydrate_snapshots(config,source,(snap,))
            self.assertEqual(hydrated.source_metadata["source_published_at"],original.replace("+00:00","+00:00"))
            self.assertEqual(hydrated.source_metadata["retrieved_at"],"2026-08-08T15:00:00+00:00")
            self.assertIsNone(hydrated.source_metadata["count_updated_at"])
            contest=next(iter(snap.contests.values()))
            self.assertFalse(identity(config,source,replace(contest,name="unrelated reused ID"))["reviewed"])

    def test_frozen_539_record_replay_is_descriptive_ingestion(self):
        manifest=json.loads((FIXTURES/"replay-2024/manifest.json").read_text())
        total=0
        with tempfile.TemporaryDirectory() as directory:
            for row in manifest["inputs"]:
                raw=(FIXTURES/"replay-2024"/row["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),row["sha256"])
                path=Path(directory)/"2026-10-23-0000.csv";path.write_bytes(raw)
                snapshot=read_snapshot(path,row["state"])
                self.assertEqual(len(snapshot.contests),row["records"])
                self.assertTrue(all(c.result_status.lifecycle=="unknown" for c in snapshot.contests.values()))
                total+=len(snapshot.contests)
        self.assertEqual(total,539)

    def test_stage_aware_full_replay_withholds_every_forecast(self):
        from election_reporting.replay import run
        records=run(FIXTURES)
        self.assertEqual([len(a.races) for a in records],[204,73,262])
        self.assertTrue(all(r.result_status.tabulation_stage != "unknown" for a in records for r in a.races))
        self.assertTrue(all(d.risk.change_probability is None for a in records for r in a.races for d in r.decisions))
        self.assertEqual(len(records[1].official_versions),12)
        self.assertEqual(len(records[2].official_versions),2)

    def test_maine_office_rule_does_not_follow_page_group(self):
        self.assertTrue(governing_rcv("United States Representative District 2",date(2024,11,5)))
        self.assertTrue(governing_rcv("President",date(2024,11,5)))
        self.assertFalse(governing_rcv("State Senator District 1",date(2024,11,5)))
        self.assertTrue(governing_rcv("Governor - Democratic",date(2026,6,9)))
        self.assertFalse(governing_rcv("Sheriff",date(2026,6,9)))


class RcvAuditTests(unittest.TestCase):
    def test_all_six_maine_exports_conserve_and_reproduce_final_totals(self):
        patterns=json.loads((FIXTURES/"maine-2024-cd2-patterns.json").read_text())
        manifest=json.loads((FIXTURES/"maine-2024-cd2-manifest.json").read_text())
        result=allocate_patterns(patterns["patterns"],manifest["finalists"],noncontinuing=manifest["noncontinuing"],rule_version=manifest["rule_version"])
        self.assertEqual(len(manifest["files"]),6)
        self.assertEqual(sum(f["ballots"] for f in manifest["files"]),402936)
        self.assertEqual(result["totals"],{"Golden, Jared Forrest":197151,"Theriault, Austin":194445})
        self.assertEqual(result["continuing"],391596)
        self.assertEqual(result["exhausted"],{"skipped_ranks":10673,"no_continuing_choice":388,"overvote":279})
        self.assertEqual(result["margin"],2706)
        self.assertFalse(result["independent_eligibility_engine"])

    def test_skips_duplicates_overvotes_and_unknown_eligibility(self):
        rows=[{"ranks":["undervote","A","A"],"count":2},{"ranks":["undervote","undervote","B"],"count":3},{"ranks":["overvote","A"],"count":4},{"ranks":["Write-in","B"],"count":5}]
        result=allocate_patterns(rows,["A","B"],noncontinuing=["Write-in"],rule_version="synthetic-reviewed")
        self.assertEqual(result["totals"],{"A":2,"B":5})
        self.assertEqual(result["ballots"],14)
        with self.assertRaises(ElectionReportError):
            allocate_patterns([{"ranks":["unknown write-in"],"count":1}],["A","B"],noncontinuing=[],rule_version="test")

    def test_official_versions_keep_recount_and_missing_coverage_separate(self):
        payload=json.loads((FIXTURES/"official-result-versions.json").read_text())
        records=validate_versions(payload)
        self.assertEqual(len({r["canonical_id"] for r in records if r["canonical_id"].startswith("ak:") and r["canonical_id"]!="ak:2024-general:measure2"}),9)
        measure=[r for r in records if r["canonical_id"].endswith("measure2")]
        self.assertEqual([r["margin"] for r in measure],[737,743])
        broken=json.loads(json.dumps(payload));broken["records"][0]["ballots"]-=1
        with self.assertRaisesRegex(ElectionReportError,"conservation"):
            validate_versions(broken)


class ValidationAuditTests(unittest.TestCase):
    def test_empty_and_final_only_data_cannot_claim_a_backtest(self):
        protocol=json.loads((FIXTURES/"validation-protocol-v1.json").read_text())
        self.assertEqual(evaluate([],protocol)["status"],"unvalidated")
        with self.assertRaises(ElectionReportError): evaluate([{}],protocol)

    def test_grouped_scores_and_leakage_refusal(self):
        protocol=json.loads((FIXTURES/"validation-protocol-v1.json").read_text())
        row=dict(protocol_sha256=protocol_hash(protocol),model_id=protocol["model_id"],information_time="2028-11-08T00:00:00Z",outcome_published_at="2028-12-01T00:00:00Z",split="held_out",used_for_tuning=False,remaining_pool_status="official_outstanding",event_scope=protocol["event_scope"],input_sha256="a"*64,election_id="synthetic:2028",canonical_id="synthetic:district",jurisdiction_id="synthetic:county",probability=.25,outcome=0,baselines={x:.5 for x in protocol["baselines"]})
        result=evaluate([row],protocol)
        self.assertEqual(result["scores"]["model"]["group_weighted_brier"],.0625)
        self.assertEqual(result["independent_groups"],1)
        bad=dict(row,used_for_tuning=True)
        with self.assertRaisesRegex(ElectionReportError,"ineligible"):evaluate([bad],protocol)
        with self.assertRaisesRegex(ElectionReportError,"duplicate"):evaluate([row,row],protocol)
        leaked=dict(protocol,training_jurisdictions=[row["jurisdiction_id"]])
        with self.assertRaisesRegex(ElectionReportError,"leaks"):
            evaluate([dict(row,protocol_sha256=protocol_hash(leaked))],leaked)
