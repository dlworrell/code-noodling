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
    def test_audited_assessor_and_ld42_sensitivities_are_reconstructed(self):
        from election_reporting.audit_sensitivity import reconstruct
        result=reconstruct(SCOPE/"data/elections/washington/2026-08-04")
        assessor=result["assessor"]["rows"]
        self.assertEqual([r["hypothetical_remaining_pair_votes"] for r in assessor],[0,44515,87279,121278])
        for row,reference in zip(assessor,[0,.269,.350,.375]):
            self.assertAlmostEqual(row["legacy_normal_probability"],reference,delta=.0005)
        self.assertAlmostEqual(assessor[2]["stable_finite_tail_probability"],.35026265,delta=1e-8)
        self.assertTrue(assessor[0]["legacy_zero_row_is_not_official_zero"])
        ld=result["ld42"]
        for row,reference in zip(ld["rows"],[.487,.432,.563,.687]):
            self.assertAlmostEqual(row["conditional_probability"],reference,delta=.0005)
        weights=ld["disjoint_effective_weights"]
        self.assertEqual(weights["historical_observed_votes"],26037)
        self.assertEqual(weights["latest_verified_pair_votes"],209)
        self.assertEqual(weights["overlap_votes_reused"],0)
        self.assertLess(weights["latest_raw_fraction"],.008)
        self.assertEqual(weights["latest_effective_weight_fraction"],.4)
        self.assertIn("Post-event audit observation",result["clock_qualification"])
        self.assertIn("Neither table is a confidence interval",result["interpretation"])

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
    def transition_race(self, rows, rule="winner", middle_stage=None, prior_changes=None):
        """Exercise actual CSV ingestion and the public analyzer, not its guard."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            for index, text in enumerate(rows, 5):
                (root / f"inputs/2026-08-{index:02}.csv").write_text(text)
            cfg = root / "election.toml"
            cfg.write_text('schema_version=1\n[election]\ntitle="transition regression"\n[[sources]]\nid="local"\npath="inputs"\njurisdiction="Local"\nscope="county"\npriority=1\nexpected_final_ballots=300\n')
            config = counting_config(cfg)
            # Equal bytes have one hash-bound status record, regardless of how
            # many retrieval filenames are present.
            config = replace(config, result_statuses=tuple({
                (r.source_id, r.snapshot_sha256, r.contest_id): r
                for r in config.result_statuses}.values()))
            config.audit_contract["rules"][0]["rule"] = rule
            if prior_changes:
                config.audit_contract["source_versions"][0].update(prior_changes)
            if middle_stage:
                digest = read_snapshot(root / "inputs/2026-08-06.csv", "local").sha256
                config = replace(config, result_statuses=tuple(
                    replace(r, status=replace(r.status, **middle_stage))
                    if r.snapshot_sha256 == digest else r for r in config.result_statuses))
            return analyze_election(config).races[0]

    def test_whole_contest_revisions_cannot_be_new_pair_or_majority_votes(self):
        old = result_rows("Local", "9", 150, [("A",80),("B",60),("C",10)])
        cases = [
            (160, [("A",90),("B",65),("C",5)], "candidate tally decreased"),
            (145, [("A",85),("B",60),("C",0)], "denominator decreased"),
            (155, [("A",90),("B",65)], "Candidate inventory changed"),
            (170, [("A",90),("B",65),("C",10),("D",5)], "Candidate inventory changed"),
            (150, [("A",85),("B",60),("C",10)], "increments exceed"),
        ]
        for rule in ("winner", "wa_judicial"):
            for ballots, choices, reason in cases:
                with self.subTest(rule=rule, choices=choices, ballots=ballots):
                    race = self.transition_race([old, result_rows("Local","9",ballots,choices)], rule)
                    self.assertFalse(race.forecast_eligible)
                    self.assertTrue(race.evidence["quarantined"])
                    self.assertIn(reason, " ".join(race.evidence["batch_transition"]["reasons"]))
                    for decision in race.decisions:
                        self.assertEqual(decision.batch_evidence["latest_verified_pair_votes"], 0)
                        self.assertIsNone(decision.risk.latest_batch_votes)
                        self.assertIsNone(decision.risk.change_probability)

    def test_identical_latest_source_does_not_reuse_an_older_increment(self):
        old = result_rows("Local","9",80,[("A",50),("B",30)])
        current = result_rows("Local","9",100,[("A",60),("B",40)])
        for rule in ("winner", "wa_judicial"):
            race = self.transition_race([old,current,current], rule)
            self.assertTrue(race.forecast_eligible)  # cumulative scenario only
            transition = race.evidence["batch_transition"]
            self.assertEqual(transition["previous_snapshot_sha256"], transition["current_snapshot_sha256"])
            self.assertIn("Identical source bytes", " ".join(transition["reasons"]))
            for decision in race.decisions:
                self.assertEqual(decision.batch_evidence["latest_verified_pair_votes"],0)
                self.assertIsNone(decision.risk.latest_batch_votes)

    def test_missing_or_incompatible_immediate_predecessor_is_not_skipped(self):
        old = result_rows("Local","9",80,[("A",50),("B",30)])
        current = result_rows("Local","9",100,[("A",60),("B",40)])
        cases = [([old,result_rows("Other","99",90,[("A",55),("B",35)]),current],None),
                 ([old,result_rows("Local","9",90,[("A",55),("B",35)]),current],{"result_version":"corrected"})]
        for rows, stage in cases:
            race = self.transition_race(rows, middle_stage=stage)
            self.assertEqual(race.decisions[0].batch_evidence["latest_verified_pair_votes"],0)
            self.assertNotEqual(race.evidence["batch_transition"]["classification"],"new_ballots")

    def test_compatible_complete_increment_retains_disjoint_pair_and_majority_weights(self):
        old = result_rows("Local","9",150,[("A",80),("B",60),("C",10)])
        current = result_rows("Local","9",170,[("A",90),("B",65),("C",15)])
        race = self.transition_race([old,current], "wa_judicial")
        self.assertTrue(race.forecast_eligible)
        self.assertEqual(race.evidence["batch_transition"]["choice_deltas"],{"A":10,"B":5,"C":5})
        pair, majority = race.decisions
        # Judicial pair is the selected second/third-place boundary B/C.
        self.assertEqual(pair.batch_evidence["latest_verified_pair_votes"],10)
        self.assertEqual(majority.batch_evidence["latest_verified_pair_votes"],20)
        for decision in (pair,majority):
            self.assertEqual(decision.batch_evidence["overlap_votes_reused"],0)
            self.assertIsNotNone(decision.risk.change_probability)

    def test_unverified_previous_coverage_or_clock_supplies_no_batch(self):
        rows = [result_rows("Local","9",80,[("A",50),("B",30)]),
                result_rows("Local","9",100,[("A",60),("B",40)])]
        for changes in ({"coverage_complete":False}, {"source_published_at":None},
                        {"source_published_at":"2026-08-06T00:00:00Z"},
                        {"source_published_at":"2026-08-07T00:00:00Z"}):
            race = self.transition_race(rows, prior_changes=changes)
            self.assertEqual(race.decisions[0].batch_evidence["latest_verified_pair_votes"],0)
            self.assertNotEqual(race.evidence["batch_transition"]["classification"],"new_ballots")

    def test_known_counter_revisions_and_changed_availability_reject_new_votes(self):
        def counted(ballots, choices, over, under):
            rows = result_rows("Local","9",ballots,choices).splitlines()
            return rows[0] + ',"Contest Overvotes","Contest Undervotes"\n' + "\n".join(
                row + f',"{over}","{under}"' for row in rows[1:]) + "\n"
        old = counted(160,[("A",80),("B",60),("C",10)],0,10)
        for over,under in ((0,5),("","")):
            race = self.transition_race([old,counted(170,[("A",90),("B",65),("C",10)],over,under)])
            self.assertFalse(race.forecast_eligible)
            self.assertEqual(race.decisions[0].batch_evidence["latest_verified_pair_votes"],0)
        positive = self.transition_race([old,counted(170,[("A",85),("B",65),("C",10)],0,10)])
        self.assertTrue(positive.forecast_eligible)
        self.assertEqual(positive.decisions[0].batch_evidence["latest_verified_pair_votes"],10)

    def test_gate_matrix_retains_tallies_and_withholds_unqualified_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"inputs").mkdir()
            (root/"inputs/2026-08-06.csv").write_text(result_rows("Local","9",100,[("A",51),("B",49)]))
            cfg=root/"election.toml"
            cfg.write_text('schema_version=1\n[election]\ntitle="t"\n[[sources]]\nid="local"\npath="inputs"\njurisdiction="Local"\nscope="county"\npriority=1\nexpected_final_ballots=120\n')
            cases=[("source_versions","source_published_at",None),
                   ("source_versions","source_published_at","2025-01-01T00:00:00Z"),
                   ("source_versions","source_published_at","2027-01-01T00:00:00Z"),
                   ("source_versions","coverage_complete",False),
                   ("source_versions","source_url",None),("source_versions","retrieved_at",None),
                   ("identities","coverage_complete",False),("identities","jurisdictions",[]),
                   ("identities","role","partial"),("rules","passage_numerator",3),
                   ("rules","passage_denominator",5),("rules","validation_requirements",["quorum"]),
                   ("rules","rule","rcv"),("remaining_pools","as_of","2025-01-01T00:00:00Z"),
                   ("remaining_pools","jurisdictions",["Different county"]),
                   ("remaining_pools","coverage_complete",False),("remaining_pools","participation_basis",None)]
            cases.append(("remaining_pools","category_basis",None))
            for table,field,value in cases:
                with self.subTest(table=table,field=field,value=value):
                    config=counting_config(cfg);config.audit_contract[table][0][field]=value
                    analysis=analyze_election(config);race=analysis.races[0]
                    self.assertEqual(race.valid_votes,100)
                    self.assertFalse(race.scientific_eligibility["eligible"])
                    self.assertIsNone(race.decisions[0].risk.change_probability)
                    self.assertIn("FORECASTS WITHHELD",facebook_summary(analysis))
            valid=analyze_election(counting_config(cfg)).races[0]
            self.assertTrue(valid.scientific_eligibility["eligible"])
            self.assertFalse(valid.scientific_eligibility["validated_forecast_eligible"])

    def test_corrections_identical_counts_and_negative_deltas_are_not_new_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"inputs").mkdir()
            first=root/"inputs/2026-08-05.csv";last=root/"inputs/2026-08-06.csv"
            first.write_text(result_rows("Local","9",80,[("A",50),("B",30)]))
            last.write_text(result_rows("Local","9",100,[("A",60),("B",40)]))
            cfg=root/"election.toml"
            cfg.write_text('schema_version=1\n[election]\ntitle="t"\n[[sources]]\nid="local"\npath="inputs"\njurisdiction="Local"\nscope="county"\npriority=1\nexpected_final_ballots=120\n')
            config=counting_config(cfg)
            self.assertEqual(analyze_election(config).races[0].decisions[0].batch_evidence["latest_verified_pair_votes"],20)
            for classification in ("correction","recount","unknown"):
                config=counting_config(cfg)
                config.audit_contract["source_versions"][-1]["delta_kind"]=classification
                decision=analyze_election(config).races[0].decisions[0]
                self.assertEqual(decision.batch_evidence["latest_verified_pair_votes"],0)
                self.assertIsNone(decision.risk.latest_batch_votes)
            for choices in ([("A",50),("B",30)],[("A",49),("B",51)]):
                last.write_text(result_rows("Local","9",sum(v for _,v in choices),choices))
                decision=analyze_election(counting_config(cfg)).races[0].decisions[0]
                self.assertEqual(decision.batch_evidence["latest_verified_pair_votes"],0)
                self.assertIsNone(decision.risk.latest_batch_votes)

    def test_office_and_election_rule_mismatches_withhold_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"inputs").mkdir()
            (root/"inputs/2026-08-06.csv").write_text(result_rows("Local","9",100,[("A",51),("B",49)]))
            cfg=root/"election.toml"
            cfg.write_text('schema_version=1\n[election]\ntitle="t"\n[[sources]]\nid="local"\npath="inputs"\njurisdiction="Local"\nscope="county"\npriority=1\nexpected_final_ballots=120\n')
            config=counting_config(cfg)
            self.assertIsNotNone(analyze_election(config).races[0].decisions[0].risk.change_probability)
            for changes in ({"office":"Different office"},{"election_type":"general"},{"office":None},{"election_type":None}):
                config=counting_config(cfg)
                config.audit_contract["rules"][0].update(changes)
                race=analyze_election(config).races[0]
                self.assertFalse(race.evidence["rule"]["scope_matches"])
                self.assertIsNone(race.decisions[0].risk.change_probability)

    def test_selected_pair_cannot_represent_all_multi_candidate_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"inputs").mkdir()
            cfg=root/"election.toml"
            cfg.write_text('schema_version=1\n[election]\ntitle="t"\n[[sources]]\nid="local"\npath="inputs"\njurisdiction="Local"\nscope="county"\npriority=1\nexpected_final_ballots=300\n')
            for cid,choices,final_choices in (("9",[("A",80),("B",60),("C",50)],{"A":80,"B":60,"C":90}),
                                             ("10",[("A",80),("B",60),("C",50),("D",49)],{"A":80,"B":60,"C":50,"D":89})):
                (root/"inputs/2026-08-06.csv").write_text(result_rows("Local",cid,sum(v for _,v in choices),choices))
                race=analyze_election(counting_config(cfg)).races[0]
                decision=race.decisions[0]
                self.assertFalse(decision.event["whole_outcome_probability"])
                self.assertEqual(decision.event["scope"],"selected_pair")
                self.assertLess(final_choices[decision.change_side],final_choices[decision.current_side])
                seats=1 if cid=="9" else 2
                self.assertNotEqual({n for n,_ in choices[:seats]},set(sorted(final_choices,key=final_choices.get,reverse=True)[:seats]))

    def test_short_summary_carries_clocks_coverage_and_pool_sensitivity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"inputs").mkdir()
            (root/"inputs/2026-08-06.csv").write_text(result_rows("Local","9",100,[("A",51),("B",49)]))
            cfg=root/"election.toml"
            cfg.write_text('schema_version=1\n[election]\ntitle="t"\n[[sources]]\nid="local"\npath="inputs"\njurisdiction="Local"\nscope="county"\npriority=1\nexpected_final_ballots=120\n')
            config=counting_config(cfg)
            config.audit_contract["remaining_pools"][0].update(low=10,base=20,high=40)
            analysis=analyze_election(config)
            social=facebook_summary(analysis)
            for phrase in ("Source publication range:","known source-age range:","unknown count-update time: 1", "incomplete/unverified coverage: 0", "low/base/high 10/20/40", "as of 2026-08-06", "Contest/pair participation: 1/1.0", "R=10:", "R=40:", "not a confidence interval", "Selected pair: B strictly overtakes A", "input SHA-256"):
                self.assertIn(phrase,social)
            version=config.audit_contract["source_versions"][0]
            version.update(source_published_at=None,coverage_complete=False)
            withheld=facebook_summary(analyze_election(config))
            self.assertIn("Unknown source age: 1",withheld)
            self.assertIn("incomplete/unverified coverage: 1",withheld)
            self.assertIn("FORECASTS WITHHELD",withheld)

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

    def test_judicial_tie_is_unresolved_and_duplicate_crosswalks_are_refused(self):
        from election_reporting.evidence import validate_contract
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"inputs").mkdir()
            (root/"inputs/2026-08-06.csv").write_text(result_rows("Justice Position 1","117",100,[("A",50),("B",50)]))
            cfg=root/"election.toml"
            cfg.write_text('schema_version=1\n[election]\ntitle="t"\n[[sources]]\nid="local"\npath="inputs"\njurisdiction="Local"\nscope="state"\npriority=1\nexpected_final_ballots=120\n')
            config=counting_config(cfg)
            race=analyze_election(config).races[0]
            self.assertEqual(race.data_status,"unresolved_tie")
            self.assertIsNone(race.decisions[0].risk.change_probability)
            contract=config.audit_contract
            ident=contract["identities"][0]
            ident["area_aliases"]=[{"contest_id":"117","name":"Justice Position 1","jurisdiction":"Local"}]*2
            with self.assertRaisesRegex(ElectionReportError,"duplicate reviewed area alias"):
                validate_contract(contract,{"local"})
            ident["area_aliases"]=[]
            contract["identities"].append(dict(ident,contest_id="118"))
            with self.assertRaisesRegex(ElectionReportError,"only one reviewed controlling"):
                validate_contract(contract,{"local"})

    def test_detail_only_candidate_is_a_contradiction(self):
        from election_reporting.evidence import geographic_comparison
        from election_reporting.areas import AreaContest,AreaSnapshot
        from election_reporting.model import ContestResult,ChoiceResult
        contest=ContestResult("1","Reviewed contest",1,100,100,0,0,(ChoiceResult("A",60,1),ChoiceResult("B",40,2)),"reviewed")
        detail=AreaContest("1","Reviewed contest",areas={"County":{"A":60,"B":40,"C":5}})
        snapshot=AreaSnapshot("state","county",Path("2026-08-06.csv"),datetime(2026,8,6),{("1","Reviewed contest"):detail})
        ident={"reviewed":True,"jurisdictions":["County"],"area_aliases":[{"contest_id":"1","name":"Reviewed contest","jurisdiction":"County"}]}
        result=geographic_comparison(contest,ident,{"publication_id":"same"},[(snapshot,{"publication_id":"same"})])
        self.assertEqual(result["status"],"contradictory")
        self.assertEqual(result["differences"]["c"],{"headline":0,"geography":5})

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
    def test_authoritative_selection_preserves_partial_accounting_and_lineage(self):
        payload=json.loads((FIXTURES/"official-result-versions.json").read_text())
        records=validate_versions(payload)
        measure=[r for r in records if r["canonical_id"].endswith("measure2")]
        self.assertNotIn("selection",measure[0])
        selected=measure[1]
        self.assertEqual(selected["selection"]["supersedes"],["archive-first-choice"])
        self.assertEqual(selected["selection"]["status"],"authoritative_vote_totals_with_partial_accounting")
        self.assertEqual(selected["ballots"]-selected["continuing"],14564)
        self.assertIsNone(selected["blanks"])
        self.assertIsNone(selected["source_published_at"])
        hd40=next(r for r in records if r["canonical_id"].endswith(":HD40"))
        self.assertEqual(hd40["continuing"]+hd40["blanks"]+hd40["exhausted"]+hd40["overvotes"],3170)
        self.assertEqual(max(hd40["votes"].values())-min(hd40["votes"].values()),533)
        for modification in ("missing_lineage","unknown_selection","raw_selection","bad_partial_accounting","naive_clock"):
            broken=json.loads(json.dumps(payload))
            if modification=="missing_lineage":broken["selections"][0]["supersedes"]=["not-present"]
            elif modification=="unknown_selection":broken["selections"][0]["round_id"]="not-present"
            elif modification=="raw_selection":broken["selections"][-1].update(result_version="me-2024-raw",round_id="0")
            elif modification=="bad_partial_accounting":next(r for r in broken["records"] if r.get("coverage_status")=="partial_accounting")["noncontinuing_total"]-=1
            else:broken["records"][0]["source_published_at"]="2024-11-30T15:21:13"
            with self.subTest(modification=modification),self.assertRaises(ElectionReportError):validate_versions(broken)

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
