# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:8401b5d8-4760-4352-859c-96cd1010e84f

"""Independent arithmetic and adversarial phase-boundary checks; synthetic data."""
import csv
import hashlib
import io
import json
import math
import tempfile
import unittest
import zipfile
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from election_reporting.archives import KING_COLUMNS
from election_reporting.model import ElectionReportError
from election_reporting.retrospective import (
    BASELINES, MODE, Corpus, _metrics, binomial_tail, build_predictions,
    digest, load_protocol, score_predictions, snapshot_clock,
)


def synthetic_csv(a=60, b=40, *, single=False):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(KING_COLUMNS)
    identities = (("1", "County", "King County", "Executive"),
                  ("2", "City", "City of Seattle", "Mayor"),
                  ("3", "City", "City Outside Scope", "Mayor"),
                  ("4", "County", "King County", "Sheriff"))
    for number, kind, district, title in identities:
        choices = [("A", a)] if number == "4" or single else [("A", a), ("B", b)]
        for order, (name, votes) in enumerate(choices, 1):
            writer.writerow((number, number, kind, "", district, title, str(a+b), "300", order, name, votes))
    return stream.getvalue().encode()


def synthetic_corpus(root, *, final=None, current=None):
    """Reserved outcome may be invalid bytes: predict must never parse it."""
    files = {"webresults-20241105.csv": current or synthetic_csv(),
             "webresults-20241126.csv": final if final is not None else synthetic_csv(80,120)}
    archive = root/"king-archives.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, raw in files.items():
            zipped.writestr(name, raw)
    manifest = {"archive_sha256": digest(archive), "files":[{
        "file":name, "election_id":"wa:king:2024-general", "sha256":hashlib.sha256(raw).hexdigest(),
        "contests":4, "url":"https://example.invalid/"+name, "retrieved_at":"2026-10-24T00:00:00Z",
        "last_modified":"Wed, 06 Nov 2024 23:00:00 GMT" if "1105" in name else "Tue, 26 Nov 2024 23:00:00 GMT",
    } for name, raw in files.items()]}
    manifest_path = root/"king-archives-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    sources = {"sources":[{"election_id":"wa:king:2024-general", "forecast_turnout_fraction":.6,
                           "source_published_at":"2024-10-20T00:00:00Z","authority_url":"https://example.invalid/forecast"}]}
    source_path = root/"sources.json"
    source_path.write_text(json.dumps(sources))
    frozen_source = root/"frozen.py"
    frozen_source.write_text("# synthetic model bytes\n")
    protocol = {"schema_version":2, "mode":MODE, "frozen_at":"2026-01-01T00:00:00Z", "used_for_tuning":False,
                "model_id":"disjoint_tempered_beta_binomial_v2", "event_scope":"selected_pair_strict_reversal",
                "training_elections":[], "training_jurisdictions":[], "baselines":list(BASELINES),
                "model_files":{"frozen.py":digest(frozen_source)}, "max_archive_age_hours":48,
                "turnout_offsets":{"low":-.05,"base":0,"high":.05},
                "forecast_sources_sha256":digest(source_path), "corpus_manifest_sha256":digest(manifest_path),
                "elections":[{"election_id":"wa:king:2024-general", "election_day":"2024-11-05",
                              "outcome_file":"webresults-20241126.csv", "outcome_authority":"https://example.invalid/certified",
                              "certification_time_lower_bound":"2024-11-26T00:00:00-08:00"}]}
    return protocol, source_path


class RetrospectiveTests(unittest.TestCase):
    def test_binomial_matches_exact_rational_oracle_and_budget(self):
        for n in range(21):
            for p in (Fraction(0), Fraction(1,4), Fraction(1,2), Fraction(3,4), Fraction(1)):
                for first in range(n+2):
                    exact = sum((math.comb(n,k)*p**k*(1-p)**(n-k) for k in range(first,n+1)), Fraction(0))
                    self.assertAlmostEqual(binomial_tail(n,float(p),first),float(exact),places=12)
        self.assertIsNone(binomial_tail(200_001,.5,100_001))
        for args in ((-1,.5,0),(2,float("nan"),1),(2,-.1,1)):
            with self.assertRaises(ElectionReportError):
                binomial_tail(*args)

    def test_archive_clock_keeps_delayed_proxy_and_excludes_late_migration(self):
        election = {"outcome_file":"webresults-20241126.csv", "certification_time_lower_bound":"2024-11-26T00:00:00-08:00"}
        spec = {"file":"webresults-20241105.csv", "last_modified":"Wed, 06 Nov 2024 23:00:00 GMT"}
        clock, reason = snapshot_clock(spec,election,48)
        self.assertIsNone(reason)
        self.assertEqual(clock["information_time"],"2024-11-06T16:30:00-08:00")
        self.assertIsNone(clock["count_updated_at"])
        late = dict(spec,last_modified="Thu, 07 Nov 2024 09:00:00 GMT")
        self.assertEqual(snapshot_clock(late,election,48)[1],"stale_archive_or_late_migration")
        migrated = dict(spec,last_modified="Fri, 13 Dec 2024 23:20:53 GMT")
        self.assertIn("after_outcome",snapshot_clock(migrated,election,48)[1])
        self.assertEqual(snapshot_clock(dict(spec,file=election["outcome_file"]),election,48)[1],"reserved_certified_outcome")

    def test_frozen_model_and_no_fitting_membership_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol, source = synthetic_corpus(root)
            path = root/"protocol.json";path.write_text(json.dumps(protocol))
            self.assertEqual(load_protocol(path,root),protocol)
            for change in ({"training_elections":["wa:king:2024-general"]}, {"used_for_tuning":True},
                           {"training_jurisdictions":["King County"]}, {"turnout_offsets":{"base":0}}):
                path.write_text(json.dumps(dict(protocol,**change)))
                with self.assertRaises(ElectionReportError):load_protocol(path,root)
            path.write_text(json.dumps(protocol));(root/"frozen.py").write_text("changed\n")
            with self.assertRaisesRegex(ElectionReportError,"bytes changed"):load_protocol(path,root)

    def test_predict_never_reads_reserved_outcomes_and_accounts_for_every_contest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol,source=synthetic_corpus(root,final=b"not even valid CSV")
            result=build_predictions(protocol,root,source)
            self.assertEqual(result["reserved_outcomes_read"],[])
            self.assertEqual(result["read_artifacts"],["webresults-20241105.csv"])
            self.assertEqual(len(result["rows"]),6)
            self.assertTrue(all(not r["operationally_eligible"] and r["posterior"]["verified_latest_pair_votes"]==0 for r in result["rows"]))
            record=result["inclusion_exclusion_ledger"][0]
            self.assertEqual(sum(len(v) for v in record["excluded_contests"].values())+len(record["candidates"]),4)
            corpus=Corpus(root,protocol)
            try:
                with self.assertRaisesRegex(ElectionReportError,"forbidden"):
                    corpus.read("webresults-20241126.csv",set(result["read_artifacts"]))
            finally:corpus.close()

    def test_reached_turnout_target_does_not_become_zero_remaining(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol,source=synthetic_corpus(root,current=synthetic_csv(160,90))
            result=build_predictions(protocol,root,source)
            self.assertEqual(result["rows"],[])
            candidates=result["inclusion_exclusion_ledger"][0]["candidates"]
            self.assertEqual(len(candidates),2)
            self.assertTrue(all(set(c["scenario_exclusions"].values())=={"turnout_target_reached_is_not_official_zero"} for c in candidates))

    def test_identical_bytes_keep_earliest_date_despite_hyphenated_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol,source=synthetic_corpus(root)
            name="webresults---20241106.csv"
            with zipfile.ZipFile(root/"king-archives.zip","a") as zipped:
                zipped.writestr(name,synthetic_csv())
            manifest_path=root/"king-archives-manifest.json"
            manifest=json.loads(manifest_path.read_text())
            manifest["archive_sha256"]=digest(root/"king-archives.zip")
            manifest["files"].append(dict(manifest["files"][0],file=name,last_modified="Thu, 07 Nov 2024 23:00:00 GMT"))
            manifest_path.write_text(json.dumps(manifest));protocol["corpus_manifest_sha256"]=digest(manifest_path)
            result=build_predictions(protocol,root,source)
            self.assertEqual(result["read_artifacts"],["webresults-20241105.csv"])
            duplicate=next(r for r in result["inclusion_exclusion_ledger"] if r["file"]==name)
            self.assertEqual(duplicate["file_exclusion"],"identical_snapshot_bytes_do_not_advance_count_clock")

    def test_scores_are_separate_explicitly_unvalidated_and_ties_are_not_reversals(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol,source=synthetic_corpus(root,final=synthetic_csv(100,100))
            result=build_predictions(protocol,root,source)
            path=root/"predictions.json";path.write_text(json.dumps(result))
            sealed=digest(path)
            scored=score_predictions(protocol,path,root,sealed)
            self.assertEqual(digest(path),sealed)
            self.assertEqual(scored["scored_base_rows"],2)
            self.assertEqual(scored["outcome_artifacts_read"],["webresults-20241126.csv"])
            self.assertTrue(all(r["outcome"]==0 for r in scored["scored_rows"]))
            self.assertEqual(scored["scores"]["fixed_half"]["brier"],.25)
            self.assertEqual(scored["scores"]["verified_latest_batch_only"]["status"],"NOT RUN")
            self.assertEqual(scored["scores"]["official_pool_margin_lock"]["status"],"NOT RUN")
            self.assertIsNone(scored["uncertainty"]["interval"])
            self.assertFalse(scored["production_eligible"] or scored["calibration_established"])

    def test_tampered_seal_leaked_outcome_duplicate_and_invalid_scope_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol,source=synthetic_corpus(root)
            original=build_predictions(protocol,root,source)
            path=root/"predictions.json";path.write_text(json.dumps(original));sealed=digest(path)
            path.write_text(json.dumps(original)+"\n")
            with self.assertRaisesRegex(ElectionReportError,"seal"):score_predictions(protocol,path,root,sealed)
            for mutation in ("leak","duplicate","scope","clock","missing_model","fabricated_baseline","missing_required_baseline"):
                result=deepcopy(original)
                if mutation=="leak":result["read_artifacts"].append("webresults-20241126.csv")
                elif mutation=="duplicate":result["rows"].append(result["rows"][0])
                elif mutation=="scope":result["rows"][0]["operationally_eligible"]=True
                elif mutation=="clock":result["rows"][0]["information_time"]="2024-11-27T00:00:00Z"
                elif mutation=="missing_model":result["rows"][0]["probability"]=None
                elif mutation=="fabricated_baseline":result["rows"][0]["baselines"]["verified_latest_batch_only"]=.5
                else:result["rows"][0]["baselines"]["cumulative_share"]=None
                path.write_text(json.dumps(result))
                with self.subTest(mutation=mutation),self.assertRaises(ElectionReportError):
                    score_predictions(protocol,path,root,digest(path))

    def test_equal_election_contest_weights_do_not_multiply_repeated_snapshots(self):
        def row(e,c,p,y):return {"election_id":e,"canonical_id":c,"probability":p,"outcome":y}
        rows=[row("one","a",.2,0),row("one","b",.8,1),row("two","c",.7,0)]
        expected=(.04+.49)/2
        self.assertAlmostEqual(_metrics(rows,"model")["brier"],expected)
        self.assertAlmostEqual(_metrics(rows+[deepcopy(rows[0]) for _ in range(100)],"model")["brier"],expected)
        self.assertTrue(math.isfinite(_metrics([row("one","a",0,1)],"model")["log_loss"]))


if __name__=="__main__":
    unittest.main()
