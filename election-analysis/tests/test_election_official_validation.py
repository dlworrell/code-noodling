# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:a92a1317-23c5-4424-9d38-da81e8c0b6f1

import copy
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from election_reporting.model import ElectionReportError
from election_reporting.official_archives import (choice_key, parse_count_lines, parse_pool_fields,
                                                   read_count_pdf, read_final_summary, read_pool_pdf)
from election_reporting import official_validation as ov
from election_reporting.retrospective import digest

FIXTURES = Path(__file__).parent/"fixtures"


def protocol(root=FIXTURES):
    return {"schema_version": 3, "mode": ov.MODE, "frozen_at": "2026-10-25T00:00:00Z",
            "used_for_tuning": False, "training_groups": [], "held_out_groups": ["ca:orange:2024-general"],
            "scenarios": copy.deepcopy(ov.SCENARIOS), "baselines": list(ov.BASELINES),
            "model_id": "disjoint_tempered_beta_binomial_v2", "information_time": "2024-11-09T19:05:00-08:00",
            "event_scope": "selected_pair_strict_reversal", "verified_new_ballot_batch": False,
            "certification_lower_bound": "2024-12-03T00:00:00-08:00",
            "whole_county_cities": ["IRVINE", "SANTA ANA"],
            "corpus_manifest_sha256": digest(root/"oc-official-2024-manifest.json"),
            "model_files": {"election_reporting/official_validation.py": digest(ov.SCOPE/"election_reporting/official_validation.py")}}


class OfficialArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with zipfile.ZipFile(FIXTURES/"oc-official-2024.zip") as z:
            cls.raw = {n: z.read(n) for n in z.namelist()}
        cls.before = read_count_pdf(cls.raw[ov.BEFORE])
        cls.current = read_count_pdf(cls.raw[ov.CURRENT])
        cls.pool_before = read_pool_pdf(cls.raw[ov.POOL_BEFORE])
        cls.pool = read_pool_pdf(cls.raw[ov.POOL])
        cls.final = read_final_summary(cls.raw[ov.FINAL])
        cls.manifest = json.loads((FIXTURES/"oc-official-2024-manifest.json").read_text())

    def test_original_count_inventory_methods_and_internal_clocks(self):
        self.assertEqual(len(self.before["contests"]), 212)
        self.assertEqual(len(self.current["contests"]), 212)
        self.assertEqual(self.current["source_ballots"]-self.before["source_ballots"], 64183)
        self.assertEqual(self.current["count_updated_at"], "2024-11-09T17:28:00-08:00")
        self.assertEqual(self.before["count_updated_at"], "2024-11-08T16:47:00-08:00")
        c = self.current["contests"]["PRESIDENT AND VICE PRESIDENT"]
        self.assertEqual(c["choices"]["KAMALA D. HARRIS TIM WALZ"]["votes"], 583791)
        self.assertEqual(c["cast_votes"], sum(v["votes"] for v in c["choices"].values()))
        self.assertIsNone(c["over_votes"])

    def test_amended_form_canonical_values_and_challenges_are_separate(self):
        self.assertTrue(self.pool_before["amended"])
        self.assertFalse(self.pool["amended"])
        self.assertEqual(self.pool["estimated_unprocessed"], 212152)
        self.assertEqual(self.pool["challenged_left_to_cure"], 9512)
        self.assertIn("unknown", self.pool["challenged_overlap_with_total"])
        self.assertEqual(sum(self.pool["unprocessed_categories"].values()), 212152)
        self.assertEqual(self.pool["source_field_2"], 205077)
        self.assertNotEqual(self.pool["source_field_2"], self.current["source_ballots"]-self.before["source_ballots"])

    def test_pool_total_and_election_identity_cannot_be_repaired_silently(self):
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(self.raw[ov.POOL]))
        fields = {name: str(item.get("/V", "")) for name, item in reader.get_fields().items()}
        for key, value in (("undefined_7", "212153"), ("County Name", "Another County"),
                           ("undefined", "November 9, 2024 5:00 pm")):
            bad = {**fields, key: value}
            if key == "undefined":
                bad["11Next vote results expected date and time"] = "November 8, 2024 5:00 pm"
            with self.subTest(field=key), self.assertRaises(ElectionReportError):
                parse_pool_fields(bad)

    def test_source_addition_statement_does_not_prove_correction_free_batches(self):
        ledger = ov.transition(self.before, self.current, self.pool_before, self.pool, self.manifest)
        self.assertEqual(ledger["processed_ballots_added"], 64183)
        self.assertEqual(ledger["processed_plus_queue_change"], 915)
        self.assertEqual(ledger["election_night_processed_baseline"], 1007149)
        self.assertEqual(ledger["operational_delta_kind"], "unknown")
        self.assertTrue(all(not r["verified_new_ballots"] for r in ledger["contests"]))
        revised = copy.deepcopy(self.current)
        contest = revised["contests"]["County Supervisor 1st District"]
        name = next(iter(contest["choices"]))
        contest["choices"][name]["votes"] = self.before["contests"][contest["title"]]["choices"][name]["votes"]-1
        bad = ov.transition(self.before, revised, self.pool_before, self.pool, self.manifest)
        self.assertFalse(next(r for r in bad["contests"] if r["title"] == contest["title"])["conditional_increment_usable"])
        bad_pool = {**self.pool, "processed_ballots": self.pool["processed_ballots"]+1}
        with self.assertRaises(ElectionReportError):
            ov.transition(self.before, self.current, self.pool_before, bad_pool, self.manifest)

    def test_final_summary_preserves_original_names_and_missing_overvotes(self):
        self.assertEqual(len(self.final["contests"]), 212)
        self.assertEqual(self.final["source_ballots"], 1417397)
        c = self.final["contests"]["PRESIDENT AND VICE PRESIDENT"]
        self.assertIn("KAMALA D. HARRIS TIM WALZ", c["choices"])
        self.assertIn("/", c["choices"]["KAMALA D. HARRIS TIM WALZ"]["source_label"])
        self.assertIsNone(c["over_votes"])
        self.assertNotEqual(choice_key("MINERVA S. GÓMEZ", "x"), choice_key("MINERVA S. G+MEZ", "x"))

    def test_bad_summary_rows_and_wrong_pdf_scope_rejected(self):
        raw = self.raw[ov.FINAL]
        cases = [b"Unofficial"+raw, raw.replace(b'Total Votes,Number Of Precincts', b'Other Votes,Number Of Precincts', 1), raw+b'"bad"\n', b'\xff']
        for value in cases:
            with self.subTest(sample=value[:20]), self.assertRaises(ElectionReportError):
                read_final_summary(value)
        for value in (b"not PDF", self.raw["gen2024-certified-sov.pdf"]):
            with self.assertRaises(ElectionReportError):
                read_count_pdf(value)

    def test_malformed_pdf_numeric_rows_cannot_hide_as_wrapped_names(self):
        lines = ["Test - Vote for One", "Precincts Counted Total Percent Voters Ballots Registered Percent", "1 1 100.00% 10 20 50.00%",
                 "Choice Party Vote Center Vote by Mail Total", "A 4 80.00% 4 80.00% 8 80.00%", "B 1 20.00% 1 20.00% 2 20.00%",
                 "Cast Votes: 5 100.00% 5 100.00% 10 100.00%", "*** End of report ***"]
        self.assertEqual(parse_count_lines(lines)["Test"]["cast_votes"], 10)
        broken = [lines[:-1], lines+["extra"], lines[:5]+["B -1 20.00% 1 20.00% 2 20.00%"]+lines[6:],
                  lines[:5]+["garbage"]+lines[5:], lines[:6]+["Cast Votes: 5 100.00% 5 100.00% 11 100.00%"]+lines[7:]]
        for value in broken:
            with self.subTest(lines=value), self.assertRaises(ElectionReportError):
                parse_count_lines(value)

    def test_protocol_refuses_training_reuse_retuning_and_changed_code(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"p.json"
            good = protocol();p.write_text(json.dumps(good));self.assertEqual(ov.load_protocol(p)["training_groups"], [])
            for field, value in (("training_groups", ["ca:orange:2024-general"]), ("used_for_tuning", True),
                                 ("verified_new_ballot_batch", True), ("scenarios", {"base": [1.1, False]}),
                                 ("model_files", {"election_reporting/official_validation.py": "0"*64})):
                p.write_text(json.dumps({**good, field: value}))
                with self.subTest(field=field), self.assertRaises(ElectionReportError):
                    ov.load_protocol(p)

    def test_prediction_phase_isolated_from_reserved_final_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = copy.deepcopy(self.manifest)
            values = {**self.raw, ov.FINAL: b"not a CSV outcome"}
            with zipfile.ZipFile(root/"oc-official-2024.zip", "w", zipfile.ZIP_DEFLATED) as z:
                for name, raw in values.items():
                    z.writestr(name, raw)
            manifest["archive_sha256"] = digest(root/"oc-official-2024.zip")
            for s in manifest["files"]:
                s["sha256"] = hashlib.sha256(values[s["file"]]).hexdigest();s["bytes"] = len(values[s["file"]])
            (root/"oc-official-2024-manifest.json").write_text(json.dumps(manifest))
            by_sha = {self.before["sha256"]: self.before, self.current["sha256"]: self.current,
                      self.pool_before["sha256"]: self.pool_before, self.pool["sha256"]: self.pool}
            def cached(raw):
                return by_sha[hashlib.sha256(raw).hexdigest()]
            with patch.object(ov, "read_count_pdf", side_effect=cached), patch.object(ov, "read_pool_pdf", side_effect=cached):
                result = ov.build_predictions(protocol(root), root)
            self.assertEqual(set(result["read_artifacts"]), ov.INPUTS)
            self.assertEqual(result["reserved_outcomes_read"], [])
            self.assertTrue(result["rows"])
            self.assertTrue(all(r["verified_new_ballot_pair_votes"] == 0 and not r["operationally_eligible"] for r in result["rows"]))
            self.assertEqual(len(result["inclusion_exclusion_ledger"]), 212)
            case = next(r for r in result["rows"] if r["scenario"] == "base")
            self.assertEqual(case["assumed_countable_county_queue"], 212152)
            self.assertEqual(case["posterior"]["latest_verified_pair_votes"], 0)
            self.assertNotIn("latest_verified_pair_votes", case["conditional_increment_posterior"])
            with self.assertRaises(ElectionReportError):
                ov.score_predictions(protocol(root), root/"oc-official-2024.zip", root, "0"*64)
            corpus = ov.Corpus(root, protocol(root))
            try:
                with self.assertRaises(ElectionReportError):corpus.read(ov.FINAL, ov.INPUTS)
            finally:corpus.close()


if __name__ == "__main__":
    unittest.main()
