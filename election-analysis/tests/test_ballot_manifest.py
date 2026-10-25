# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:53575cd9-8e6f-4cc5-8e12-bddfe5a60dd5

import hashlib
import unittest
from pathlib import Path

from election_reporting.ballot_manifest import qualify_manifest
from election_reporting.model import ElectionReportError


class BallotManifestTests(unittest.TestCase):
    def qualify(self, raw):
        return qualify_manifest(raw, hashlib.sha256(raw).hexdigest())

    def test_publisher_hash_and_full_inventory_do_not_imply_dated_votes(self):
        raw = (Path(__file__).parent/"fixtures/oc-ballot-manifest-2024.csv").read_bytes()
        report = qualify_manifest(raw, "36b6c0079cc3882e601dc7ef7024d987973e3a17bceb9263dabbc433c6e78cff")
        self.assertEqual(report["rows"], 10208)
        self.assertEqual(report["unique_composite_batch_ids"], 10208)
        self.assertEqual(report["source_count_sum"], 2757630)
        self.assertEqual(report["counts_by_container"], {"in-person": 578577, "mail": 2179053})
        self.assertFalse(report["new_ballot_likelihood_eligible"])
        for field in ("publication_or_count_clock", "candidate_votes", "correction_partition"):
            self.assertIsNone(report[field])
        with self.assertRaises(ElectionReportError):qualify_manifest(raw+b"\n", report["sha256"])

    def test_tabulator_is_part_of_batch_identity(self):
        header = b"Container,Tabulator,Batch Name,Number of Ballots\n"
        self.assertEqual(self.qualify(header+b"mail,pod1-vbm,1,10\nmail,pod2-vbm,1,11\n")["rows"], 2)
        with self.assertRaises(ElectionReportError):
            self.qualify(header+b"mail,pod1-vbm,1,10\nmail,pod1-vbm,1,11\n")

    def test_malformed_unbounded_or_unknown_inventory_cannot_be_promoted(self):
        header = b"Container,Tabulator,Batch Name,Number of Ballots\n"
        for row in (b"mail,pod1-vbm,1,-1\n", b"mail,pod1-vbm,1,0\n", b"mail,pod1-vbm,1,10001\n", b"unknown,pod1-vbm,1,10\n", b"mail,pod1-vbm,1,10,extra\n", b"mail,pod1-vbm,1,\xff\n"):
            with self.subTest(row=row), self.assertRaises(ElectionReportError):self.qualify(header+row)


if __name__ == "__main__":unittest.main()
