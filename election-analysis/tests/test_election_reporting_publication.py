# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:6c20e94a-0b43-41ee-82d3-01a2e62282d9

"""Publication precision and staleness (Em's publication findings), offline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from election_reporting.analysis import analyze_election
from election_reporting.publish import publish

from test_election_reporting import counting_config, result_rows


class PublicationPrecisionAndStalenessTests(unittest.TestCase):
    """Em's publication findings.

    Acceptance cases:
    P1 a probability above 99.9% but below 1 never shows as "100.0%";
    P2 exactly 1 still shows as "100.0%", and ordinary values keep one decimal;
    P3 the lower bound is unchanged ("<0.1%");
    P4 a second generation with no county breakdown removes the earlier
       county-breakdown.csv rather than leaving a stale one;
    P5 a generation with a breakdown still writes it.
    """

    def test_p1_near_certainty_is_not_certainty(self) -> None:
        from election_reporting.publish import _probability_label

        for value in (0.9991, 0.9995, 0.9996, 0.99999999):
            with self.subTest(value=value):
                self.assertEqual(_probability_label(value), ">99.9%")

    def test_p2_exact_one_and_ordinary_values(self) -> None:
        from election_reporting.publish import _probability_label

        self.assertEqual(_probability_label(1.0), "100.0%")
        self.assertEqual(_probability_label(0.999), "99.9%")
        self.assertEqual(_probability_label(0.5), "50.0%")
        self.assertEqual(_probability_label(None), "Not estimable")

    def test_p3_lower_bound_unchanged(self) -> None:
        from election_reporting.publish import _probability_label

        self.assertEqual(_probability_label(0.0004), "<0.1%")
        self.assertEqual(_probability_label(0.001), "0.1%")

    def two_generations(self, first: str | None, second: str | None) -> Path:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "2026-08-06.csv").write_text(
                result_rows("Local Proposition 1 (Vote for 1)", "9", 100, [("Yes", 51), ("No", 48)]),
                encoding="utf-8",
            )
            config_path = root / "election.toml"
            config_path.write_text(
                'schema_version = 1\n[election]\ntitle = "Test"\n[[sources]]\nid = "local"\n'
                'path = "inputs"\njurisdiction = "Local"\nscope = "local"\npriority = 1\n'
                "expected_final_ballots = 120\n",
                encoding="utf-8",
            )
            output = root / "reports"
            analysis = analyze_election(counting_config(config_path))
            for breakdown in (first, second):
                with mock.patch("election_reporting.publish.county_breakdown_csv", return_value=breakdown):
                    publish(analysis, output)
            path = output / "county-breakdown.csv"
            return path.read_text(encoding="utf-8") if path.exists() else None

    def test_p4_second_generation_without_areas_removes_stale_breakdown(self) -> None:
        self.assertIsNone(self.two_generations("Contest,County\nold,King\n", None))

    def test_p5_generation_with_areas_writes_breakdown(self) -> None:
        self.assertEqual(self.two_generations(None, "Contest,County\nnew,King\n"), "Contest,County\nnew,King\n")


if __name__ == "__main__":
    unittest.main()
