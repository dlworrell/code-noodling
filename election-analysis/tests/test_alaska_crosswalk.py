"""Consistency checks on the committed Alaska precinct-borough crosswalk."""

import collections
import csv
import unittest
from pathlib import Path

CROSSWALK = Path(__file__).resolve().parents[1] / "data" / "crosswalks" / "alaska"


def _rows(name):
    with (CROSSWALK / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class AlaskaCrosswalkTests(unittest.TestCase):
    def check_shares(self, rows, key):
        shares = collections.defaultdict(float)
        for row in rows:
            self.assertGreater(int(row["population_2020"]), 0)
            shares[row[key]] += float(row["population_share"])
        for unit, total in shares.items():
            self.assertAlmostEqual(total, 1.0, places=4, msg=unit)
        return shares

    def test_precincts_cover_the_whole_state(self):
        rows = _rows("precinct-borough.csv")
        precincts = self.check_shares(rows, "precinct_id")
        self.assertEqual(len(precincts), 402)
        self.assertEqual(sum(int(r["population_2020"]) for r in rows), 733391)
        self.assertEqual(len({r["borough_fips"] for r in rows}), 30)
        for row in rows:
            self.assertRegex(row["precinct_id"], r"^\d\d-\d\d\d$")
            self.assertTrue(row["precinct_name"].startswith(row["precinct_id"] + " "))
            self.assertEqual(row["precinct_id"][:2], row["house_district"])
            self.assertRegex(row["borough_fips"], r"^02\d\d\d$")

    def test_house_districts_agree_with_precincts(self):
        precincts = _rows("precinct-borough.csv")
        districts = _rows("house-district-borough.csv")
        self.assertEqual(len(self.check_shares(districts, "house_district")), 40)
        expected = collections.Counter()
        for row in precincts:
            expected[(row["house_district"], row["borough_fips"])] += int(row["population_2020"])
        actual = {(r["house_district"], r["borough_fips"]): int(r["population_2020"]) for r in districts}
        self.assertEqual(actual, dict(expected))


if __name__ == "__main__":
    unittest.main()
