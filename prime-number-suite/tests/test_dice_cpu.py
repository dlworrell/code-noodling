"""Black-box integration tests for the CPU dice and prime-seed workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


if len(sys.argv) != 3:
    raise SystemExit(
        "usage: test_dice_cpu.py /path/to/dice_cpu /path/to/ose_cpu"
    )
DICE_EXECUTABLE = Path(sys.argv[1]).resolve()
OSE_EXECUTABLE = Path(sys.argv[2]).resolve()
del sys.argv[1:]


class DiceCpuIntegrationTests(unittest.TestCase):
    """Verify deterministic seeding, bounds, and serialized outputs."""

    def run_dice(self, *arguments: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(DICE_EXECUTABLE), *(str(value) for value in arguments)],
            check=False,
            capture_output=True,
            text=True,
        )

    def create_prime_file(self, directory: Path, upper_bound: int = 10_000) -> Path:
        path = directory / "primes.json"
        result = subprocess.run(
            [
                str(OSE_EXECUTABLE),
                "2",
                str(upper_bound),
                "--json",
                str(path),
                "--no-list",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return path

    def test_prime_seeded_single_die_is_reproducible_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            prime_path = self.create_prime_file(Path(directory_name))
            arguments = (
                "--faces",
                20,
                "--count",
                250,
                "--use-prime-seeds",
                prime_path,
            )
            first = self.run_dice(*arguments)
            second = self.run_dice(*arguments)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first.stdout, second.stdout)

            rolls = [int(value) for value in first.stdout.split()]
            self.assertEqual(len(rolls), 250)
            self.assertTrue(all(1 <= value <= 20 for value in rolls))

    def test_bundle_outputs_have_expected_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            prime_path = self.create_prime_file(directory)
            json_path = directory / "rolls.json"
            csv_path = directory / "rolls.csv"
            result = self.run_dice(
                "--spec",
                "3d6+2",
                "--count",
                40,
                "--use-prime-seeds",
                prime_path,
                "--log-json",
                json_path,
                "--csv",
                csv_path,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            totals = [int(value) for value in result.stdout.split()]
            self.assertEqual(len(totals), 40)
            self.assertTrue(all(5 <= value <= 20 for value in totals))

            records = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 40)
            self.assertTrue(all(len(record["rolls"]) == 3 for record in records))

            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 120)

    def test_wrapped_prime_list_mixes_the_logical_roll_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            prime_path = self.create_prime_file(Path(directory_name), upper_bound=11)
            result = self.run_dice(
                "--faces",
                1_000_003,
                "--count",
                10,
                "--use-prime-seeds",
                prime_path,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rolls = [int(value) for value in result.stdout.split()]
            self.assertEqual(len(rolls), 10)
            # The five-prime input wraps once. Index mixing ensures the second
            # pass does not replay the first five RNG states verbatim.
            self.assertNotEqual(rolls[:5], rolls[5:])

    def test_invalid_count_is_rejected(self) -> None:
        result = self.run_dice("--faces", 6, "--count", 0)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
