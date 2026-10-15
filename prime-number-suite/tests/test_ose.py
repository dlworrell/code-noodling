"""Black-box correctness and CLI tests for the CPU prime generator."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


if len(sys.argv) != 2:
    raise SystemExit("usage: test_ose.py /path/to/ose_cpu")
OSE_EXECUTABLE = Path(sys.argv[1]).resolve()
del sys.argv[1:]


def expected_primes(start: int, end: int) -> list[int]:
    """Return the reference prime list for the small integration-test ranges."""

    output: list[int] = []
    for candidate in range(max(start, 2), end + 1):
        upper = math.isqrt(candidate)
        if all(candidate % divisor for divisor in range(2, upper + 1)):
            output.append(candidate)
    return output


class OseIntegrationTests(unittest.TestCase):
    """Exercise range boundaries, serialization, and argument validation."""

    def run_ose(self, *arguments: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(OSE_EXECUTABLE), *(str(value) for value in arguments)],
            check=False,
            capture_output=True,
            text=True,
        )

    def assert_json_range(self, start: int, end: int) -> None:
        result = self.run_ose(start, end, "--json", "-")
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(result.stdout)
        reference = expected_primes(start, end)
        self.assertEqual(document["range"], {"start": start, "end": end})
        self.assertEqual(document["count"], len(reference))
        self.assertEqual(document["primes"], reference)

    def test_inclusive_boundaries_and_empty_ranges(self) -> None:
        for start, end in ((2, 2), (-10, 30), (14, 16), (17, 17), (18, 18)):
            with self.subTest(start=start, end=end):
                self.assert_json_range(start, end)

    def test_segment_crossing_range(self) -> None:
        self.assert_json_range(999_900, 1_000_100)

    def test_json_file_is_valid_without_human_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "primes.json"
            result = self.run_ose(1, 100, "--json", output_path, "--no-list")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            document = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(document["count"], 25)
            self.assertEqual(document["primes"][-1], 97)

    def test_invalid_range_and_column_count_are_rejected(self) -> None:
        reversed_range = self.run_ose(10, 1)
        self.assertNotEqual(reversed_range.returncode, 0)

        zero_columns = self.run_ose(1, 10, "--cols", 0)
        self.assertNotEqual(zero_columns.returncode, 0)


if __name__ == "__main__":
    unittest.main()
