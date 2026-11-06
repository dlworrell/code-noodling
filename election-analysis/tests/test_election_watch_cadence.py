# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:156d64bd-4eff-4714-b541-569b828637cf

"""Daily cadence gate (fitment #214, #215).

K1 before the cutoff every scheduled fire runs;
K2 manual dispatches always run;
K3 after the cutoff the 02:07 fire runs when nothing ran that cadence day;
K4 a late start (03:07, 23:07, even 01:07 next UTC day) still runs the day's check;
K5 a later fire skips once a run succeeded in the same cadence day;
K6 a failed or skipped watcher does not count, so the next fire retries;
K7 the cadence day starts at 02:00 UTC;
K8 the CLI reads prior runs as JSON lines and prints run=...
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from unittest import mock

from election_reporting.watch import cadence


def t(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


class CadenceTests(unittest.TestCase):
    def test_k1_hourly_before_cutoff(self) -> None:
        for now in ("2026-11-11T15:07:00Z", "2026-11-12T07:59:00Z"):
            self.assertTrue(cadence.should_run("schedule", t(now), [(t("2026-11-11T02:07:00Z"), "success")])[0])

    def test_k2_dispatch_always_runs(self) -> None:
        self.assertTrue(cadence.should_run("workflow_dispatch", t("2026-11-13T03:12:00Z"),
                                           [(t("2026-11-13T02:07:00Z"), "success")])[0])

    def test_k3_first_fire_of_the_day_runs(self) -> None:
        self.assertTrue(cadence.should_run("schedule", t("2026-11-13T02:07:00Z"),
                                           [(t("2026-11-12T02:07:00Z"), "success")])[0])

    def test_k4_late_start_still_runs(self) -> None:
        prior = [(t("2026-11-12T07:07:00Z"), "success")]  # last hourly run before the cutoff
        for now in ("2026-11-12T08:07:00Z", "2026-11-12T23:07:00Z", "2026-11-13T01:07:00Z"):
            # Nov 12 02:00 day already had a success before the cutoff: skip
            self.assertFalse(cadence.should_run("schedule", t(now), prior)[0], now)
        for now in ("2026-11-13T03:07:00Z", "2026-11-13T23:07:00Z", "2026-11-14T01:07:00Z"):
            self.assertTrue(cadence.should_run("schedule", t(now), prior)[0], now)

    def test_k5_later_fires_skip_after_success(self) -> None:
        prior = [(t("2026-11-13T04:07:00Z"), "success")]
        for now in ("2026-11-13T05:07:00Z", "2026-11-14T01:07:00Z"):
            self.assertFalse(cadence.should_run("schedule", t(now), prior)[0], now)
        self.assertTrue(cadence.should_run("schedule", t("2026-11-14T02:07:00Z"), prior)[0])

    def test_k6_failed_or_skipped_runs_do_not_count(self) -> None:
        prior = [(t("2026-11-13T02:07:00Z"), "failure"), (t("2026-11-13T03:07:00Z"), "skipped"),
                 (t("2026-11-13T03:30:00Z"), None)]
        self.assertTrue(cadence.should_run("schedule", t("2026-11-13T04:07:00Z"), prior)[0])

    def test_k7_day_start(self) -> None:
        self.assertEqual(cadence.day_start(t("2026-11-13T01:59:00Z")), t("2026-11-12T02:00:00Z"))
        self.assertEqual(cadence.day_start(t("2026-11-13T02:00:00Z")), t("2026-11-13T02:00:00Z"))

    def test_k8_cli(self) -> None:
        rows = "\n".join(json.dumps(r) for r in (
            {"created_at": "2026-11-13T02:07:00Z", "watch": "success"},
            {"created_at": "2026-11-12T02:07:00Z", "watch": "success"},
        ))
        for now, expected in (("2026-11-13T05:07:00Z", "run=false"), ("2026-11-14T02:07:00Z", "run=true")):
            out = io.StringIO()
            with mock.patch("sys.stdin", io.StringIO(rows)), redirect_stdout(out), redirect_stderr(io.StringIO()):
                self.assertEqual(cadence.main(["--event", "schedule", "--now", now]), 0)
            self.assertEqual(out.getvalue().strip(), expected)
        with self.assertRaises(ValueError):
            cadence._time("2026-11-13T02:07:00")


if __name__ == "__main__":
    unittest.main()
