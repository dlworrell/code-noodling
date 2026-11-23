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
K9 fails closed (fitment #221, #229): a failing gh call on the run list, on any
   run's jobs, a missing gh, unreadable JSON or an unexpected shape give run=false and exit 0;
K10 an empty run history after the cutoff runs the watcher;
K11 before the cutoff, and for dispatches, the history is not read at all;
K12 the collected history drives the decision and leaves out this run.
"""

from __future__ import annotations

import io
import json
import subprocess
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


RUNS = "repos/o/r/actions/workflows/election-source-watch.yml/runs"


class FakeGh:
    """Stands in for ``gh api``: answers by path prefix, or raises for paths marked failing."""

    def __init__(self, runs: list[dict], jobs: dict[int, list[dict]], fail: str | None = None) -> None:
        self.runs, self.jobs, self.fail, self.calls = runs, jobs, fail, []

    def __call__(self, path: str):
        self.calls.append(path)
        if self.fail is not None and path.startswith(self.fail):
            raise cadence.HistoryUnavailable(f"gh api {path}: exit 1")
        if path.startswith(RUNS):
            return {"workflow_runs": self.runs}
        run_id = int(path.split("/runs/")[1].split("/")[0])
        return {"jobs": self.jobs.get(run_id, [])}


AFTER = "2026-11-13T05:07:00Z"
HISTORY = [{"id": 1, "created_at": "2026-11-13T02:07:00Z"}, {"id": 2, "created_at": "2026-11-13T05:07:00Z"}]
JOBS = {1: [{"name": "cadence", "conclusion": "success"}, {"name": "watch", "conclusion": "success"}],
        2: [{"name": "cadence", "conclusion": None}]}


class FailClosedTests(unittest.TestCase):
    def cli(self, argv: list[str], run_result=None, side_effect=None) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cadence.subprocess, "run", return_value=run_result, side_effect=side_effect) as run, \
                redirect_stdout(out), redirect_stderr(err):
            code = cadence.main(argv)
        self.subprocess_calls = run.call_args_list
        return code, out.getvalue().strip(), err.getvalue()

    def test_k9_failing_gh_on_each_call_fails_closed(self) -> None:
        for fail in (RUNS, "repos/o/r/actions/runs/1/", "repos/o/r/actions/runs/"):
            gh = FakeGh(HISTORY, JOBS, fail=fail)
            run, reason = cadence.decide("schedule", t(AFTER), "o/r", 2, gh)
            self.assertFalse(run, fail)
            self.assertIn("run history unavailable", reason)

    def test_k9_cli_exit_status_and_unreadable_answers(self) -> None:
        argv = ["--event", "schedule", "--now", AFTER, "--repo", "o/r", "--run-id", "2"]
        failed = subprocess.CompletedProcess(["gh"], 1, stdout="", stderr="HTTP 502")
        garbled = subprocess.CompletedProcess(["gh"], 0, stdout="<html>", stderr="")
        wrong_shape = subprocess.CompletedProcess(["gh"], 0, stdout='{"message": "Not Found"}', stderr="")
        for kwargs in ({"run_result": failed}, {"run_result": garbled}, {"run_result": wrong_shape},
                       {"side_effect": FileNotFoundError("gh")},
                       {"side_effect": subprocess.TimeoutExpired("gh", 60)}):
            code, out, err = self.cli(argv, **kwargs)
            self.assertEqual((code, out), (0, "run=false"), kwargs)
            self.assertIn("::warning::run history unavailable", err)

    def test_k9_cli_jobs_call_failing_after_list_succeeds(self) -> None:
        listing = subprocess.CompletedProcess(["gh"], 0, stdout=json.dumps({"workflow_runs": HISTORY}), stderr="")
        failed = subprocess.CompletedProcess(["gh"], 1, stdout="", stderr="HTTP 500")
        code, out, _ = self.cli(["--event", "schedule", "--now", AFTER, "--repo", "o/r", "--run-id", "2"],
                                side_effect=[listing, failed])
        self.assertEqual((code, out), (0, "run=false"))
        self.assertEqual(len(self.subprocess_calls), 2)
        self.assertEqual(self.subprocess_calls[1].args[0], ["gh", "api", "repos/o/r/actions/runs/1/jobs"])
        for body in ('{"jobs": ["x"]}', '{"jobs": [["watch"]]}', '{"jobs": null}', '["x"]'):
            malformed = subprocess.CompletedProcess(["gh"], 0, stdout=body, stderr="")
            code, out, err = self.cli(["--event", "schedule", "--now", AFTER, "--repo", "o/r", "--run-id", "2"],
                                      side_effect=[listing, malformed])
            self.assertEqual((code, out), (0, "run=false"), body)
            self.assertIn("::warning::run history unavailable", err)

    def test_k10_empty_history_runs(self) -> None:
        gh = FakeGh([], {})
        self.assertTrue(cadence.decide("schedule", t(AFTER), "o/r", 2, gh)[0])
        self.assertEqual(len(gh.calls), 1)
        self.assertIn("created=>=2026-11-11T05:07:00Z", gh.calls[0])
        listing = subprocess.CompletedProcess(["gh"], 0, stdout='{"total_count": 0, "workflow_runs": []}', stderr="")
        self.assertEqual(self.cli(["--event", "schedule", "--now", AFTER, "--repo", "o/r", "--run-id", "2"],
                                  run_result=listing)[:2], (0, "run=true"))

    def test_k11_history_not_read_before_cutoff_or_for_dispatch(self) -> None:
        for event, now in (("schedule", "2026-11-12T07:07:00Z"), ("workflow_dispatch", AFTER)):
            gh = FakeGh(HISTORY, JOBS, fail="")  # every call would fail
            self.assertTrue(cadence.decide(event, t(now), "o/r", 2, gh)[0], (event, now))
            self.assertEqual(gh.calls, [])
            code, out, _ = self.cli(["--event", event, "--now", now, "--repo", "o/r"],
                                    side_effect=AssertionError("gh must not be called"))
            self.assertEqual((code, out), (0, "run=true"))
            self.assertEqual(self.subprocess_calls, [])

    def test_k12_collected_history_decides_and_skips_this_run(self) -> None:
        gh = FakeGh(HISTORY, JOBS)
        prior = cadence.collect_prior("o/r", 2, t(AFTER), gh)
        self.assertEqual(prior, [(t("2026-11-13T02:07:00Z"), "success")])
        self.assertNotIn("repos/o/r/actions/runs/2/jobs", gh.calls)
        self.assertFalse(cadence.decide("schedule", t(AFTER), "o/r", 2, gh)[0])
        self.assertTrue(cadence.decide("schedule", t(AFTER), "o/r", 2, FakeGh(HISTORY[1:], JOBS))[0])


if __name__ == "__main__":
    unittest.main()
