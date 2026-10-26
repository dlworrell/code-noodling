# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:6abdb03f-22fb-44e4-a1af-8f0a25b24129

"""Collection boundaries and zero-network behavior for the 2026 deployment."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from election_reporting.model import ElectionReportError
from election_reporting.watch import __main__ as cli
from election_reporting.watch.config import load_watch_settings
from election_reporting.watch.run import fetch_snapshots, plan
from election_reporting.watch.schedule import before_start, is_due, watch_window
from test_election_watch import FakeWeb

SCOPE = Path(__file__).resolve().parents[1]
PRODUCTION = SCOPE / "data" / "watch.toml"
FLOORS = {
    "maine": datetime(2026, 11, 4, 2, tzinfo=timezone.utc),
    "washington": datetime(2026, 11, 4, 5, tzinfo=timezone.utc),
    "alaska": datetime(2026, 11, 4, 7, tzinfo=timezone.utc),
}


def settings_in(root: Path):
    shutil.copytree(SCOPE / "data" / "crosswalks", root / "crosswalks")
    path = root / "watch.toml"
    path.write_bytes(PRODUCTION.read_bytes())
    return path, load_watch_settings(path)


class StartGateTests(unittest.TestCase):
    def test_deployment_pins_the_election_and_each_post_dst_floor(self):
        settings = load_watch_settings(PRODUCTION)
        self.assertEqual(len(settings.watches), 3)
        for config in settings.watches:
            with self.subTest(state=config.state_id):
                self.assertEqual(config.election, "2026-11-03")
                self.assertEqual(config.not_before, FLOORS[config.state_id])
                self.assertEqual(config.not_before.astimezone(config.timezone).date(), date(2026, 11, 3))
                self.assertFalse(before_start(config, FLOORS[config.state_id]))
        alaska = next(c for c in settings.watches if c.state_id == "alaska")
        self.assertEqual(alaska.not_before.astimezone(ZoneInfo("America/Adak")).hour, 21)
        self.assertEqual(alaska.not_before.astimezone(ZoneInfo("America/Anchorage")).hour, 22)

    def test_before_each_floor_even_force_makes_zero_requests_or_election_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, settings = settings_in(root)
            for config in settings.watches:
                for force in (False, True):
                    with self.subTest(state=config.state_id, force=force):
                        fake = FakeWeb({})
                        result = fetch_snapshots(config, fake, FLOORS[config.state_id] - timedelta(seconds=1),
                                                 force=force, log=lambda _: None)
                        self.assertFalse(result.due)
                        self.assertTrue(result.skipped)
                        self.assertEqual(fake.requests, [])
                        self.assertFalse(settings.elections_root.exists())

    def test_exact_and_later_boundaries_allow_real_adapter_requests(self):
        for delta in (timedelta(), timedelta(seconds=1)):
            with tempfile.TemporaryDirectory() as directory:
                _, settings = settings_in(Path(directory))
                for config in settings.watches:
                    with self.subTest(state=config.state_id, delta=delta):
                        fake = FakeWeb({})
                        try:
                            result = fetch_snapshots(config, fake, FLOORS[config.state_id] + delta,
                                                     log=lambda _: None)
                        except ElectionReportError as error:
                            # Washington's discovery request fails closed on this empty fake.
                            self.assertIn("HTTP 404", str(error))
                        else:
                            self.assertTrue(result.due)
                        self.assertTrue(fake.requests)

    def test_states_open_independently(self):
        now = datetime(2026, 11, 4, 4, tzinfo=timezone.utc)
        settings = load_watch_settings(PRODUCTION)
        open_states = {c.state_id for c in settings.watches if not before_start(c, now)}
        self.assertEqual(open_states, {"maine"})

    def test_plan_cannot_leak_metadata_requests_before_floor(self):
        for config in load_watch_settings(PRODUCTION).watches:
            fake = FakeWeb({})
            with self.subTest(state=config.state_id), self.assertRaisesRegex(ElectionReportError, "held until"):
                plan(config, fake, FLOORS[config.state_id] - timedelta(seconds=1))
            self.assertEqual(fake.requests, [])

    def test_fixed_2026_end_prevents_later_annual_source_requests(self):
        for config in load_watch_settings(PRODUCTION).watches:
            fake = FakeWeb({})
            result = fetch_snapshots(config, fake, datetime(2027, 11, 4, 8, tzinfo=timezone.utc),
                                     log=lambda _: None)
            self.assertFalse(result.due)
            self.assertEqual(fake.requests, [])

    def test_start_override_cannot_move_the_collection_window_earlier(self):
        for config in load_watch_settings(PRODUCTION).watches:
            config = replace(config, start=FLOORS[config.state_id] - timedelta(hours=2))
            window = watch_window(config, date(2026, 11, 3), "general", True)
            self.assertEqual(window.start, FLOORS[config.state_id])
            self.assertFalse(is_due(config, window, window.start - timedelta(seconds=1)))
            self.assertTrue(is_due(config, window, window.start))

    def test_naive_clock_is_rejected(self):
        config = load_watch_settings(PRODUCTION).watches[0]
        with self.assertRaisesRegex(ElectionReportError, "UTC offset"):
            fetch_snapshots(config, FakeWeb({}), datetime(2026, 11, 3), log=lambda _: None)

    def test_hard_floor_requires_offset_and_fixed_election(self):
        for election, moment, message in (
            ("2026-11-03", "2026-11-03T21:00:00", "explicit UTC offset"),
            ("next", "2026-11-03T21:00:00-08:00", "fixed election date"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "watch.toml"
                path.write_text(f'schema_version = 1\n[[watch]]\nstate = "Washington"\nelection = "{election}"\n'
                                f'[watch.schedule]\nnot_before = "{moment}"\n')
                with self.assertRaisesRegex(ElectionReportError, message):
                    load_watch_settings(path)


class CommandGateTests(unittest.TestCase):
    def test_forced_live_command_uses_real_clock_and_skips_all_held_states(self):
        with tempfile.TemporaryDirectory() as directory:
            config, settings = settings_in(Path(directory))
            with patch.object(cli, "_now", return_value=datetime(2026, 10, 26, tzinfo=timezone.utc)), \
                 patch.object(cli, "http_fetch") as network, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["fetch", "--config", str(config), "--force"]), 0)
            network.assert_not_called()
            self.assertFalse(settings.elections_root.exists())

    def test_future_testing_clock_cannot_bypass_live_commands(self):
        for command in ("plan", "fetch"):
            with self.subTest(command=command), patch.object(cli, "http_fetch") as network, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_status:
                cli.main([command, "--force", "--now", "2026-11-04T08:00:00Z"])
            self.assertEqual(exit_status.exception.code, 2)
            network.assert_not_called()

    def test_plan_before_all_floors_is_successful_offline_status(self):
        with patch.object(cli, "_now", return_value=datetime(2026, 10, 26, tzinfo=timezone.utc)), \
             patch.object(cli, "http_fetch") as network, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(["plan", "--config", str(PRODUCTION)]), 0)
        network.assert_not_called()
        self.assertEqual(output.getvalue().count("held until"), 3)

    def test_offline_status_outputs_workflow_gate_before_during_and_after_season(self):
        cases = (("2026-11-04T01:59:59Z", "false"),
                 ("2026-11-04T02:00:00Z", "true"),
                 ("2027-11-04T02:00:00Z", "false"))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "output"
            for moment, expected in cases:
                target.write_text("")
                with self.subTest(moment=moment), patch.dict(os.environ, GITHUB_OUTPUT=str(target)), \
                     patch.object(cli, "http_fetch") as network, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(["status", "--config", str(PRODUCTION), "--now", moment]), 0)
                self.assertEqual(target.read_text(), f"eligible={expected}\n")
                network.assert_not_called()

    def test_workflow_command_distinguishes_a_hold_from_a_failure(self):
        with patch.object(cli, "http_fetch") as network, contextlib.redirect_stdout(io.StringIO()):
            result = cli.main(["status", "--config", str(PRODUCTION), "--require-eligible",
                               "--force", "--now", "2026-11-04T01:59:59Z"])
        self.assertEqual(result, 3)
        network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
