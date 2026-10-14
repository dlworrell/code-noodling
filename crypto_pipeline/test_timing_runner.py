"""Fault-injection tests for orchestration; these are not timing measurements."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_timing_diagnosis as runner


class RunnerTests(unittest.TestCase):
    def exercise(self, fault=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = root / 'vendor/src'
            vendor.mkdir(parents=True)
            (vendor / 'dudect.h').write_text('header')
            for name in ('crypto_montgomery_ct.c', 'crypto_montgomery_ct.h',
                         'crypto_multiplier.h', 'dudect_montgomery.c'):
                (root / name).write_text('source')
            output = root / 'evidence'
            executions = 0

            def fake_run(argv, **kwargs):
                nonlocal executions
                code = 0
                if '-c' in argv and argv[0] == 'cc' and fault == 'base_build':
                    code = 1
                elif '-DCT_TIMING_NEGATIVE_CONTROL' in argv and fault == 'build':
                    code = 1  # Must never be mistaken for a successful leak detection.
                elif argv[0] == 'taskset':
                    executions += 1
                    # Reports must already identify the in-flight execution.
                    checkpoint = json.loads((output / 'report.json').read_text())
                    self.assertTrue(any(p['status'] == 'running' for p in checkpoint['plan']))
                    if fault == 'timeout':
                        raise subprocess.TimeoutExpired(argv, 1)
                    if fault == 'launch':
                        raise FileNotFoundError('injected launch failure')
                    if fault == 'drift' and executions == 1:
                        (root / 'crypto_multiplier.h').unlink()
                    if fault == 'drift' and executions == 2:
                        (root / 'crypto_multiplier.h').write_text('source')
                    code = 1 if 'leaky-control' in argv[-1] else 0
                    if fault == 'taskset':
                        code = 1
                    else:
                        kwargs['stdout'].write('LEAKAGE EVIDENCE DETECTED' if code == 1 else
                            'NO LEAKAGE DETECTED IN THIS BOUNDED RUN; NOT A PROOF')
                return subprocess.CompletedProcess(argv, code)

            argv = ['runner', '--compiler', 'cc', '--limbs', '16', '--batches', '2',
                    '--dudect', str(vendor.parent), '--output', str(output)]
            with patch.object(runner, 'ROOT', root), patch.object(sys, 'argv', argv), \
                 patch.object(runner.subprocess, 'check_output', return_value=runner.PIN), \
                 patch.object(runner.subprocess, 'run', side_effect=fake_run), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = runner.main()
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(len(report['plan']), 10)
            self.assertEqual(rc, int(fault is not None))
            self.assertEqual(report['diagnostic_failure'], fault is not None)
            self.assertNotIn('report.json.tmp', report['artifact_sha256'])
            if fault == 'drift':
                self.assertTrue(report['source_modified_mid_run'])
                self.assertEqual(report['initial_source_hashes'], report['current_source_hashes'])
            if fault in ('base_build', 'launch'):
                self.assertFalse(report['matrix_completed'])
                self.assertIn('exception_message', report)
            else:
                self.assertTrue(report['matrix_completed'])
            if fault == 'build':
                self.assertEqual(report['plan'][0]['status'], 'build_failed')
            if fault == 'timeout':
                self.assertTrue(all(p['status'] == 'timeout' for p in report['plan']))

    def test_taskset_failure_is_not_leakage(self): self.exercise('taskset')

    def test_expected_outcomes(self): self.exercise()
    def test_build_failure_is_not_leakage(self): self.exercise('build')
    def test_candidate_build_failure_retained(self): self.exercise('base_build')
    def test_timeouts_retained(self): self.exercise('timeout')
    def test_launch_failure_retained(self): self.exercise('launch')
    def test_deleted_then_restored_source_stays_failed(self): self.exercise('drift')


if __name__ == '__main__':
    unittest.main()
