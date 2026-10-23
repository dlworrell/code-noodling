# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:46d40e88-b776-438c-8ad4-bf58ce91839d

"""Reject malformed/irrelevant solver evidence before it can become a proof."""
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pycparser import c_generator
from verify_carry_loop import evaluate_json, extract, harness, run, ROOT, PIN


def output(mutation=False):
    properties = [
        {'property': 'main.assertion.1', 'description': 'MATH_INVARIANT',
         'status': 'FAILURE' if mutation else 'SUCCESS'},
        {'property': 'main.assertion.2', 'description': 'FRAME_ISOLATION', 'status': 'SUCCESS'},
        {'property': 'main.assertion.3', 'description': 'TERMINAL_CARRY_BIT', 'status': 'SUCCESS'},
        {'property': 'main.unwind.0', 'description': 'unwinding assertion', 'status': 'SUCCESS'},
        {'property': 'main.pointer.1', 'description': 'pointer check', 'status': 'SUCCESS'}]
    if mutation:
        properties[0]['trace'] = [{'stepType': 'assignment'}, {'stepType': 'failure'}]
    return [{'result': properties}, {'cProverStatus': 'failure' if mutation else 'success'}]


class EvidenceTests(unittest.TestCase):
    def reject(self, data, code=0, mutation=False):
        with self.assertRaises((ValueError, TypeError)):
            evaluate_json(json.dumps(data), code, mutation)

    def test_pristine_success(self):
        self.assertEqual(evaluate_json(json.dumps(output()), 0)['failed_property_ids'], [])

    def test_intended_arithmetic_counterexample(self):
        self.assertEqual(evaluate_json(json.dumps(output(True)), 10, True)['failed_property_ids'],
                         ['main.assertion.1'])

    def test_text_fallback_and_old_schema_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_json('broken JSON VERIFICATION SUCCESSFUL', 0)
        for data in ([], {}, [{'cproverResult': 'success', 'properties': []}],
                     [{'result': []}, {'cProverStatus': 'success'}]):
            with self.subTest(data=data): self.reject(data)

    def test_non_solver_exit_cannot_pass(self):
        for code in (None, -9, 1, 6, 10):
            with self.subTest(code=code): self.reject(output(), code)
        self.reject(output(True), 1, True)

    def test_missing_or_wrong_summary(self):
        for summary in ([], [{'cProverStatus': 'failure'}], [{'cProverStatus': 'unknown'}]):
            with self.subTest(summary=summary): self.reject(output()[:1] + summary)

    def test_unknown_or_missing_property(self):
        for status in ('UNKNOWN', 'NOT_CHECKED', None):
            data = output(); data[0]['result'][3]['status'] = status
            with self.subTest(status=status): self.reject(data)
        data = output(); data[0]['result'].pop(1)
        self.reject(data)

    def test_duplicate_identifiers_and_labels(self):
        data = output(); data[0]['result'][1]['property'] = 'main.assertion.1'
        self.reject(data)
        data = output(); prop = copy.deepcopy(data[0]['result'][0]); prop['property'] = 'other'
        data[0]['result'].append(prop)
        self.reject(data)

    def test_wrong_or_additional_failure_rejected(self):
        data = output(True); data[0]['result'][0]['status'] = 'SUCCESS'
        data[0]['result'][3]['status'] = 'FAILURE'
        self.reject(data, 10, True)
        data = output(True); data[0]['result'][4]['status'] = 'FAILURE'
        self.reject(data, 10, True)

    def test_description_collision_does_not_hide_failure(self):
        data = output(True)
        data[0]['result'].extend([
            {'property': 'ptr.a', 'description': 'same check text', 'status': 'FAILURE'},
            {'property': 'ptr.b', 'description': 'same check text', 'status': 'SUCCESS'}])
        self.reject(data, 10, True)

    def test_counterexample_trace_required(self):
        data = output(True); del data[0]['result'][0]['trace']
        self.reject(data, 10, True)
        data = output(True); data[0]['result'][0]['trace'] = []
        self.reject(data, 10, True)

    def test_error_message_never_passes(self):
        data = output(); data.append({'messageType': 'ERROR', 'messageText': 'frontend failed'})
        self.reject(data)


class ExtractionTests(unittest.TestCase):
    def test_current_nested_block_is_extracted(self):
        block = extract((ROOT / 'crypto_montgomery_ct.c').read_text())
        self.assertEqual(len(block.block_items), 2)
        self.assertEqual(block.block_items[0].init.decls[0].name, 'k')
        self.assertEqual(block.block_items[1].lvalue.name, 'high')
        self.assertIn('high += (limb_t) carry;', c_generator.CGenerator().visit(block))
        self.assertIn('#include "source/crypto_montgomery_ct.h"', harness(block))

    def test_changed_source_mapping_rejected(self):
        source = (ROOT / 'crypto_montgomery_ct.c').read_text()
        with self.assertRaisesRegex(ValueError, 'mapping must be reviewed'):
            extract(source.replace('carry = sum >> 64;', 'carry = sum >> 63;'))

    def test_timeout_retains_failed_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / 'new-run'
            with patch('verify_carry_loop.subprocess.check_output', return_value=PIN), \
                 patch('verify_carry_loop.subprocess.run',
                       side_effect=subprocess.TimeoutExpired(['cbmc'], 1)), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(run('cbmc', output_dir, 30))
            certificate = json.loads((output_dir / 'certificate.json').read_text())
            self.assertFalse(certificate['complete'])
            self.assertFalse(certificate['passed'])
            self.assertTrue(certificate['checks'][0]['timed_out'])
            self.assertIn('TimeoutExpired', certificate['error'])
            self.assertTrue((output_dir / 'row-00.command.json').is_file())
            with self.assertRaises(FileExistsError):
                run('cbmc', output_dir, 30)


if __name__ == '__main__':
    unittest.main()
