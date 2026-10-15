#!/usr/bin/env python3
"""CBMC checks of the AST-extracted REDC suffix carry block at n=16.

The parser, extractor, harness construction and C-to-model boundary are trusted.
This discharges a local conditional contract, not whole-kernel correctness.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from pycparser import c_ast, c_generator
from check_information_flow import parse
from prove_c_steps import SCHEMA
from restricted_c import ROOT

PIN = '6.8.0 (cbmc-6.8.0)'
LABELS = {'MATH_INVARIANT', 'FRAME_ISOLATION', 'TERMINAL_CARRY_BIT'}
FLAGS = ['--function', 'main', '--unwind', '33', '--unwinding-assertions',
         '--bounds-check', '--pointer-check', '--pointer-overflow-check',
         '--signed-overflow-check', '--unsigned-overflow-check',
         '--undefined-shift-check', '--div-by-zero-check',
         '--all-properties', '--slice-formula', '--json-ui', '--trace']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def loop_counter(node):
    if not isinstance(node, c_ast.For) or not isinstance(node.init, c_ast.DeclList):
        return None
    return node.init.decls[0].name if len(node.init.decls) == 1 else None


def extract(source):
    """Recognize the complete kernel, then take its adjacent propagation/update."""
    function = parse(source)
    generator = c_generator.CGenerator()
    if sha(generator.visit(function).encode()) != SCHEMA:
        raise ValueError('C kernel changed; carry extraction mapping must be reviewed')
    matches = []
    for outer in function.body.block_items:
        if loop_counter(outer) != 'i' or not isinstance(outer.stmt, c_ast.Compound):
            continue
        items = outer.stmt.block_items or []
        for index, node in enumerate(items[:-1]):
            update = items[index + 1]
            if (loop_counter(node) == 'k' and isinstance(update, c_ast.Assignment)
                    and isinstance(update.lvalue, c_ast.ID)
                    and update.lvalue.name == 'high' and update.op == '+='):
                matches.append(c_ast.Compound(block_items=copy.deepcopy([node, update])))
    if len(matches) != 1:
        raise ValueError('Expected exactly one adjacent REDC carry loop/high update')
    return matches[0]


def harness(block, mutation=None):
    """Use actual header types and source statements; alter only explicit controls."""
    block = copy.deepcopy(block)
    prefix = ''
    if mutation == 'drop-incoming':
        prefix = 'carry = 0;'
    elif mutation == 'drop-terminal':
        block.block_items[1].rvalue = c_ast.Constant('int', '0')
    elif mutation == 'truncate-propagation':
        block.block_items[0].cond.right = c_ast.BinaryOp(
            '+', c_ast.BinaryOp('+', c_ast.ID('i'), c_ast.ID('n')), c_ast.Constant('int', '1'))
    elif mutation is not None:
        raise ValueError('Unknown mutation: ' + mutation)
    extracted = c_generator.CGenerator().visit(block)
    return '''/* Generated from the recognized production AST; see certificate.json. */
#include "source/crypto_montgomery_ct.h"
#include <limits.h>
_Static_assert(sizeof(limb_t) * CHAR_BIT == 64, "64-bit limbs");
_Static_assert(sizeof(dlimb_t) * CHAR_BIT == 128, "128-bit intermediates");
_Static_assert((limb_t)-1 > 0 && (dlimb_t)-1 > 0, "unsigned words");
typedef unsigned __CPROVER_bitvector[1025] packed_t;
extern limb_t nondet_limb(void);
extern size_t nondet_size(void);

static packed_t pack(const limb_t *words, size_t length) {
    packed_t value = 0;
    for (size_t j = 0; j < length; ++j)
        value |= (packed_t)words[j] << (64 * j);
    return value;
}

int main(void) {
    const size_t n = 16;
    const size_t i = CT_ROW;
    _Static_assert(CT_ROW >= 0 && CT_ROW < 16, "valid REDC offset");
    limb_t t[32];
    for (size_t j = 0; j < 32; ++j) t[j] = nondet_limb();
    dlimb_t carry = (dlimb_t)nondet_limb();
    limb_t high = 0;
    size_t length = n - i;
    packed_t expected = pack(t + i + n, length) + (packed_t)carry;
    size_t low_idx = nondet_size();
    __CPROVER_assume(low_idx < i + n);
    limb_t old_low = t[low_idx];
''' + prefix + '\n' + extracted + '''
    packed_t actual = pack(t + i + n, length) + ((packed_t)high << (64 * length));
    __CPROVER_assert(actual == expected, "MATH_INVARIANT");
    __CPROVER_assert(t[low_idx] == old_low, "FRAME_ISOLATION");
    __CPROVER_assert(carry <= 1, "TERMINAL_CARRY_BIT");
    return 0;
}
'''


def evaluate_json(text, returncode, mutation=False):
    """Fail closed on malformed/incomplete output, wrong exits, or other failures."""
    data = json.loads(text)
    if not isinstance(data, list) or not data or any(not isinstance(x, dict) for x in data):
        raise ValueError('Expected CBMC JSON message array')
    if any(x.get('messageType') == 'ERROR' for x in data):
        raise ValueError('CBMC reported an error')
    result_sets = [x['result'] for x in data if 'result' in x]
    statuses = [x['cProverStatus'] for x in data if 'cProverStatus' in x]
    if len(result_sets) != 1 or not isinstance(result_sets[0], list) or not result_sets[0]:
        raise ValueError('Missing/ambiguous CBMC property results')
    properties = result_sets[0]
    if any(not isinstance(p, dict) for p in properties):
        raise ValueError('Invalid CBMC property record')
    ids = [p.get('property') for p in properties]
    if any(not isinstance(p, str) or not p for p in ids) or len(set(ids)) != len(ids):
        raise ValueError('Missing/duplicate property identifiers')
    if any(p.get('status') not in ('SUCCESS', 'FAILURE') for p in properties):
        raise ValueError('Unresolved CBMC property')
    for label in LABELS:
        if sum(p.get('description') == label for p in properties) != 1:
            raise ValueError('Missing/duplicate required property: ' + label)
    failed = [p for p in properties if p['status'] == 'FAILURE']
    if not mutation:
        if returncode != 0 or statuses != ['success'] or failed:
            raise ValueError('Pristine verification did not succeed completely')
    else:
        if (returncode != 10 or statuses != ['failure'] or len(failed) != 1
                or failed[0].get('description') != 'MATH_INVARIANT'):
            raise ValueError('Mutation must fail only MATH_INVARIANT with exit 10')
        trace = failed[0].get('trace')
        if not isinstance(trace, list) or not any(x.get('stepType') == 'failure' for x in trace):
            raise ValueError('Missing arithmetic counterexample trace')
    return {'property_count': len(properties),
            'properties': [{k: p[k] for k in ('property', 'description', 'status')} for p in properties],
            'failed_property_ids': [p['property'] for p in failed]}


def run(cbmc, output, timeout):
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + timeout
    report = {'passed': False, 'complete': False, 'limbs': 16, 'packed_bits': 1025,
              'schema_sha256': SCHEMA, 'checks': [],
              'scope': 'All 16 REDC suffix offsets; arbitrary initialized words and full-word incoming carry; initial high=0',
              'trusted': ['restricted parser and AST extraction', 'harness construction and header/type binding',
                          'pycparser 2.23', 'CBMC 6.8.0 and its frontend/preprocessor/SAT solver'],
              'open': ['MAC establishes the incoming full-word carry range',
                       'outer REDC invariants and retained high accumulation across rows',
                       'independent frontend/refinement verification', 'machine-code preservation and hardware timing']}

    def save():
        report['artifact_sha256'] = {str(p.relative_to(output)): sha(p.read_bytes())
            for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('certificate.json', 'certificate.json.tmp')}
        tmp = output / 'certificate.json.tmp'
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(output / 'certificate.json')

    try:
        save()
        report['cbmc'] = subprocess.check_output([cbmc, '--version'], text=True, timeout=10).strip()
        if report['cbmc'] != PIN or importlib.metadata.version('pycparser') != '2.23':
            raise ValueError('Unpinned CBMC/pycparser')
        snapshot = output / 'source'
        snapshot.mkdir()
        for name in ('crypto_montgomery_ct.c', 'crypto_montgomery_ct.h', 'crypto_multiplier.h'):
            (snapshot / name).write_bytes((ROOT / name).read_bytes())
        source = (snapshot / 'crypto_montgomery_ct.c').read_text()
        report['source_sha256'] = sha(source.encode())
        report['header_sha256'] = {p.name: sha(p.read_bytes()) for p in snapshot.glob('*.h')}
        block = extract(source)
        extracted = c_generator.CGenerator().visit(block)
        (output / 'extracted.c').write_text(extracted + '\n')
        report['extracted_sha256'] = sha(extracted.encode())
        # A stale arithmetic mapping must be rejected, not silently reinterpreted.
        changed = source.replace('carry = sum >> 64;', 'carry = sum >> 63;')
        if changed == source:
            raise ValueError('Source mutation anchor drift')
        try:
            extract(changed)
        except ValueError as error:
            if 'mapping must be reviewed' not in str(error):
                raise
        else:
            raise ValueError('Changed source mapping accepted')
        report['stale_source_rejected'] = True
        configurations = [(f'row-{row:02d}', row, None) for row in range(16)]
        configurations += [(name, 0, name) for name in
                           ('drop-incoming', 'drop-terminal', 'truncate-propagation')]
        for name, row, mutation in configurations:
            record = {'name': name, 'row': row, 'mutation': mutation, 'passed': False}
            report['checks'].append(record)
            c_file = output / (name + '.c')
            c_file.write_text(harness(block, mutation))
            argv = [cbmc, str(c_file), f'-DCT_ROW={row}', *FLAGS]
            record['command'] = argv
            (output / (name + '.command.json')).write_text(json.dumps(argv, indent=2) + '\n')
            save()
            remaining = min(60, deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError('Carry verification total time budget exhausted')
            try:
                with (output / (name + '.json')).open('w') as stdout, (output / (name + '.stderr')).open('w') as stderr:
                    result = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=remaining)
                record['returncode'] = result.returncode
                record.update(evaluate_json((output / (name + '.json')).read_text(),
                                            result.returncode, mutation is not None))
                record['passed'] = True
            except subprocess.TimeoutExpired:
                record['timed_out'] = True
                raise
            print(name + ': ' + ('counterexample isolated' if mutation else 'proved'), flush=True)
            save()
        report['complete'] = True
        report['passed'] = all(x['passed'] for x in report['checks'])
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        print(report['error'], file=sys.stderr, flush=True)
    finally:
        save()
    return report['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cbmc', default='cbmc')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=480)
    args = parser.parse_args()
    if not 0 < args.timeout <= 540:
        parser.error('Need 0 < timeout <= 540 seconds')
    passed = run(args.cbmc, args.output.resolve(), args.timeout)
    if passed:
        print('PASS: 16 extracted carry offsets; 3 arithmetic mutations isolated; stale mapping rejected')
    sys.exit(0 if passed else 1)
