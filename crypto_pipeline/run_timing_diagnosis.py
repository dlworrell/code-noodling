#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:51e2ffe8-9f15-461d-acf2-a9f1f98570e9

"""Retain a predeclared timing/control matrix, including failures. Linux/x86 only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PIN = "dc269651fb2567e46755cfb2a13d3875592968b5"
ROOT = Path(__file__).resolve().parent


def hashes(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
            for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dudect', type=Path, default=ROOT / 'vendor/dudect')
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory')
    parser.add_argument('--compiler', default='gcc')
    parser.add_argument('--optimization', choices=['O2', 'O3'], default='O2')
    parser.add_argument('--cpu', type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument('--batches', type=int, default=101)
    parser.add_argument('--limbs', type=int, choices=[4, 16, 32])
    parser.add_argument('--timeout', type=float, default=400)
    args = parser.parse_args()
    if args.batches < 2 or args.cpu not in os.sched_getaffinity(0) or not 0 < args.timeout <= 400:
        parser.error('Need >=2 batches, an allowed CPU, and 0 < timeout <= 400')
    vendor = args.dudect.resolve()
    revision = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        parser.error('Unexpected dudect revision')
    subprocess.run(['git', '-C', str(vendor), 'diff', '--exit-code', 'HEAD', '--', 'src/dudect.h'], check=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / n for n in ('crypto_montgomery_ct.c', 'crypto_montgomery_ct.h',
               'crypto_multiplier.h', 'dudect_montgomery.c')]
    sources += [Path(__file__).resolve(), vendor / 'src/dudect.h']
    modes = [('leaky-control', 'CT_TIMING_NEGATIVE_CONTROL'),
             ('public-baseline', 'CT_TIMING_BASELINE'), ('input-only', 'CT_TIMING_INPUT_ONLY'),
             ('random-labels', 'CT_TIMING_RANDOM_LABELS'), ('candidate', None)]
    flags = ['-' + args.optimization, '-fno-lto', '-fPIC', '-Wall', '-Wextra']
    report = {'dudect': PIN, 'compiler': args.compiler, 'flags': flags,
              'cpu': args.cpu, 'batches': args.batches, 'timeout': args.timeout,
              'security_status': 'NOT CLEARED FOR SECRET USE',
              'affinity_caveat': 'Affinity is not isolation from other tasks, SMT or interrupts.',
              'matrix_completed': False, 'diagnostic_failure': True,
              'source_modified_mid_run': False, 'initial_source_hashes': hashes(sources),
              'commands': [], 'plan': [dict(mode=name, macro=macro, limbs=n, fixed_class=k,
                  status='pending', outcome=None, returncode=None)
                  for name, macro in modes for n in ([args.limbs] if args.limbs else [4,16,32])
                  for k in (0,1)]}
    stage = 'snapshot'

    def save():
        current = hashes(sources)
        report['current_source_hashes'] = current
        report['source_modified_mid_run'] |= current != report['initial_source_hashes']
        if report['source_modified_mid_run']:
            report['diagnostic_failure'] = True
        report['artifact_sha256'] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('report.json', 'report.json.tmp')}
        tmp = output / 'report.json.tmp'
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        os.replace(tmp, output / 'report.json')

    def command(argv, logname, timeout=60):
        entry = {'stage': stage, 'argv': argv, 'log': logname, 'returncode': None}
        report['commands'].append(entry)
        save()
        try:
            with (output / logname).open('w') as log:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            entry['returncode'] = result.returncode
            return result.returncode
        except subprocess.TimeoutExpired:
            entry['timeout'] = True
            return None

    try:
        save()
        if any(v is None for v in report['initial_source_hashes'].values()):
            raise RuntimeError('Missing required source file')
        snapshot = output / 'sources'
        snapshot.mkdir()
        for p in sources:
            data = p.read_bytes()
            if hashlib.sha256(data).hexdigest() != report['initial_source_hashes'][str(p)]:
                raise RuntimeError('Source changed while taking snapshot')
            (snapshot / p.name).write_bytes(data)
        # Compile only the recorded snapshot, including the pinned dudect header.
        stage = 'environment'
        for i, argv in enumerate(([args.compiler, '--version'], ['uname','-a'], ['lscpu'])):
            if command(argv, f'environment-{i}.log') != 0:
                raise RuntimeError('Environment command failed')
        with (output / 'cpu.txt').open('w') as log:
            for suffix in ('cpufreq/scaling_governor', 'topology/thread_siblings_list'):
                p = Path(f'/sys/devices/system/cpu/cpu{args.cpu}') / suffix
                try:
                    value = p.read_text()
                except OSError as error:
                    value = f'unavailable: {error}\n'
                log.write(str(p) + ': ' + value)
        stage = 'candidate_compilation'
        obj = output / 'candidate.o'
        if command([args.compiler, '-std=c11', *flags, '-c', str(snapshot / 'crypto_montgomery_ct.c'), '-o', str(obj)], 'candidate-build.log') != 0:
            raise RuntimeError('Candidate compilation failed; matrix entries not executed')
        stage = 'disassembly'
        if command(['objdump', '-drwC', str(obj)], 'candidate.asm') != 0:
            raise RuntimeError('Disassembly failed')
        for item in report['plan']:
            tag = f"{item['mode']}-L{item['limbs']}-C{item['fixed_class']}"
            binary = output / tag
            stage = tag + ':build'
            item['status'] = 'building'
            argv = [args.compiler, '-std=gnu11', *flags, '-I' + str(snapshot),
                    f"-DCT_LIMBS={item['limbs']}", f"-DCT_FIXED_CLASS={item['fixed_class']}",
                    f'-DCT_BATCHES={args.batches}']
            if item['macro']:
                argv.append('-D' + item['macro'])
            argv += [str(snapshot / 'dudect_montgomery.c'), str(obj), '-lm', '-o', str(binary)]
            item['build_cmd'] = argv
            code = command(argv, tag + '.build.log')
            if code != 0:
                item.update(status='build_failed', outcome='compilation_failed', returncode=code)
                save()
                continue
            stage = tag + ':execution'
            item['status'] = 'running'
            argv = ['taskset', '-c', str(args.cpu), str(binary)]
            item['exec_cmd'] = argv
            code = command(argv, tag + '.log', args.timeout)
            outcome = ('timeout' if code is None else 'threshold_crossed' if code == 1
                       else 'no_crossing' if code == 0 else 'execution_error')
            # taskset itself can return 1: require the harness completion marker.
            text = (output / (tag + '.log')).read_text()
            marker = ('LEAKAGE EVIDENCE DETECTED' if code == 1 else
                      'NO LEAKAGE DETECTED IN THIS BOUNDED RUN; NOT A PROOF')
            if code in (0, 1) and marker not in text:
                outcome = 'execution_error'
            item.update(status='timeout' if code is None else 'completed', outcome=outcome, returncode=code)
            print(tag + ': ' + outcome, flush=True)
            save()
        report['matrix_completed'] = True  # All configurations attempted, not necessarily successful.
        save()
        report['diagnostic_failure'] = report['source_modified_mid_run'] or any(
            r['status'] != 'completed' or r['outcome'] !=
            ('threshold_crossed' if r['mode'] == 'leaky-control' else 'no_crossing')
            for r in report['plan'])
    except Exception as error:
        report.update(failed_stage=stage, exception_message=str(error), diagnostic_failure=True)
        print(f'{stage}: {error}', file=sys.stderr)
    finally:
        save()
    return int(report['diagnostic_failure'])


if __name__ == '__main__':
    sys.exit(main())
