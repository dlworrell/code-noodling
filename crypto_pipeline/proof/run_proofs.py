#!/usr/bin/env python3
"""Check the scoped 16-limb proof package; fail on unknown/error/counterexample."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FLAGS = ['--function','main','--unwind','34','--unwinding-assertions',
         '--bounds-check','--pointer-check','--pointer-overflow-check',
         '--signed-overflow-check','--undefined-shift-check','--div-by-zero-check',
         '--slice-formula']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cbmc', default='cbmc')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    version = subprocess.check_output([args.cbmc,'--version'],text=True).strip()
    if version != '6.8.0 (cbmc-6.8.0)':
        raise RuntimeError('Unpinned CBMC: '+version)
    for package, pin in [('pycparser','2.23'),('z3-solver','4.15.4.0')]:
        if importlib.metadata.version(package) != pin:
            raise RuntimeError('Unpinned dependency: '+package)
    report = {'cbmc':version,'flags':FLAGS,'complete':False,'checks':[],
              'scope':'16-limb source checks and conditional packed-state/field-contract proofs',
              'open_obligations':['independent C-to-model refinement, including memory primitive contracts',
                  'direct context byte-array verifier discrepancy',
                  'independent audit of trusted source abstractions',
                  'compiled machine-code preservation','hardware timing signal'],
              'security_status':'NOT CLEARED FOR SECRET USE'}

    def save():
        files = sorted(HERE.glob('*')) + [ROOT/'crypto_montgomery_ct.c',
                 ROOT/'crypto_montgomery_ct.h',ROOT/'crypto_multiplier.h']
        report['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in files if p.is_file()}
        report['artifact_sha256'] = {str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'report.json'}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

    def check(name, command, expected, marker):
        log = output/(name+'.log')
        timed_out = False
        try:
            with log.open('w') as stream:
                proc = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = None
            with log.open('a') as stream:
                stream.write('\nTIMEOUT: exceeded 600 seconds\n')
        passed = not timed_out and exit_code == expected and marker in log.read_text()
        report['checks'].append({'name':name,'command':command,'exit_code':exit_code,
                                 'expected_exit':expected,'timed_out':timed_out,'passed':passed})
        save()
        print(name+': '+('PASS' if passed else 'FAIL'),flush=True)
        return passed

    save()
    try:
        passed = check('memory16',[args.cbmc,str(HERE/'memory16.c'),*FLAGS],0,'VERIFICATION SUCCESSFUL')
        passed &= check('information-flow',[sys.executable,str(HERE/'check_information_flow.py'),
                                           '--output',str(output/'information-flow.json')],
                        0,'4 deliberate leaks rejected')
        passed &= check('initialization',[sys.executable,str(HERE/'check_initialization.py'),
                                          '--output',str(output/'initialization.json')],
                        0,'5 mutations rejected')
        passed &= check('setup-inverse',[sys.executable,str(HERE/'prove_setup_inverse.py'),
                                         '--output',str(output/'setup-inverse')],
                        0,'2 mutations rejected')
        passed &= check('c-steps',[sys.executable,str(HERE/'prove_c_steps.py'),
                                   '--output',str(output/'c-steps')],
                        0,'stale mapping rejected')
        passed &= check('arithmetic',[sys.executable,str(HERE/'prove_arithmetic.py'),
                                      '--output',str(output/'arithmetic')],0,'dropped_high_carry_control: sat')
        passed &= check('context-contract',[sys.executable,str(HERE/'prove_context_contract.py'),
                          '--dependency',str(output/'setup-inverse/certificate.json'),
                          '--output',str(output/'context-contract')],0,'4 mutations produce counterexamples')
        passed &= check('loop-composition',[sys.executable,str(HERE/'prove_loop_composition.py'),
                          '--dependencies',str(output),'--output',str(output/'loop-composition')],
                          0,'PASS: packed-state loop composition')
        report['complete'] = True
        report['all_scoped_checks_passed'] = passed
        return 0 if passed else 1
    finally:
        save()


if __name__ == '__main__':
    sys.exit(main())
