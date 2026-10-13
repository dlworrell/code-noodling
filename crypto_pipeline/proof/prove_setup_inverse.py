#!/usr/bin/env python3
"""Source-bound modular lifting proof for the recognized setup implementation.

Canonical AST matching is a trusted specialization boundary. Changes beyond the
initial inverse and iteration count are rejected, not silently reinterpreted.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
from pycparser import c_ast,c_parser,c_generator
import z3
from restricted_c import ROOT,PRELUDE

B=1<<64
SCHEMA='b85da1e107a1d41d9b035936b053af241ed991cd678ed5ad7139c6aa2c183833'


def extract(source):
    text=re.sub(r'^\s*#include[^\n]*','',source,flags=re.M)
    tokens=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*',re.S)
    text=tokens.sub(lambda m:'' if m.group().startswith('/') else m.group(),text)
    ast=c_parser.CParser().parse(PRELUDE+text)
    functions=[n for n in ast.ext if isinstance(n,c_ast.FuncDef) and n.decl.name=='crypto_montgomery_ctx_init']
    if len(functions)!=1: raise ValueError('setup definition missing')
    function=functions[0]
    if len(function.body.block_items)!=12: raise ValueError('unsupported setup schema')
    initial=function.body.block_items[4].init
    count=function.body.block_items[5].cond.right
    if not isinstance(initial,c_ast.Constant) or not isinstance(count,c_ast.Constant):
        raise ValueError('setup constants required')
    initial_value=int(initial.value,0);iterations=int(count.value,0)
    if not 0<=initial_value<B or not 0<=iterations<=6: raise ValueError('unsupported setup constants')
    normalized=copy.deepcopy(function)
    normalized.body.block_items[4].init=c_ast.Constant('int','1')
    normalized.body.block_items[5].cond.right=c_ast.Constant('int','6')
    canonical=c_generator.CGenerator().visit(normalized)
    if hashlib.sha256(canonical.encode()).hexdigest()!=SCHEMA:
        raise ValueError('setup schema changed; proof mapping must be reviewed')
    return initial_value,iterations,canonical


def prove(source,out):
    initial,iterations,canonical=extract(source)
    records=[]
    def obligation(name,premises,conclusion):
        solver=z3.Solver();solver.set(timeout=30000)
        solver.add(*premises,z3.Not(conclusion))
        (out/(name+'.smt2')).write_text(solver.to_smt2())
        status=solver.check()
        records.append({'name':name,'result':str(status)})
        if status==z3.sat: records[-1]['counterexample']=str(solver.model())
        if status==z3.unknown: records[-1]['reason']=solver.reason_unknown()
        return status==z3.unsat

    m,x,Q,p,s,q=z3.Ints('m x Q product_wrap subtract_wrap result_wrap')
    passed=obligation('base_odd_inverse',[m%2==1],(m*initial-1)%2==0)
    bits=1
    for i in range(iterations):
        M=1<<bits
        # Model every unsigned 64-bit C intermediate, with an integer wrap
        # quotient for each multiplication/subtraction. Each real execution
        # has such quotients; no bound or guessed value is assumed for them.
        product=m*x-B*p
        subtract=2-product-B*s
        result=x*subtract-B*q
        witness=-Q*Q+(B//(M*M))*(m*x*(p-s)-m*q)
        passed &= obligation(f'lift_{bits}_to_{bits*2}',[m*x==1+M*Q],
                             m*result-1==M*M*witness)
        bits*=2
    neg,wrap=z3.Ints('negative_inverse negation_wrap')
    passed &= obligation('stored_negative_inverse',[m*x==1+B*Q,neg==-x+B*wrap],
                         m*neg+1==B*(m*wrap-Q))
    # The final lemma is conditional on 64-bit precision. Do not claim the
    # setup theorem when the source loop fails to establish that premise.
    precision=bits>=64
    certificate={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                 'schema_sha256':SCHEMA,'initial_inverse':initial,'iterations':iterations,
                 'established_precision_bits':bits,'precision_sufficient':precision,
                 'obligations':records,'passed':bool(passed and precision),
                 'conclusion':'For odd modulus[0], stored inverse satisfies m[0]*inverse = -1 modulo 2^64',
                 'trusted':['AST specialization','unsigned 64-bit C/modular-integer correspondence','pycparser','Z3']}
    (out/'certificate.json').write_text(json.dumps(certificate,indent=2)+'\n')
    (out/'recognized_setup.c').write_text(canonical+'\n')
    return certificate


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'crypto_montgomery_ct.c').read_text()
    result=prove(source,args.output)
    controls=[]
    for name,old,new in [('five-iterations','i < 6','i < 5'),('zero-initial-inverse','limb_t inv = 1;','limb_t inv = 0;')]:
        if source.count(old)!=1: raise ValueError('mutation anchor drift')
        path=args.output/name;path.mkdir(exist_ok=True)
        bad=prove(source.replace(old,new),path)
        if bad['passed']: raise ValueError('missed broken setup '+name)
        controls.append({'name':name,'rejected':True,'precision_bits':bad['established_precision_bits']})
    result['negative_controls']=controls
    (args.output/'certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']: raise SystemExit('setup inverse proof failed')
    print('PASS: source-bound 64-bit setup inverse; 2 mutations rejected')
