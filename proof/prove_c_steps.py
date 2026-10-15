#!/usr/bin/env python3
"""Local arithmetic refinements tied to the exact current C kernel AST.

A trusted specialization maps recognized C statements to integer/bit-vector
semantics. The whole-loop invariant composition is explicitly not proved here.
"""
import argparse
import hashlib
import json
from pathlib import Path
from pycparser import c_generator
import z3
from check_information_flow import parse
from restricted_c import ROOT

SCHEMA='172b99ca3c7c9e045903e954b7ef1def968706e8f7f019c0fb4661a51fc9deac'
B=1<<64


def run(source,out):
    function=parse(source);gen=c_generator.CGenerator()
    canonical=gen.visit(function)
    if hashlib.sha256(canonical.encode()).hexdigest()!=SCHEMA:
        raise ValueError('C kernel changed; local refinement mapping must be reviewed')
    statements=function.body.block_items
    bindings={
        'product-row':gen.visit(statements[6]),
        'redc-row-and-carry':gen.visit(statements[8]),
        'borrow-loop':gen.visit(statements[10]),
        'mask':gen.visit(statements[11]),
        'selection-loop':gen.visit(statements[12])}
    records=[]
    def prove(name,assumptions,claim,binding):
        solver=z3.Solver();solver.set(timeout=30000)
        solver.add(*assumptions,z3.Not(claim))
        (out/(name+'.smt2')).write_text(solver.to_smt2())
        result=solver.check()
        records.append({'name':name,'result':str(result),'source_binding':binding})
        if result==z3.sat: records[-1]['counterexample']=str(solver.model())
        if result==z3.unknown: records[-1]['reason']=solver.reason_unknown()
    x,y,t,c=z3.Ints('x y t incoming_carry')
    word=lambda v:z3.And(0<=v,v<B)
    raw=x*y+t+c
    domain=[word(v) for v in (x,y,t,c)]
    prove('source_mac_no_128_overflow',domain,z3.And(0<=raw,raw<B*B),'product-row; redc-row-and-carry')
    total=z3.Int('total')
    # Proven range discharges the wrapping step. Unsigned narrowing and logical
    # >>64 correspond respectively to remainder/division by B.
    wrapped=total%(B*B)
    prove('source_mac_store_and_shift',[0<=total,total<B*B],
          z3.And(wrapped==total,wrapped%B+B*(wrapped/B)==total,word(wrapped/B)),
          'product-row; redc-row-and-carry')
    small=t+c
    prove('source_carry_propagation',[word(t),word(c)],
          z3.And(0<=small,small<2*B,small%B+B*(small/B)==small,
                 z3.Or(small/B==0,small/B==1)),'redc-row-and-carry')
    m,borrow=z3.Ints('modulus_limb borrow_in')
    sub=m+borrow
    delta=t-sub
    # C subtracts in unsigned 128 bits, then narrows to unsigned 64.
    low=(delta%(B*B))%B
    next_borrow=z3.If(t<sub,1,0)
    prove('source_borrow_and_narrow',[word(t),word(m),z3.Or(borrow==0,borrow==1)],
          z3.And(0<=sub,sub<=B,low==delta%B,delta==low-B*next_borrow),
          'borrow-loop')
    hi,br,old,diff=z3.BitVecs('high borrow old difference',64)
    mask=- (hi | (br ^ 1))
    prove('source_mask_and_selection',[z3.ULE(hi,1),z3.ULE(br,1)],
          ((diff&mask)|(old&~mask))==z3.If(z3.Or(hi==1,br==0),diff,old),
          'mask; selection-loop')
    report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'schema_sha256':SCHEMA,'bindings':bindings,'obligations':records,
            'passed':all(x['result']=='unsat' for x in records),
            'trusted':['AST specialization and local C semantics mapping','pycparser','Z3'],
            'open':'Whole-loop induction: product packing, REDC coefficient accumulation/top carry, and global postcondition'}
    (out/'certificate.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'crypto_montgomery_ct.c').read_text()
    report=run(source,args.output)
    # A source edit must not keep using the old hand-audited mapping silently.
    changed=source.replace('carry = sum >> 64;', 'carry = sum >> 63;')
    if changed==source: raise ValueError('mutation anchor drift')
    try: run(changed,args.output/'must-not-be-used')
    except ValueError as error:
        if 'mapping must be reviewed' not in str(error): raise
    else: raise ValueError('stale source mapping accepted')
    if not report['passed']: raise SystemExit('local refinement failed')
    print('PASS: 5 source-bound C arithmetic steps; stale mapping rejected')
