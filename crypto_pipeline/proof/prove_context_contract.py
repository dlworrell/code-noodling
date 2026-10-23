#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:06a426d2-6802-4f47-a12b-e97fad910a8e

"""Field-level setup proof under explicit memset/memcpy word contracts.

Exact setup AST matching binds this specialized model to the implementation.
No claim is made that CBMC's original byte-array discrepancy is resolved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import z3
from prove_setup_inverse import extract,SCHEMA
from restricted_c import ROOT

N=16
CAP=1024


def run(source,out,dependency,mutation=None):
    initial,count,canonical=extract(source)
    digest=hashlib.sha256(source.encode()).hexdigest()
    dep=json.loads(dependency.read_text())
    if not dep['passed'] or dep['source_sha256']!=digest:
        raise ValueError('setup inverse prerequisite failed or stale')
    bv=z3.BitVecSort(64)
    modulus=z3.Array('input_modulus',z3.IntSort(),bv)
    old=z3.Array('old_context_modulus',z3.IntSort(),bv)
    oldlen,oldinv,inverse=z3.BitVecs('old_length old_inverse computed_inverse',64)
    witness=z3.Int('witness')
    ctx_null,mod_null,overlap=z3.Bools('ctx_null mod_null overlap')
    count=z3.BitVec('count',64)
    upper=z3.BitVecVal(0,64)
    for i in range(1,N):upper=upper|modulus[i]
    invalid=z3.Or(ctx_null,mod_null,overlap,count==0,z3.UGT(count,CAP))
    domain=z3.And((modulus[0]&1)==1,z3.Or(upper!=0,z3.UGT(modulus[0],1)))
    success=z3.And(z3.Not(invalid),domain)
    # Only successful n=16 is included; invalid count covers all 0 or >1024.
    conditions=[z3.Or(count==N,count==0,z3.UGT(count,CAP)),0<=witness,witness<CAP]
    zeroed=old if mutation=='missing-clear' else z3.K(z3.IntSort(),z3.BitVecVal(0,64))
    copied=zeroed
    for i in range(N-(mutation=='short-copy')):
        copied=z3.Store(copied,i,modulus[i])
    after=z3.If(success,copied,old)
    after_len=z3.If(success,z3.BitVecVal(N,64),oldlen)
    after_inv=z3.If(success,inverse,z3.BitVecVal(0,64) if mutation=='error-write' else oldinv)
    rc=z3.If(success,0,-1)
    # Abstract error names, not platform-dependent numeric errno values.
    errno=z3.If(success,0,z3.If(invalid,1,2))
    if mutation=='wrong-error':errno=z3.If(success,0,2)
    records=[]
    def prove(name,premises,claim):
        solver=z3.Solver();solver.set(timeout=30000)
        solver.add(*conditions,*premises,z3.Not(claim))
        (out/(name+'.smt2')).write_text(solver.to_smt2())
        status=solver.check()
        item={'name':name,'result':str(status)}
        if status==z3.sat:item['counterexample']=str(solver.model())
        if status==z3.unknown:item['reason']=solver.reason_unknown()
        records.append(item)
    prove('copy_prefix',[success,witness<N],after[witness]==modulus[witness])
    prove('zero_unused',[success,witness>=N],after[witness]==0)
    prove('store_length',[success],after_len==N)
    prove('store_computed_inverse',[success],after_inv==inverse)
    prove('preserve_all_fields_on_error',[z3.Not(success)],
          z3.And(after[witness]==old[witness],after_len==oldlen,after_inv==oldinv))
    prove('public_argument_errors',[invalid],z3.And(rc==-1,errno==1))
    prove('modulus_domain_errors',[z3.Not(invalid),z3.Not(domain)],z3.And(rc==-1,errno==2))
    prove('success_status',[success],z3.And(rc==0,errno==0))
    # Region contracts: 16 copied words are inside both the 16-word input and
    # 1024-word context; zeroing covers all context fields before assignments.
    k=z3.Int('copy_index')
    prove('copy_region_bounds',[success,0<=k,k<N],z3.And(k<CAP,8*k+8<=8*N))
    report={'source_sha256':digest,'schema_sha256':SCHEMA,
            'dependency_sha256':hashlib.sha256(dependency.read_bytes()).hexdigest(),
            'obligations':records,'passed':all(r['result']=='unsat' for r in records),
            'scope':'typed context fields: success at n=16, null pointers, rejected overlap, n=0, all n>1024, invalid modulus at n=16; padding bytes excluded',
            'trusted':['exact-AST specialization','word contracts for memset zero and nonoverlapping memcpy','C layout/64-bit unsigned-word mapping; flat-address overlap predicate specialization','Z3'],
            'primitive_contracts':{'memset':'zero every context field; ignores padding observations',
                                   'memcpy':'copy exactly n initialized aligned nonoverlapping limbs'},
            'not_claimed':'independent byte-level C proof or resolution of the earlier CBMC discrepancy'}
    (out/'certificate.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dependency',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'crypto_montgomery_ct.c').read_text()
    result=run(source,args.output,args.dependency)
    controls=[]
    for name,expected in [('short-copy','copy_prefix'),('missing-clear','zero_unused'),
                          ('error-write','preserve_all_fields_on_error'),('wrong-error','public_argument_errors')]:
        path=args.output/name;path.mkdir(exist_ok=True)
        bad=run(source,path,args.dependency,name)
        target=next(x for x in bad['obligations'] if x['name']==expected)
        if target['result']!='sat':raise ValueError('mutation must produce a concrete counterexample: '+name)
        controls.append({'name':name,'counterexample_property':expected})
    result['negative_controls']=controls
    (args.output/'certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit('context contract proof failed')
    print('PASS: context field contracts; 4 mutations produce counterexamples')
