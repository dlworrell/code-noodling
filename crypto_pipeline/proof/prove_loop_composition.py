#!/usr/bin/env python3
"""16-limb packed-state composition for the exact recognized C kernel.

Source-to-equation specialization and lemma composition are trusted proof glue.
This is not an independently verified C frontend or a machine-code theorem.
"""
import argparse
import hashlib
import json
from pathlib import Path
import z3
from prove_c_steps import SCHEMA
from check_information_flow import parse
from pycparser import c_generator
from restricted_c import ROOT

B=1<<64
N=16
R=B**N


def prove_all(out,dependencies):
    source=(ROOT/'crypto_montgomery_ct.c').read_text()
    digest=hashlib.sha256(source.encode()).hexdigest()
    canonical=c_generator.CGenerator().visit(parse(source))
    if hashlib.sha256(canonical.encode()).hexdigest()!=SCHEMA:
        raise ValueError('kernel mapping drift')
    expected={
        'c-steps/certificate.json':5,
        'setup-inverse/certificate.json':8,
        'arithmetic/arithmetic.json':14}
    deps={}
    for name,count in expected.items():
        path=dependencies/name
        certificate=json.loads(path.read_text())
        results=certificate.get('obligations',certificate.get('results',[]))
        positives=[r for r in results if r.get('expected','unsat')=='unsat']
        if len(positives)!=count or any(r['result']!='unsat' for r in positives):
            raise ValueError('unproved dependency '+name)
        if name!='arithmetic/arithmetic.json' and (not certificate.get('passed') or certificate['source_sha256']!=digest):
            raise ValueError('stale/failed source dependency '+name)
        deps[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    init_path=dependencies/'initialization.json'
    init=json.loads(init_path.read_text())
    if init['source_sha256']!=digest or not init['output_initialized'] or not init['scratch_arrays_zero_on_return']:
        raise ValueError('stale/failed initialization dependency')
    deps['initialization.json']=hashlib.sha256(init_path.read_bytes()).hexdigest()
    carry_path=dependencies/'carry-loop/certificate.json'
    carry_certificate=json.loads(carry_path.read_text())
    if (not carry_certificate.get('passed') or not carry_certificate.get('complete')
            or carry_certificate.get('source_sha256')!=digest
            or carry_certificate.get('schema_sha256')!=SCHEMA
            or carry_certificate.get('limbs')!=N or carry_certificate.get('packed_bits')!=1025):
        raise ValueError('stale/failed extracted carry dependency')
    carry_checks=carry_certificate.get('checks',[])
    required_carry=[(f'row-{row:02d}',row,None,0) for row in range(N)]
    required_carry += [(name,0,name,10) for name in
                      ('drop-incoming','drop-terminal','truncate-propagation')]
    if ([(r.get('name'),r.get('row'),r.get('mutation'),r.get('returncode')) for r in carry_checks]
            != required_carry or any(not r.get('passed') for r in carry_checks)):
        raise ValueError('incomplete extracted carry checks')
    deps['carry-loop/certificate.json']=hashlib.sha256(carry_path.read_bytes()).hexdigest()
    records=[]
    def prove(name,premises,claim,expected="unsat"):
        solver=z3.Solver();solver.set(timeout=30000)
        solver.add(*premises,z3.Not(claim))
        (out/(name+'.smt2')).write_text(solver.to_smt2())
        status=solver.check()
        item={'name':name,'result':str(status),'expected':expected}
        if status==z3.sat:item['counterexample']=str(solver.model())
        if status==z3.unknown:item['reason']=solver.reason_unknown()
        records.append(item)
        print(name+': '+str(status),flush=True)

    def pack(words):return sum(word*(B**i) for i,word in enumerate(words))
    old=[z3.Int(f'old_{j}') for j in range(N)]
    new=[z3.Int(f'new_{j}') for j in range(N)]
    y=[z3.Int(f'y_{j}') for j in range(N)]
    carry=[z3.Int(f'carry_{j}') for j in range(N+1)]
    x=z3.Int('x')
    mac=[new[j]+B*carry[j+1]==old[j]+x*y[j]+carry[j] for j in range(N)]
    # Explicit telescoping: all internal carries cancel, only the row-end
    # carry remains. Premises are exactly the source-bound MAC contract.
    prove('mac_row_telescopes',[carry[0]==0,*mac],
          pack(new)+R*carry[N]==pack(old)+x*pack(y))
    # Outer product invariant and zero overwrite. Previous row r writes only
    # indices r..r+16; thus at row i, index i+16 has never been written.
    footprints=[]
    initialized=set(range(2*N))
    ever_written=set()
    for i in range(N):
        target=i+N
        if target in ever_written:raise ValueError('carry overwrite not known zero')
        reads=set(range(i,i+N));writes=reads|{target}
        if not reads<=initialized or max(writes)>=2*N:raise ValueError('bad product footprint')
        footprints.append({'row':i,'zero_carry_destination':target,'writes':sorted(writes)})
        ever_written |= writes
    A,Y,T,W=z3.Ints('prefix_A whole_B packed_T radix_weight')
    prove('product_outer_step',[T==A*Y],T+x*W*Y==(A+x*W)*Y)
    a=[z3.Int(f'a_{i}') for i in range(N)]
    prefix=z3.IntVal(0)
    for i in range(N):prefix=prefix+a[i]*B**i
    prove('product_prefix_complete',[],prefix==pack(a))
    # The extracted CBMC block establishes this local suffix identity for each
    # offset, given an incoming full-word carry and initial high=0. Connecting
    # the MAC and existing outer high state remains trusted composition glue.
    for i in range(N):
        length=N-i
        before=[z3.Int(f'cp_{i}_old_{j}') for j in range(length)]
        after=[z3.Int(f'cp_{i}_new_{j}') for j in range(length)]
        c=[z3.Int(f'cp_{i}_c_{j}') for j in range(length+1)]
        premises=[after[j]+B*c[j+1]==before[j]+c[j] for j in range(length)]
        prove(f'propagation_row_{i}',premises,
              pack(after)+B**length*c[-1]==pack(before)+c[0])

    # Combine both windows, at all actual offsets, retaining the extra limb.
    low,oldwin,newwin,oldtop,newtop,cend,cout,H,M,U=z3.Ints(
        'unchanged_low old_window new_window old_top new_top row_carry top_carry high modulus u')
    for i in range(N):
        weight=B**i
        oldpacked=low+weight*oldwin+B**(i+N)*oldtop
        newpacked=low+weight*newwin+B**(i+N)*newtop
        prove(f'redc_extended_row_{i}',
              [newwin+R*cend==oldwin+U*M,
               newtop+B**(N-i)*cout==oldtop+cend],
              newpacked+(H+cout)*R*R==oldpacked+H*R*R+U*M*weight)
    # u is a wrapped 64-bit product. The setup theorem supplies m0*inv+1=B*k.
    t0,m0,inv,k,q=z3.Ints('t0 m0 inverse inverse_witness multiply_wrap')
    prove('redc_low_limb_zero',
          [m0*inv+1==B*k,U==t0*inv-B*q],
          t0+U*m0==B*(t0*k-q*m0))
    # Zeroed low limbs remain untouched by every later row.
    zero_prefix=set()
    zero_footprints=[]
    for i in range(N):
        writes=set(range(i,2*N))
        if zero_prefix & writes:raise ValueError('REDC overwrites cancelled prefix')
        zero_prefix.add(i)
        zero_footprints.append({'row':i,'zero_prefix_after':sorted(zero_prefix)})
    if zero_prefix!=set(range(N)):raise ValueError('incomplete cancellation')

    X,T,Q,U,M=z3.Ints('extended_X initial_T coefficient_Q digit_U modulus_M')
    for i in range(N):
        weight=B**i
        prove(f'redc_coefficient_bound_{i}',
              [0<=Q,Q<weight,0<=U,U<B],
              z3.And(0<=Q+U*weight,Q+U*weight<B*weight))
    prove('redc_outer_invariant',[X==T+M*Q],
          X+U*M*W==T+M*(Q+U*W))
    hnext,packed=z3.Ints('high_next packed_words')
    prove('retained_high_does_not_wrap',
          [1<M,M<R,0<=T,T<M*R,0<=Q,Q<R,0<=packed,packed<R*R,
           0<=hnext,packed+hnext*R*R==T+M*Q],
          z3.And(hnext<=1,hnext<B))
    upper,h=z3.Ints('upper_words high_bit')
    z=upper+h*R
    prove('redc_final_quotient',
          [1<M,M<R,0<=T,T<M*R,0<=Q,Q<R,
           0<=upper,upper<R,0<=h,R*upper+h*R*R==T+M*Q],
          z3.And(0<=z,z<2*M,z*R-T==M*Q))

    before=[z3.Int(f'upper_{j}') for j in range(N)]
    modulus=[z3.Int(f'mod_{j}') for j in range(N)]
    diff=[z3.Int(f'diff_{j}') for j in range(N)]
    borrow=[z3.Int(f'borrow_{j}') for j in range(N+1)]
    steps=[before[j]-modulus[j]-borrow[j]==diff[j]-B*borrow[j+1] for j in range(N)]
    prove('borrow_loop_telescopes',[borrow[0]==0,*steps],
          pack(before)-pack(modulus)==pack(diff)-R*borrow[N])
    D,br=z3.Ints('packed_difference final_borrow')
    prove('borrow_identifies_underflow',
          [0<=upper,upper<R,0<=M,M<R,0<=D,D<R,z3.Or(br==0,br==1),upper-M==D-R*br],
          (br==1)==(upper<M))
    # Source mask is common to every output limb: pack after selection equals
    # selection after pack, checked for both possible decisions.
    choose=z3.Bool('choose_difference')
    selected=[z3.If(choose,diff[j],before[j]) for j in range(N)]
    prove('selection_packs',[],pack(selected)==z3.If(choose,pack(diff),pack(before)))
    result=z3.If(z3.Or(h==1,br==0),D,upper)
    assumptions=[1<M,M<R,0<=upper,upper<R,z3.Or(h==0,h==1),0<=D,D<R,
                 z3.Or(br==0,br==1),upper-M==D-R*br,0<=z,z<2*M]
    prove('canonical_result',assumptions,
          z3.And(0<=result,result<M,result==z3.If(z>=M,z-M,z)))
    # Congruence witness after final subtraction avoids variable-modulus SMT.
    choose_sub=z3.If(z>=M,1,0)
    prove('final_montgomery_equation',
          [z*R==T+M*Q,result==z-choose_sub*M],
          result*R-T==M*(Q-choose_sub*R))

    # Concrete legal word assignment detects a dropped row-end carry.
    # x=1, y_last=B-1, old_last=1 yields new_last=0, carry_out=1.
    witness=[x==1,carry[0]==0,*mac]
    witness += [old[j]==(1 if j==N-1 else 0) for j in range(N)]
    witness += [y[j]==(B-1 if j==N-1 else 0) for j in range(N)]
    witness += [new[j]==0 for j in range(N)]
    witness += [carry[j]==(1 if j==N else 0) for j in range(1,N+1)]
    prove('dropped_row_carry_control',witness,
          pack(new)==pack(old)+x*pack(y),expected='sat')
    # A valid reduced representative above m must take the subtraction.
    prove('skipped_final_subtraction_control',assumptions+[M==3,upper==4,h==0,D==1,br==0],
          upper<M,expected='sat')

    report={'source_sha256':digest,'schema_sha256':SCHEMA,'limbs':N,
            'dependencies':deps,'obligations':records,'product_footprints':footprints,
            'redc_zero_footprints':zero_footprints,
            'passed':all(r['result']==r['expected'] for r in records),
            'conclusion':'Packed-state loop composition for the recognized 16-limb algorithm, conditional on the source-to-equation specialization and listed local contracts',
            'trusted':['exact-AST source specialization','composition/footprint checker','pycparser','Z3'],
            'not_claimed':['independently verified C semantics/refinement','compiled-code constant time','hardware timing clearance']}
    (out/'certificate.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dependencies',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result=prove_all(args.output,args.dependencies)
    if not result['passed']:raise SystemExit('loop composition failed or unknown')
    print('PASS: packed-state loop composition under the recorded source mapping')
