#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:268ee6ae-ef9d-4559-923f-2589a2de98d7

"""SMT-checked arithmetic lemmas, not a completed refinement proof of the C.

Each claim is universal over its free integer variables. A claim passes only
when its negation is UNSAT. Counterexamples and UNKNOWN are failures.
"""
import argparse
import json
from pathlib import Path
import z3

B = 1 << 64
R = B ** 16
RESULTS = []


def prove(name, assumptions, claim, out):
    solver = z3.Solver()
    solver.set(timeout=30000)
    solver.add(*assumptions, z3.Not(claim))
    (out / (name + '.smt2')).write_text(solver.to_smt2())
    result = solver.check()
    record = {'name': name, 'result': str(result)}
    if result == z3.sat:
        record['counterexample'] = str(solver.model())
    if result == z3.unknown:
        record['reason'] = solver.reason_unknown()
    RESULTS.append(record)
    print(name + ': ' + str(result), flush=True)


def run(out):
    x, y, t, c = z3.Ints('x y t c')
    word = lambda v: z3.And(0 <= v, v < B)
    total = x*y + t + c
    prove('mac_fits_128', [word(v) for v in (x,y,t,c)],
          z3.And(0 <= total, total < B*B), out)
    s = z3.Int('s')
    prove('split_128', [0 <= s, s < B*B],
          z3.And(s == s % B + B*(s / B), word(s % B), word(s / B)), out)
    b = z3.Int('borrow')
    delta = t-y-b
    borrow = z3.If(t < y+b, 1, 0)
    prove('subtract_with_borrow', [word(t), word(y), z3.Or(b == 0,b == 1)],
          z3.And(delta == delta % B - B*borrow, word(delta % B)), out)

    # Ring identity behind Newton lifting. Divisibility by 2^(2k) follows
    # when m*x-1 is divisible by 2^k; this is a lemma, not a C setup proof.
    m, inv = z3.Ints('m inv')
    prove('newton_error_squares', [],
          m*(inv*(2-m*inv))-1 == -(m*inv-1)*(m*inv-1), out)
    u, k, q = z3.Ints('u k q')
    prove('low_limb_cancels', [u == t*inv-B*q, m*inv == -1+B*k],
          t+u*m == B*(t*k-q*m), out)

    # Inner product-row invariant. P is the already accumulated low part,
    # H the unprocessed high part; W=B^j is a public radix weight.
    # Replacing t by lo and carry by hi must preserve packed value.
    P, H, W, lo, hi = z3.Ints('P H W lo hi')
    prove('mac_packed_step', [x*y+t+c == lo+B*hi],
          P + W*(t+B*H+c+x*y) == (P+W*lo)+B*W*(H+hi), out)
    prove('carry_packed_step', [t+c == lo+B*hi],
          P+W*(t+B*H+c) == (P+W*lo)+B*W*(H+hi), out)

    # Abstract outer REDC state. At iteration i, W=B^i and S=T+m*Q.
    # This identity and coefficient range compose across all 16 iterations.
    S, T, Q = z3.Ints('S T Q')
    prove('redc_invariant_step', [S == T+m*Q],
          S+u*m*W == T+m*(Q+u*W), out)
    prove('redc_coefficient_range', [W >= 1, 0 <= Q, Q < W, word(u)],
          z3.And(0 <= Q+u*W, Q+u*W < B*W), out)
    a, bb = z3.Ints('a b')
    domain = [1 < m, m < R, 0 <= a, a < m, 0 <= bb, bb < m]
    prove('reduced_product_bound', domain, z3.And(0 <= a*bb, a*bb < m*R), out)
    prove('redc_global_bound', [1 < m,m < R,0 <= T,T < m*R,0 <= Q,Q < R],
          z3.And(0 <= T+m*Q,T+m*Q < 2*m*R,T+m*Q < 2*R*R), out)
    z = z3.Int('z')
    prove('redc_quotient', [1 < m,m < R,0 <= T,T < m*R,0 <= Q,Q < R,z*R == T+m*Q],
          z3.And(0 <= z,z < 2*m,z*R-T == m*Q), out)

    low = z % R
    high = z / R
    br = z3.If(low < m, 1, 0)
    select = z3.Or(high == 1, br == 0)
    result = z3.If(select, (low-m) % R, low)
    prove('final_subtraction', [1 < m,m < R,0 <= z,z < 2*m],
          z3.And(z3.Or(high == 0,high == 1),0 <= result,result < m,
                 result == z3.If(z >= m,z-m,z)), out)
    # Real limb mask selection uses two's-complement unsigned wrap.
    bv_borrow, bv_high = z3.BitVecs('bv_borrow bv_high',64)
    old, diff = z3.BitVecs('old diff',64)
    mask = -(bv_high | (bv_borrow ^ 1))
    prove('limb_mask_selection', [z3.ULE(bv_borrow,1),z3.ULE(bv_high,1)],
          ((diff & mask) | (old & ~mask)) == z3.If(z3.Or(bv_high == 1,bv_borrow == 0),diff,old), out)

    # Deliberately false claim: dropping the retained top carry is unsound.
    bad = z3.If(br == 0,(low-m) % R,low)
    solver = z3.Solver()
    solver.add(1 < m,m < R,0 <= z,z < 2*m,bad != z3.If(z >= m,z-m,z))
    (out/'negative_control.smt2').write_text(solver.to_smt2())
    status = solver.check()
    RESULTS.append({'name':'dropped_high_carry_control','result':str(status),
                    'expected':'sat','counterexample':str(solver.model()) if status == z3.sat else None})
    print('dropped_high_carry_control: ' + str(status), flush=True)
    return all(r['result'] == r.get('expected','unsat') for r in RESULTS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    passed = run(args.output)
    (args.output/'arithmetic.json').write_text(json.dumps({'z3':z3.get_version_string(),
        'scope':'Arithmetic lemmas only; C refinement/composition remains open',
        'results':RESULTS},indent=2)+'\n')
    raise SystemExit(0 if passed else 1)
