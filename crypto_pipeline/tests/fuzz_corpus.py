# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:886498d3-e067-4e0b-a392-0d17aa0ae64d

"""Writes valid seed certificates for tests/fuzz_certificates.c.

Each seed uses the harness's byte format (see the comment at the top of
fuzz_certificates.c) and every value in full-width form (tag 3), so the fuzzer
starts from certificates the checkers accept and mutates them into near-misses.
Usage: python3 tests/fuzz_corpus.py OUTPUT_DIRECTORY
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

SCOPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCOPE_ROOT))
import ecpp_cert as ec  # noqa: E402
import mixed_cert as mc  # noqa: E402
import pocklington_cert as pc  # noqa: E402

POCKLINGTON, ECPP, MIXED = 0, 1, 2
LEAF = 255


def value(v: int, limbs: int) -> bytes:
    assert 0 <= v < 1 << (64 * limbs), v
    return bytes([3]) + v.to_bytes(8 * limbs, 'little')


def small(v: int) -> bytes:
    """A value below 256 in the one-byte form (tag 0)."""
    assert 0 <= v < 256, v
    return bytes([0, v])


def small_pocklington_seed(n: int, factors: list[tuple[int, int, int]]) -> bytes:
    """One node for n < 256, every value in the one-byte form, every factor a
    64-bit leaf. Mutations of these stay among small numbers, where a weakened
    check accepts composites (for example n = 49 with q = 3, a = 18 when the
    F^2 > N condition is missing), and the harness can decide primality."""
    out = bytes([POCKLINGTON, 0, 0, 0, 0]) + small(n) + bytes([len(factors) - 1])
    for q, e, a in factors:
        out += small(q) + small(a) + bytes([e - 1]) + child(-1)
    return out


def child(c: int) -> bytes:
    return bytes([LEAF if c < 0 else c])


def pock_node(node: pc.Node, limbs: int) -> bytes:
    out = value(node.n, limbs) + bytes([len(node.factors) - 1])
    for f in node.factors:
        out += value(f.q, limbs) + value(f.base, limbs) + bytes([f.exponent - 1]) + child(f.child)
    return out


def ecpp_step(s: ec.Step, limbs: int) -> bytes:
    return b''.join(value(getattr(s, f), limbs) for f in ('n', 'a', 'x', 'y', 'k', 'q'))


def limbs_for(*values: int) -> int:
    return max(1, max((v.bit_length() + 63) // 64 for v in values))


def pocklington_seed(nodes: list[pc.Node], root: int) -> bytes:
    limbs = limbs_for(*(n.n for n in nodes))
    out = bytes([POCKLINGTON, limbs - 1, len(nodes) - 1, root, 0])
    return out + b''.join(pock_node(n, limbs) for n in nodes)


def ecpp_seed(steps: list[ec.Step]) -> bytes:
    limbs = limbs_for(*(s.n for s in steps))
    out = bytes([ECPP, limbs - 1, len(steps) - 1, 0, 0])
    return out + b''.join(ecpp_step(s, limbs) for s in steps)


def mixed_seed(cert: mc.MixedCertificate) -> bytes:
    limbs = limbs_for(*(n.n for n in cert.nodes))
    out = bytes([MIXED, limbs - 1, len(cert.nodes) - 1, cert.root, 0, 0, 0])
    for node in cert.nodes:
        if node.kind == mc.POCKLINGTON:
            out += bytes([0]) + child(-1) + pock_node(node.pocklington, limbs)
        else:
            out += bytes([1]) + child(node.q_child) + ecpp_step(node.ecpp, limbs)
    return out


def depth_tree() -> list[pc.Node]:
    """The shared-node tree of tests/test_mixed_depth.c (U, R, S, T)."""
    s, t, r = 4304389, 146349227, 20158208091433697
    u = 36 * s * r + 1
    S = pc.Node(s, [pc.Factor(2, 2, 2, -1), pc.Factor(3, 1, 2, -1), pc.Factor(11, 1, 2, -1),
                    pc.Factor(32609, 1, 2, -1)])
    T = pc.Node(t, [pc.Factor(2, 1, 2, -1), pc.Factor(17, 1, 2, -1), pc.Factor(s, 1, 2, 2)])
    R = pc.Node(r, [pc.Factor(2, 5, 3, -1), pc.Factor(s, 1, 2, 2), pc.Factor(t, 1, 2, 3)])
    U = pc.Node(u, [pc.Factor(2, 2, 2, -1), pc.Factor(3, 2, 2, -1), pc.Factor(s, 1, 2, 2),
                    pc.Factor(r, 1, 2, 1)])
    return [U, R, S, T]


def generated_mixed(bits: int, seed: int) -> mc.MixedCertificate:
    rng = random.Random(seed)
    while True:
        n = rng.getrandbits(bits) | 1 | (1 << (bits - 1))
        if pc.probable_prime(n):
            break
    cert = mc.certify(n, random.Random(seed))
    assert cert is not None
    return cert


def seeds() -> dict[str, bytes]:
    tree = depth_tree()
    out = {
        'pock-shared-tree': pocklington_seed(tree, 0),
        'pock-shared-subtree': pocklington_seed(tree, 1),
        'pock-single': pocklington_seed([tree[2]], 0),
        'ecpp-p127': ecpp_seed(ec.from_pari((SCOPE_ROOT / 'tests/fixtures/ecpp/p127.txt').read_text())),
        'pock-small-7': small_pocklington_seed(7, [(2, 1, 3), (3, 1, 3)]),
        'pock-small-13': small_pocklington_seed(13, [(2, 2, 2), (3, 1, 2)]),
        'pock-small-97': small_pocklington_seed(97, [(2, 5, 5), (3, 1, 5)]),
        'pock-small-251': small_pocklington_seed(251, [(2, 1, 6), (5, 3, 6)]),
        'ecpp-single': ecpp_seed([ec.Step(599146781, 544854974, 40260662, 197163010, 9544, 62773)]),
    }
    for bits, seed in ((128, 4), (160, 1)):
        out[f'mixed-{bits}'] = mixed_seed(generated_mixed(bits, seed))
    return out


def main(argv: list[str]) -> int:
    target = Path(argv[1])
    target.mkdir(parents=True, exist_ok=True)
    for name, data in seeds().items():
        (target / name).write_bytes(data)
    print(f'wrote {len(seeds())} seeds to {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
