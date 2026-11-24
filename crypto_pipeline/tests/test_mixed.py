# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:c83a2327-e0ed-461e-877f-ecf0db6281b9

"""Tests for the mixed Pocklington/ECPP checker (crypto_mixed.h).

Valid certificates come from mixed_cert.py (seeds chosen so that both link
directions occur) and are re-checked here with plain integer arithmetic,
independent of the C code. Invalid ones change one field at a time, and each
test shows the checker rejects exactly the defect it targets, at the node that
holds it.
"""
import copy
import ctypes as c
import errno
import math
import random
import sys
import unittest
from pathlib import Path

SCOPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCOPE_ROOT))
import ecpp_cert as ec  # noqa: E402
import mixed_cert as mc  # noqa: E402
import pocklington_cert as pc  # noqa: E402

U = c.c_uint64
P = c.POINTER(U)
MASK = (1 << 64) - 1
ECPP_FIELDS = ('n', 'a', 'x', 'y', 'k', 'q')
LIB = c.CDLL(str(SCOPE_ROOT / 'libcryptomult.so'), use_errno=True)
FIXTURES = SCOPE_ROOT / 'tests' / 'fixtures' / 'ecpp'
# (bits, seed) whose certificates contain both Pocklington -> ECPP and ECPP -> Pocklington links.
GENERATED = ((128, 4), (160, 1), (200, 6), (256, 4))


class CFactor(c.Structure):
    _fields_ = [('prime', P), ('base', P), ('exponent', c.c_uint32), ('child', c.c_int32)]


class CNode(c.Structure):
    _fields_ = [('n', P), ('limbs', c.c_size_t), ('factors', c.POINTER(CFactor)),
                ('factor_count', c.c_size_t)]


class CStep(c.Structure):
    _fields_ = [(f, P) for f in ECPP_FIELDS] + [('limbs', c.c_size_t)]


class CMixed(c.Structure):
    _fields_ = [('kind', c.c_uint32), ('q_child', c.c_int32),
                ('pocklington', c.POINTER(CNode)), ('ecpp', c.POINTER(CStep))]


LIB.crypto_mixed_verify.argtypes = [c.POINTER(CMixed), c.c_size_t, c.c_size_t, U, U,
                                    c.POINTER(c.c_size_t)]
LIB.crypto_mixed_verify.restype = c.c_int
KIND = {mc.POCKLINGTON: 1, mc.ECPP: 2}


def limbs(value, n):
    return (U * n)(*[(value >> (64 * i)) & MASK for i in range(n)])


def width(value):
    return max(1, (value.bit_length() + 63) // 64)


def verify(cert, max_modexps=0, max_point_ops=0, pad=0, root=None, count=None, kinds=None):
    keep, arr = [], (CMixed * len(cert.nodes))()
    for i, node in enumerate(cert.nodes):
        w = width(node.n) + pad
        kind = (kinds or {}).get(i, KIND[node.kind])
        if node.kind == mc.POCKLINGTON:
            pn = node.pocklington
            facs = (CFactor * len(pn.factors))()
            for j, f in enumerate(pn.factors):
                q, a = limbs(f.q, w), limbs(f.base, w)
                keep += [q, a]
                facs[j] = CFactor(q, a, f.exponent, f.child)
            n_arr = limbs(pn.n, w)
            cnode = CNode(n_arr, w, facs, len(pn.factors))
            keep += [facs, n_arr, cnode]
            arr[i] = CMixed(kind, -1, c.pointer(cnode), None)
        else:
            vals = [limbs(getattr(node.ecpp, f), w) for f in ECPP_FIELDS]
            step = CStep(*vals, w)
            keep += vals + [step]
            arr[i] = CMixed(kind, node.q_child, None, c.pointer(step))
    failed = c.c_size_t(9999)
    rc = LIB.crypto_mixed_verify(arr, len(cert.nodes) if count is None else count,
                                 cert.root if root is None else root, max_modexps, max_point_ops,
                                 c.byref(failed))
    del keep
    return rc, c.get_errno(), failed.value


def reachable(cert):
    seen, todo = set(), [cert.root]
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        node = cert.nodes[i]
        if node.kind == mc.POCKLINGTON:
            todo += [f.child for f in node.pocklington.factors if f.child >= 0]
        elif node.q_child >= 0:
            todo.append(node.q_child)
    return seen


def costs(cert):
    """(modexps, point additions) the checker needs, counting each node once."""
    modexps = point_ops = 0
    for i in reachable(cert):
        node = cert.nodes[i]
        if node.kind == mc.POCKLINGTON:
            modexps += 2 * len(node.pocklington.factors)
        else:
            point_ops += 2 * (node.ecpp.k.bit_length() + node.ecpp.q.bit_length())
    return modexps, point_ops


def python_check(cert):
    """Independent re-check of every condition, with plain integers."""
    def proven(value, child):
        if child == -1:
            assert value < 2 ** 64 and pc.probable_prime(value)  # exact below 2^64
        else:
            assert cert.nodes[child].n == value

    for i in reachable(cert):
        node = cert.nodes[i]
        if node.kind == mc.POCKLINGTON:
            pn = node.pocklington
            f = math.prod(x.q ** x.exponent for x in pn.factors)
            assert pn.n % 2 and (pn.n - 1) % f == 0 and f * f > pn.n
            for x in pn.factors:
                assert 2 <= x.q < pn.n and 2 <= x.base < pn.n - 1
                assert pow(x.base, pn.n - 1, pn.n) == 1
                assert math.gcd(pow(x.base, (pn.n - 1) // x.q, pn.n) - 1, pn.n) == 1
                proven(x.q, x.child)
        else:
            s, n = node.ecpp, node.n
            assert n % 2 and n % 3 and ec.q_bound(n) <= s.q < n and 1 <= s.k < n
            assert (4 * s.a ** 3 + 27 * s.b ** 2) % n
            q_point = ec.affine_mul(s.k, (s.x, s.y), s.a, n)
            assert q_point is not None and ec.affine_mul(s.q, q_point, s.a, n) is None
            proven(s.q, node.q_child)


def generated(bits, seed):
    rng = random.Random(seed)
    while True:
        n = rng.getrandbits(bits) | 1 | (1 << (bits - 1))
        if pc.probable_prime(n):
            break
    cert = mc.certify(n, random.Random(seed))
    assert cert is not None, (bits, seed)
    return n, cert


def first_node(cert, kind, predicate=lambda node: True):
    return next(i for i in sorted(reachable(cert)) if cert.nodes[i].kind == kind
                and predicate(cert.nodes[i]))


class ValidCertificateTests(unittest.TestCase):
    def test_generated_mixed_certificates_prove_their_primes(self):
        for bits, seed in GENERATED:
            with self.subTest(bits=bits, seed=seed):
                n, cert = generated(bits, seed)
                self.assertEqual(cert.nodes[cert.root].n, n)
                self.assertLessEqual({(mc.POCKLINGTON, mc.ECPP), (mc.ECPP, mc.POCKLINGTON)}, cert.edges())
                python_check(cert)
                self.assertEqual(verify(cert)[:2], (0, 0))
                self.assertEqual(verify(cert, pad=1)[:2], (0, 0))  # zero-padded limbs

    def test_single_method_certificates_are_mixed_certificates(self):
        steps = ec.from_pari((FIXTURES / 'p127.txt').read_text())
        chain = mc.MixedCertificate()
        for i, s in enumerate(steps):
            chain.add(mc.MixedNode(mc.ECPP, ecpp=s, q_child=i + 1 if i + 1 < len(steps) else -1))
        python_check(chain)
        self.assertEqual(verify(chain)[:2], (0, 0))
        _, tree = pc.provable_prime(512, random.Random(3))
        plain = mc.MixedCertificate(root=tree.root)
        for node in tree.nodes:
            plain.add(mc.MixedNode(mc.POCKLINGTON, pocklington=node))
        python_check(plain)
        self.assertEqual(verify(plain)[:2], (0, 0))

    def test_work_caps_at_their_exact_boundary(self):
        _, cert = generated(*GENERATED[2])
        modexps, point_ops = costs(cert)
        self.assertGreater(modexps, 0)
        self.assertGreater(point_ops, 0)
        self.assertEqual(verify(cert, modexps, point_ops)[:2], (0, 0))
        self.assertEqual(verify(cert, modexps - 1, 0)[:2], (-1, errno.ERANGE))
        self.assertEqual(verify(cert, 0, point_ops - 1)[:2], (-1, errno.ERANGE))


class RejectionTests(unittest.TestCase):
    def setUp(self):
        self.n, self.cert = generated(*GENERATED[3])

    def assert_rejected(self, cert, code, node):
        self.assertEqual(verify(cert), (-1, code, node))

    def test_wrong_ecpp_cofactor_is_rejected_at_its_node(self):
        cert = copy.deepcopy(self.cert)
        i = first_node(cert, mc.ECPP)
        cert.nodes[i].ecpp.k += 1
        self.assert_rejected(cert, errno.EDOM, i)

    def test_wrong_pocklington_exponent_is_rejected_at_its_node(self):
        cert = copy.deepcopy(self.cert)
        i = first_node(cert, mc.POCKLINGTON)
        cert.nodes[i].pocklington.factors[0].exponent += 1   # F no longer divides N - 1
        self.assert_rejected(cert, errno.EDOM, i)

    def test_link_to_a_node_proving_a_different_number(self):
        cert = copy.deepcopy(self.cert)
        i = first_node(cert, mc.ECPP, lambda node: node.q_child >= 0)
        cert.nodes[i].q_child = cert.root                     # the root proves N, not this q
        self.assert_rejected(cert, errno.EINVAL, i)
        cert = copy.deepcopy(self.cert)
        i = first_node(cert, mc.POCKLINGTON, lambda node: any(f.child >= 0 for f in node.pocklington.factors))
        f = next(f for f in cert.nodes[i].pocklington.factors if f.child >= 0)
        f.child = len(cert.nodes)                             # out of range
        self.assert_rejected(cert, errno.EINVAL, i)

    def test_large_q_cannot_be_a_64_bit_leaf(self):
        cert = copy.deepcopy(self.cert)
        i = first_node(cert, mc.ECPP, lambda node: node.ecpp.q >= 2 ** 64)
        cert.nodes[i].q_child = -1
        self.assert_rejected(cert, errno.EINVAL, i)

    def test_unknown_kind_and_structural_arguments(self):
        self.assertEqual(verify(self.cert, kinds={self.cert.root: 9}), (-1, errno.EINVAL, self.cert.root))
        self.assertEqual(verify(self.cert, root=len(self.cert.nodes))[:2], (-1, errno.EINVAL))
        self.assertEqual(verify(self.cert, count=0)[:2], (-1, errno.EINVAL))
        self.assertEqual(LIB.crypto_mixed_verify(None, 1, 0, 0, 0, None), -1)
        self.assertEqual(c.get_errno(), errno.EINVAL)

    def test_forged_root_for_a_composite_is_rejected(self):
        # Claim the root proves N * 3: every condition tied to N fails.
        cert = copy.deepcopy(self.cert)
        root = cert.nodes[cert.root]
        if root.kind == mc.POCKLINGTON:
            root.pocklington.n *= 3
        else:
            root.ecpp.n *= 3
        self.assertEqual(verify(cert)[:2], (-1, errno.EDOM))


if __name__ == '__main__':
    unittest.main()
