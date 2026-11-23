# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:7de9abeb-118d-4fcb-bff6-5981f5b99287

"""Tests for the Pocklington certificate checker (crypto_cert.h).

Valid certificates come from pocklington_cert.py and are re-checked here in
Python before use. Invalid ones are made by changing one field at a time, so
each test shows the checker rejects exactly the defect it targets.
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
import pocklington_cert as pc  # noqa: E402

U = c.c_uint64
P = c.POINTER(U)
MASK = (1 << 64) - 1
LIB = c.CDLL(str(SCOPE_ROOT / 'libcryptomult.so'), use_errno=True)


class Factor(c.Structure):
    _fields_ = [('prime', P), ('base', P), ('exponent', c.c_uint32), ('child', c.c_int32)]


class Node(c.Structure):
    _fields_ = [('n', P), ('limbs', c.c_size_t), ('factors', c.POINTER(Factor)),
                ('factor_count', c.c_size_t)]


LIB.crypto_pocklington_verify.argtypes = [c.POINTER(Node), c.c_size_t, c.c_size_t, U,
                                          c.POINTER(c.c_size_t)]
LIB.crypto_pocklington_verify.restype = c.c_int
LIB.crypto_is_prime_u64.argtypes = [U]
LIB.crypto_is_prime_u64.restype = c.c_int


def limbs(value, n):
    return (U * n)(*[(value >> (64 * i)) & MASK for i in range(n)])


def width(value):
    return max(1, (value.bit_length() + 63) // 64)


def to_ctypes(cert, pad=0):
    """Builds ctypes nodes; returns (array, keepalive). pad adds zero limbs."""
    keep = []
    nodes = (Node * len(cert.nodes))()
    for i, node in enumerate(cert.nodes):
        n = width(node.n) + pad
        facs = (Factor * max(1, len(node.factors)))()
        for k, f in enumerate(node.factors):
            q, a = limbs(f.q, n), limbs(f.base, n)
            keep += [q, a]
            facs[k] = Factor(q, a, f.exponent, f.child)
        nv = limbs(node.n, n)
        keep += [nv, facs]
        nodes[i] = Node(nv, n, facs, len(node.factors))
    return nodes, keep


def verify(cert, max_modexps=0, pad=0, node_count=None, root=None):
    nodes, keep = to_ctypes(cert, pad)
    failed = c.c_size_t(9999)
    count = len(cert.nodes) if node_count is None else node_count
    rc = LIB.crypto_pocklington_verify(nodes, count, cert.root if root is None else root,
                                       max_modexps, c.byref(failed))
    del keep
    return rc, c.get_errno(), failed.value


def python_check(cert):
    """Independent re-check of every Pocklington condition."""
    for node in cert.nodes:
        n = node.n
        f = math.prod(x.q ** x.exponent for x in node.factors)
        assert n % 2 == 1 and n > 2 and (n - 1) % f == 0 and f * f > n
        for x in node.factors:
            assert pow(x.base, n - 1, n) == 1
            assert math.gcd(pow(x.base, (n - 1) // x.q, n) - 1, n) == 1
            if x.child == -1:
                assert x.q < 2 ** 64 and pc.probable_prime(x.q)
            else:
                assert cert.nodes[x.child].n == x.q


MERSENNE_127_PM1 = {2: 1, 3: 3, 7: 2, 19: 1, 43: 1, 73: 1, 127: 1, 337: 1, 5419: 1,
                    92737: 1, 649657: 1, 77158673929: 1}


class LeafTests(unittest.TestCase):
    def test_u64_against_trial_division(self):
        for v in range(0, 5000):
            expected = v >= 2 and all(v % f for f in range(2, math.isqrt(v) + 1))
            self.assertEqual(LIB.crypto_is_prime_u64(v), int(expected), v)

    def test_generator_filter_against_trial_division(self):
        # Regression: primes 41..71 are also extra bases and must not fail as a = 0.
        for v in range(0, 5000):
            expected = v >= 2 and all(v % f for f in range(2, math.isqrt(v) + 1))
            self.assertEqual(pc.probable_prime(v), expected, v)

    def test_u64_pseudoprimes_and_primes(self):
        for v in (2047, 1373653, 25326001, 3215031751, 3825123056546413051, 2 ** 64 - 1):
            self.assertEqual(LIB.crypto_is_prime_u64(v), 0, v)
        for v in (2 ** 61 - 1, 2 ** 64 - 59, 18446744073709551533, 4294967291):
            self.assertEqual(LIB.crypto_is_prime_u64(v), 1, v)


class ValidCertificateTests(unittest.TestCase):
    def test_proth_and_fermat_primes(self):
        for n, fac in ((65537, {2: 16}), (998244353, {2: 23}), (1004535809, {2: 21}),
                       (3 * 2 ** 189 + 1, {2: 189})):
            cert = pc.certify(n, fac)
            python_check(cert)
            self.assertEqual(verify(cert)[:2], (0, 0), n)

    def test_mersenne_127_full_factorization(self):
        n = 2 ** 127 - 1
        self.assertEqual(math.prod(q ** e for q, e in MERSENNE_127_PM1.items()), n - 1)
        cert = pc.certify(n, MERSENNE_127_PM1)
        python_check(cert)
        self.assertEqual(verify(cert)[:2], (0, 0))

    def test_random_provable_primes_with_nested_certificates(self):
        rng = random.Random(20261122)
        for bits in (64, 100, 200, 512, 1024, 2048):
            n, cert = pc.provable_prime(bits, rng)
            self.assertEqual(n.bit_length(), bits)
            python_check(cert)
            self.assertEqual(verify(cert)[:2], (0, 0), bits)
        self.assertGreaterEqual(len(cert.nodes), 5)  # 2048 bits nests several levels

    def test_zero_padded_limbs(self):
        n, cert = pc.provable_prime(200, random.Random(3))
        self.assertEqual(verify(cert, pad=2)[:2], (0, 0))

    def test_shared_child_is_checked_once(self):
        # Two middle nodes q1 and q2 both rely on the same grandchild q, and the
        # root relies on q1 and q2: a DAG, where q is verified only once.
        rng = random.Random(4)
        q, q_cert = pc.provable_prime(130, rng)

        def above(base_q, lo_bits):
            while True:
                r = rng.randrange(1 << lo_bits, 1 << (lo_bits + 1))
                n = 2 * r * base_q + 1
                if pc.probable_prime(n):
                    return n

        q1 = above(q, 120)
        q2 = above(q, 121)
        cert = pc.Certificate()
        g = pc.graft(cert, q_cert)
        mids = []
        for m in (q1, q2):
            mids.append(cert.add(pc.Node(m, [pc.Factor(2, 1, pc.find_base(m, 2), -1),
                                             pc.Factor(q, 1, pc.find_base(m, q), g)])))
        n = above(q1 * q2, 40)
        cert.root = cert.add(pc.Node(n, [pc.Factor(2, 1, pc.find_base(n, 2), -1),
                                         pc.Factor(q1, 1, pc.find_base(n, q1), mids[0]),
                                         pc.Factor(q2, 1, pc.find_base(n, q2), mids[1])]))
        python_check(cert)
        needed = 2 * sum(len(nd.factors) for nd in cert.nodes)  # each node once
        self.assertEqual(verify(cert, max_modexps=needed)[:2], (0, 0))
        self.assertEqual(verify(cert, max_modexps=needed - 1)[:2], (-1, errno.ERANGE))


class RejectionTests(unittest.TestCase):
    def setUp(self):
        self.n, self.cert = pc.provable_prime(256, random.Random(11))
        python_check(self.cert)
        self.root = self.cert.nodes[self.cert.root]

    def assert_rejected(self, cert, code, **kwargs):
        rc, err, failed = verify(cert, **kwargs)
        self.assertEqual((rc, err), (-1, code))
        return failed

    def test_bad_fermat_base(self):
        cert = copy.deepcopy(self.cert)
        f = cert.nodes[cert.root].factors[0]
        f.base = next(a for a in range(2, 1000) if pow(a, self.n - 1, self.n) == 1 and
                      math.gcd(pow(a, (self.n - 1) // f.q, self.n) - 1, self.n) != 1)
        self.assertEqual(self.assert_rejected(cert, errno.EDOM), cert.root)

    def test_factored_part_too_small(self):
        cert = copy.deepcopy(self.cert)
        root = cert.nodes[cert.root]
        root.factors = [f for f in root.factors if f.q == 2]  # F = 2 < sqrt(N)
        self.assert_rejected(cert, errno.EDOM)

    def test_factor_not_dividing_n_minus_1(self):
        cert = copy.deepcopy(self.cert)
        two = next(f for f in cert.nodes[cert.root].factors if f.q == 2)
        while (self.n - 1) % (2 ** (two.exponent + 1)) == 0:
            two.exponent += 1
        two.exponent += 1  # one more power of 2 than divides N - 1
        self.assert_rejected(cert, errno.EDOM)

    def test_composite_leaf(self):
        # In both cases the composite leaf divides N - 1, F^2 > N and its base
        # passes the Pocklington conditions, so leaf primality is the only defect.
        cert = pc.certify(7681, {2: 9})
        cert.nodes[0].factors.append(pc.Factor(15, 1, 2, -1))  # 15 = 3 * 5; 7680 = 2^9 * 15
        self.assert_rejected(cert, errno.EDOM)
        cert = pc.certify(998244353, {2: 23})
        cert.nodes[0].factors.append(pc.Factor(119, 1, 3, -1))  # 119 = 7 * 17 divides N - 1
        self.assert_rejected(cert, errno.EDOM)

    def test_large_leaf_without_child_is_malformed(self):
        cert = copy.deepcopy(self.cert)
        f = next(f for f in cert.nodes[cert.root].factors if f.child >= 0)
        f.child = -1
        self.assert_rejected(cert, errno.EINVAL)

    def test_child_proving_a_different_number(self):
        cert = copy.deepcopy(self.cert)
        f = next(f for f in cert.nodes[cert.root].factors if f.child >= 0)
        cert.nodes[f.child].n += 2
        self.assert_rejected(cert, errno.EINVAL)

    def test_cycle_and_bad_index(self):
        cert = copy.deepcopy(self.cert)
        f = next(f for f in cert.nodes[cert.root].factors if f.child >= 0)
        f.child = cert.root  # self-reference: the root cannot prove its own factor
        self.assertEqual(self.assert_rejected(cert, errno.EINVAL), cert.root)
        cert = copy.deepcopy(self.cert)
        f = next(f for f in cert.nodes[cert.root].factors if f.child >= 0)
        f.child = len(cert.nodes) + 5
        self.assert_rejected(cert, errno.EINVAL)

    def test_forged_certificates_for_composites(self):
        # A Carmichael number and a strong pseudoprime cannot be certified even
        # with factors and bases chosen to pass as many checks as possible.
        for n in (561, 41041, 3215031751, 3825123056546413051):
            m = n - 1
            fac, x = {}, m
            for p in range(2, 10 ** 6):
                while x % p == 0:
                    fac[p] = fac.get(p, 0) + 1
                    x //= p
            if x > 1:
                fac[x] = fac.get(x, 0) + 1
            node = pc.Node(n, [pc.Factor(q, e, 2, -1) for q, e in fac.items()])
            for a in range(2, 200):
                for f in node.factors:
                    if math.gcd(pow(a, m // f.q, n) - 1, n) == 1 and pow(a, m, n) == 1:
                        f.base = a
            cert = pc.Certificate([node], 0)
            rc, err, _ = verify(cert)
            self.assertEqual((rc, err), (-1, errno.EDOM), n)

    def test_even_and_tiny_n(self):
        for n in (4, 2, 1):
            cert = pc.Certificate([pc.Node(n, [pc.Factor(2, 1, 3, -1)])], 0)
            self.assert_rejected(cert, errno.EDOM)

    def test_base_out_of_range(self):
        cert = copy.deepcopy(self.cert)
        cert.nodes[cert.root].factors[0].base = self.n - 1
        self.assert_rejected(cert, errno.EDOM)

    def test_work_cap(self):
        needed = 2 * sum(len(nd.factors) for nd in self.cert.nodes)
        self.assertEqual(verify(self.cert, max_modexps=needed)[:2], (0, 0))
        self.assertEqual(verify(self.cert, max_modexps=needed - 1)[:2], (-1, errno.ERANGE))

    def test_structural_arguments(self):
        nodes, keep = to_ctypes(self.cert)
        self.assertEqual(LIB.crypto_pocklington_verify(None, 1, 0, 0, None), -1)
        self.assertEqual(c.get_errno(), errno.EINVAL)
        self.assertEqual(LIB.crypto_pocklington_verify(nodes, 0, 0, 0, None), -1)
        self.assertEqual(c.get_errno(), errno.EINVAL)
        self.assertEqual(LIB.crypto_pocklington_verify(nodes, len(self.cert.nodes),
                                                       len(self.cert.nodes), 0, None), -1)
        self.assertEqual(c.get_errno(), errno.EINVAL)
        del keep


if __name__ == '__main__':
    unittest.main()
