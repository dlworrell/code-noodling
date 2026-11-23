# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:003b0abf-5ea0-49eb-851b-5694473a48a1

"""Tests for the ECPP checker (crypto_ecpp.h) and its shared helpers.

Valid chains come from ecpp_cert.py and are re-checked here with plain affine
arithmetic, independent of the checker's projective formula. Invalid ones
change one field at a time. The addition law itself is checked exhaustively
over small prime fields: it must give the true sum or (0:0:0), never a wrong
point.
"""
import copy
import ctypes as c
import errno
import math
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCOPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCOPE_ROOT))
import ecpp_cert as ec  # noqa: E402

U = c.c_uint64
P = c.POINTER(U)
MASK = (1 << 64) - 1
FIELDS = ('n', 'a', 'x', 'y', 'k', 'q')
LIB = c.CDLL(str(SCOPE_ROOT / 'libcryptomult.so'), use_errno=True)
FIXTURES = SCOPE_ROOT / 'tests' / 'fixtures' / 'ecpp'


class Step(c.Structure):
    _fields_ = [(f, P) for f in FIELDS] + [('limbs', c.c_size_t)]


LIB.crypto_ecpp_verify.argtypes = [c.POINTER(Step), c.c_size_t, U, c.POINTER(c.c_size_t)]
LIB.crypto_ecpp_verify.restype = c.c_int
LIB.crypto_ecpp_scalar_mul.argtypes = [P] * 9 + [c.c_size_t]
LIB.crypto_ecpp_scalar_mul.restype = c.c_int
LIB.crypto_bn_isqrt.argtypes = [P, P, c.c_size_t]
LIB.crypto_bn_isqrt.restype = None
LIB.crypto_bn_mod3.argtypes = [P, c.c_size_t]
LIB.crypto_bn_mod3.restype = c.c_uint


def limbs(value, n):
    return (U * n)(*[(value >> (64 * i)) & MASK for i in range(n)])


def to_int(arr, n):
    return sum(arr[i] << (64 * i) for i in range(n))


def width(value):
    return max(1, (value.bit_length() + 63) // 64)


def verify(steps, max_ops=0, pad=0):
    keep, arr = [], (Step * len(steps))()
    for i, s in enumerate(steps):
        w = width(s.n) + pad
        vals = [limbs(getattr(s, f), w) for f in FIELDS]
        keep += vals
        arr[i] = Step(*vals, w)
    failed = c.c_size_t(9999)
    rc = LIB.crypto_ecpp_verify(arr, len(steps), max_ops, c.byref(failed))
    del keep
    return rc, c.get_errno(), failed.value


def c_scalar_mul(k, x, y, a, b, m):
    n = max(width(m), width(k))
    out = [limbs(0, n) for _ in range(3)]
    rc = LIB.crypto_ecpp_scalar_mul(*out, *(limbs(v, n) for v in (a, b, x, y, k, m)), n)
    assert rc == 0
    return tuple(to_int(o, n) for o in out)


def python_check(steps):
    """Independent affine re-check of every condition (N prime here)."""
    for i, s in enumerate(steps):
        n = s.n
        assert n % 2 and n % 3 and ec.q_bound(n) <= s.q < n and 1 <= s.k < n
        assert (4 * s.a ** 3 + 27 * s.b ** 2) % n
        q_point = ec.affine_mul(s.k, (s.x, s.y), s.a, n)
        assert q_point is not None and ec.affine_mul(s.q, q_point, s.a, n) is None
        assert steps[i + 1].n == s.q if i + 1 < len(steps) else s.q < 2 ** 64


def ops_needed(steps):
    return sum(2 * (s.k.bit_length() + s.q.bit_length()) for s in steps)


def collapse_mod(k, point, a, b, p):
    """k * point mod p with the checker's law; point is projective."""
    r, b3 = (0, 1 % p, 0), 3 * b % p
    for bit in bin(k)[2:]:
        r = ec.rcb_add(r, r, a, b3, p)
        if bit == '1':
            r = ec.rcb_add(r, point, a, b3, p)
    return r


class AdditionLawTests(unittest.TestCase):
    def test_true_sum_or_zero_never_wrong(self):
        zeros = 0
        for p in (5, 7, 11, 13):
            for a in range(p):
                for b in range(p):
                    if (4 * a ** 3 + 27 * b * b) % p == 0:
                        continue
                    pts = [None] + [(x, y) for x in range(p) for y in range(p)
                                    if (y * y - x ** 3 - a * x - b) % p == 0]
                    proj = [(0, 1, 0) if q is None else (q[0], q[1], 1) for q in pts]
                    for i, p1 in enumerate(pts):
                        for j, p2 in enumerate(pts):
                            X, Y, Z = ec.rcb_add(proj[i], proj[j], a, 3 * b % p, p)
                            want = ec.affine_add(p1, p2, a, p)
                            if (X, Y, Z) == (0, 0, 0):
                                zeros += 1
                                diff = ec.affine_add(p1, None if p2 is None else
                                                     (p2[0], (-p2[1]) % p), a, p)
                                self.assertTrue(diff is not None and diff[1] == 0)
                            elif want is None:
                                self.assertEqual((X % p, Z % p), (0, 0))
                                self.assertNotEqual(Y % p, 0)
                            else:
                                zi = pow(Z, -1, p)
                                self.assertEqual((X * zi % p, Y * zi % p), want)
        self.assertGreater(zeros, 0)  # collapses do occur, on curves with 2-torsion

    def test_zero_is_absorbing(self):
        rng = random.Random(1)
        for _ in range(500):
            n = rng.choice([1009, 2 ** 61 - 1, rng.getrandbits(130) | 1])
            pt = tuple(rng.randrange(n) for _ in range(3))
            a, b3 = rng.randrange(n), rng.randrange(n)
            self.assertEqual(ec.rcb_add((0, 0, 0), pt, a, b3, n), (0, 0, 0))
            self.assertEqual(ec.rcb_add(pt, (0, 0, 0), a, b3, n), (0, 0, 0))

    def test_c_scalar_mul_matches_reference(self):
        rng = random.Random(2)
        for p in (7, 11, 13):  # exhaustive points and small scalars, composite-free
            for a in range(p):
                for b in range(p):
                    for x in range(p):
                        for y in range(p):
                            if (y * y - x ** 3 - a * x - b) % p:
                                continue
                            for k in (0, 1, 2, 3, 5, 12, 13):
                                self.assertEqual(c_scalar_mul(k, x, y, a, b, p),
                                                 ec.projective_mul(k, x, y, a, b, p))
        for _ in range(60):  # large odd moduli, prime or not, multi-limb scalars
            m = rng.getrandbits(rng.choice([64, 130, 300])) | 1
            vals = [rng.randrange(m) for _ in range(4)]
            k = rng.getrandbits(rng.choice([1, 64, 200]))
            k %= m
            self.assertEqual(c_scalar_mul(k, vals[2], vals[3], vals[0], vals[1], m),
                             ec.projective_mul(k, vals[2], vals[3], vals[0], vals[1], m))


class HelperTests(unittest.TestCase):
    def test_isqrt_and_mod3(self):
        rng = random.Random(3)
        for v in list(range(300)) + [rng.getrandbits(rng.randrange(1, 900)) for _ in range(300)]:
            n = width(v)
            out = limbs(0, n)
            LIB.crypto_bn_isqrt(out, limbs(v, n), n)
            self.assertEqual(to_int(out, n), math.isqrt(v), v)
            self.assertEqual(LIB.crypto_bn_mod3(limbs(v, n), n), v % 3)
        for r in (2 ** 32 - 1, 2 ** 64 - 1, 2 ** 100 + 7):  # squares and their neighbours
            for v in (r * r - 1, r * r, r * r + 1):
                n = width(v)
                out = limbs(0, n)
                LIB.crypto_bn_isqrt(out, limbs(v, n), n)
                self.assertEqual(to_int(out, n), math.isqrt(v))


class ValidChainTests(unittest.TestCase):
    def test_random_provable_primes(self):
        for bits in (80, 128, 192, 256, 384):
            n, steps = ec.provable_prime(bits, random.Random(bits))
            self.assertEqual(n.bit_length(), bits)
            python_check(steps)
            self.assertEqual(verify(steps)[:2], (0, 0), bits)
        self.assertGreaterEqual(len(steps), 8)

    def test_zero_padded_limbs_and_exact_work(self):
        n, steps = ec.provable_prime(200, random.Random(9))
        self.assertEqual(verify(steps, pad=2)[:2], (0, 0))
        need = ops_needed(steps)
        self.assertEqual(verify(steps, max_ops=need)[:2], (0, 0))
        self.assertEqual(verify(steps, max_ops=need - 1)[:2], (-1, errno.ERANGE))

    def test_generator_math(self):
        for n in (2 ** 61 - 1, 1000003, 10 ** 9 + 7):
            for v in range(1, 50):
                r = ec.sqrt_mod(v, n)
                if r is not None:
                    self.assertEqual(r * r % n, v)
        # Cornacchia finds a solution of t^2 + |D| v^2 = 4p exactly when one exists.
        primes = [v for v in range(5, 600) if ec.probable_prime(v)]
        for d, _ in ec.CM_DISCRIMINANTS:
            for p in primes:
                exists = any(math.isqrt((4 * p - t * t) // -d) ** 2 * -d == 4 * p - t * t
                             for t in range(0, math.isqrt(4 * p) + 1) if (4 * p - t * t) % -d == 0)
                got = ec.cornacchia4(d, p)
                self.assertEqual(got is not None, exists, (d, p))
                if got:
                    self.assertEqual(got[0] ** 2 - d * got[1] ** 2, 4 * p)


class RejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n, cls.steps = ec.provable_prime(192, random.Random(11))
        python_check(cls.steps)

    def mutate(self, index=0, **fields):
        steps = copy.deepcopy(self.steps)
        for name, value in fields.items():
            setattr(steps[index], name, value)
        return steps

    def assert_rejected(self, steps, code, at=0):
        self.assertEqual(verify(steps), (-1, code, at))

    def test_one_field_defects(self):
        s = self.steps[0]
        self.assert_rejected(self.mutate(k=s.k + 1), errno.EDOM)
        self.assert_rejected(self.mutate(x=(s.x + 1) % s.n), errno.EDOM)
        self.assert_rejected(self.mutate(a=(s.a + 1) % s.n), errno.EDOM)
        self.assert_rejected(self.mutate(x=s.n), errno.EDOM)
        self.assert_rejected(self.mutate(k=0), errno.EDOM)
        self.assert_rejected(self.mutate(a=0, x=1, y=1), errno.EDOM)  # y^2 = x^3: singular

    def test_identity_q(self):
        # k = full group order makes Q the identity.
        s = self.steps[0]
        self.assert_rejected(self.mutate(k=s.k * s.q), errno.EDOM)

    def test_q_below_the_bound_with_valid_arithmetic(self):
        # A genuine point of small prime order: every curve check passes, only
        # the Hasse-bound condition fails.
        s = self.steps[0]
        m = s.k * s.q
        small = next(f for f in range(3, 10 ** 6, 2) if m % f == 0 and ec.probable_prime(f))
        step = ec.Step(s.n, s.a, s.x, s.y, m // small, small)
        qp = ec.affine_mul(step.k, (s.x, s.y), s.a, s.n)
        self.assertIsNotNone(qp)
        self.assertIsNone(ec.affine_mul(small, qp, s.a, s.n))
        self.assertLess(small, ec.q_bound(s.n))
        self.assert_rejected([step], errno.EDOM)

    def test_n_divisible_by_three_or_even(self):
        s = self.steps[0]
        self.assert_rejected(self.mutate(n=3 * s.q + 3 * 2 ** 70), errno.EDOM)
        self.assert_rejected(self.mutate(n=s.n + 1), errno.EDOM)

    def test_chain_structure(self):
        steps = copy.deepcopy(self.steps)
        steps[1].n += 2
        self.assert_rejected(steps, errno.EINVAL, 0)
        self.assert_rejected(copy.deepcopy(self.steps[:-1]), errno.EINVAL, len(self.steps) - 2)
        steps = copy.deepcopy(self.steps)
        leaf = steps[-1].q
        composite = next(v for v in range(leaf + 2, leaf + 10 ** 6, 2) if not ec.probable_prime(v))
        steps[-1].q = composite
        self.assertEqual(verify(steps)[:2], (-1, errno.EDOM))

    def test_collapse_forgery_for_a_composite(self):
        # N = p * s. Modulo the prime s the step is genuine. Modulo the small
        # prime p, a, x and y are chosen so the addition law collapses to
        # (0:0:0) while computing qQ. Then Z_qQ = 0 (mod N) and Q is proper, so
        # a checker testing only Z would accept the composite N; the
        # gcd(Y_qQ, N) = 1 condition is what rejects it.
        rng = random.Random(7)
        s_prime, chain = ec.provable_prime(192, rng)
        st = chain[0]
        for p in (101, 103, 107):
            big_n = p * s_prime
            self.assertTrue(big_n % 3 and ec.q_bound(big_n) <= st.q < big_n)
            for _ in range(20000):
                ap, xp, yp = rng.randrange(p), rng.randrange(p), rng.randrange(p)
                bp = (yp * yp - xp ** 3 - ap * xp) % p
                if (4 * ap ** 3 + 27 * bp * bp) % p == 0:
                    continue
                qp = collapse_mod(st.k, (xp, yp, 1), ap, bp, p)
                if qp[2] % p and collapse_mod(st.q, qp, ap, bp, p) == (0, 0, 0):
                    break
            else:
                self.fail(f'no collapse found mod {p}')

            def crt(vs, vp):
                return (vs * p * pow(p, -1, s_prime) + vp * s_prime * pow(s_prime, -1, p)) % big_n

            forged = ec.Step(big_n, crt(st.a, ap), crt(st.x, xp), crt(st.y, yp), st.k, st.q)
            qf = collapse_mod(st.k, (forged.x, forged.y, 1), forged.a, forged.b, big_n)
            rf = collapse_mod(st.q, qf, forged.a, forged.b, big_n)
            self.assertEqual(math.gcd(qf[2], big_n), 1)
            self.assertEqual(rf[2] % big_n, 0)                # passes a Z-only test
            self.assertEqual(math.gcd(rf[1], big_n), p)       # caught by the Y test
            self.assertEqual(verify([forged] + chain[1:]), (-1, errno.EDOM, 0))

    def test_plain_composite_forgery(self):
        # Same CRT construction with random values mod p: qQ is not the
        # identity mod p.
        rng = random.Random(8)
        s_prime, chain = ec.provable_prime(160, rng)
        st, p = chain[0], 1009
        big_n = p * s_prime
        for _ in range(20):
            vp = [rng.randrange(p) for _ in range(3)]

            def crt(vs, v):
                return (vs * p * pow(p, -1, s_prime) + v * s_prime * pow(s_prime, -1, p)) % big_n

            forged = ec.Step(big_n, crt(st.a, vp[0]), crt(st.x, vp[1]), crt(st.y, vp[2]),
                             st.k, st.q)
            self.assertEqual(verify([forged] + chain[1:])[:2], (-1, errno.EDOM))

    def test_arguments(self):
        failed = c.c_size_t(5)
        self.assertEqual(LIB.crypto_ecpp_verify(None, 1, 0, c.byref(failed)), -1)
        self.assertEqual((c.get_errno(), failed.value), (errno.EINVAL, 5))
        arr = (Step * 1)()
        self.assertEqual(LIB.crypto_ecpp_verify(arr, 1, 0, c.byref(failed)), -1)
        self.assertEqual((c.get_errno(), failed.value), (errno.EINVAL, 0))
        self.assertEqual(LIB.crypto_ecpp_verify(arr, 0, 0, None), -1)
        self.assertEqual(c.get_errno(), errno.EINVAL)


def next_prime_above(v):
    v += 1
    while not ec.probable_prime(v):
        v += 1
    return v


class PariFixtureTests(unittest.TestCase):
    """Certificates for chosen primes from PARI/GP's primecert, converted with
    ecpp_cert.from_pari. Several are primes the class-number-1 generator in
    ecpp_cert.py cannot certify."""

    EXPECTED = {
        'p127': 2 ** 127 + 45,
        'p255': 2 ** 255 - 19,
        'p256': 2 ** 256 - 2 ** 224 + 2 ** 192 + 2 ** 96 - 1,
        'p333': next_prime_above(10 ** 100),
        'p513': next_prime_above(3 * 2 ** 511),
        'p1024': next_prime_above(2 ** 1023 + 2 ** 512),
    }

    @staticmethod
    def load(name):
        return ec.from_pari((FIXTURES / f'{name}.txt').read_text())

    def test_fixtures_prove_their_primes(self):
        self.assertEqual(sorted(f.stem for f in FIXTURES.glob('*.txt')), sorted(self.EXPECTED))
        for name, n in self.EXPECTED.items():
            steps = self.load(name)
            self.assertEqual(steps[0].n, n, name)
            python_check(steps)
            self.assertEqual(verify(steps)[:2], (0, 0), name)
        self.assertEqual(len(self.load('p1024')), 41)

    def test_mutated_fixture_rejected_at_that_step(self):
        steps = self.load('p1024')
        for index in (0, 20, 40):
            bad = copy.deepcopy(steps)
            bad[index].k += 1
            self.assertEqual(verify(bad), (-1, errno.EDOM, index))
        bad = copy.deepcopy(steps)
        bad[20].q = bad[21].n + 2
        self.assertEqual(verify(bad)[:2], (-1, errno.EDOM))

    def test_from_pari_input_errors(self):
        with self.assertRaises(ValueError):
            ec.from_pari('1 2 3 4 5')
        with self.assertRaises(ValueError):
            ec.from_pari('101 2 7 1 1 1')  # 7 does not divide 101 + 1 - 2
        self.assertEqual(ec.from_pari('# comment only\n\n'), [])

    def test_pari_cli_rejects_inputs_below_chain_domain(self):
        for n in (6, 41, (1 << 64) - 1):
            with self.subTest(n=n):
                result = subprocess.run(
                    [sys.executable, str(SCOPE_ROOT / 'ecpp_cert.py'), 'pari', str(n)],
                    env={'PATH': ''}, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')
                self.assertIn('N >= 2^64', result.stderr)
                self.assertNotIn('Traceback', result.stderr)

    def test_pari_error_detection_with_stub_gp(self):
        """gp can report an error and still exit 0, so pari_certificate scans
        its output. Exercised with a stub gp, since CI does not install PARI."""
        fixture = shlex.quote(str(FIXTURES / 'p127.txt'))
        cases = {
            # (stub body after draining stdin, expected outcome)
            'error_on_stdout': ("echo '  ***   user error: boom'; cat " + fixture, RuntimeError),
            'error_on_stderr': ("echo '  ***   not enough memory' >&2; cat " + fixture, RuntimeError),
            'no_output': (":", RuntimeError),
            'warning_only': ("echo '  *** Warning: increasing stack size' >&2; cat " + fixture, None),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, (body, expected) in cases.items():
                with self.subTest(name=name):
                    stub = Path(tmp) / f'gp_{name}'
                    stub.write_text('#!/bin/sh\ncat >/dev/null\n' + body + '\nexit 0\n')
                    stub.chmod(0o755)
                    if expected is None:
                        steps = ec.from_pari(ec.pari_certificate(1 << 64, gp=str(stub)))
                        self.assertEqual(steps[0].n, self.EXPECTED['p127'])
                    else:
                        with self.assertRaises(expected):
                            ec.pari_certificate(1 << 64, gp=str(stub))

    @unittest.skipUnless(shutil.which('gp'), 'PARI/GP not installed')
    def test_live_pari_certificates(self):
        rng = random.Random(12)
        with self.assertRaisesRegex(RuntimeError, 'not prime'):
            ec.pari_certificate(1 << 64)
        primes = [18446744073709551629]  # first prime above 2**64
        primes.extend(next_prime_above(rng.getrandbits(bits) | (1 << (bits - 1)))
                      for bits in (100, 300))
        for n in primes:
            steps = ec.from_pari(ec.pari_certificate(n))
            self.assertEqual(steps[0].n, n)
            self.assertEqual(verify(steps)[:2], (0, 0))


if __name__ == '__main__':
    unittest.main()
