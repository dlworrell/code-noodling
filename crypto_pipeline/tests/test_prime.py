# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:75bd010e-f8a7-4581-935a-bfde6aaa3c8e

"""Differential tests for crypto_prime.h against Python integers.

Python's pow() is the oracle for modular exponentiation. For primality the
oracles are trial division (small values), a pure-Python Miller test over the
same Bach bound (independent re-implementation), and known primes and
pseudoprimes whose status each test re-checks before use.
"""
import ctypes as c
import errno
import math
import random
import unittest
from pathlib import Path

U = c.c_uint64
P = c.POINTER(U)
MASK = (1 << 64) - 1
SCOPE_ROOT = Path(__file__).resolve().parent.parent
LIB = c.CDLL(str(SCOPE_ROOT / 'libcryptomult.so'), use_errno=True)
LIB.crypto_mod_exp.argtypes = [P, P, P, c.c_size_t, P, c.c_size_t]
LIB.crypto_mod_exp.restype = c.c_int
LIB.crypto_grh_witness_bound.argtypes = [P, c.c_size_t]
LIB.crypto_grh_witness_bound.restype = U
LIB.crypto_is_prime_grh.argtypes = [c.POINTER(c.c_int), P, c.c_size_t, U]
LIB.crypto_is_prime_grh.restype = c.c_int


def limbs(value, n):
    return (U * n)(*[(value >> (64 * i)) & MASK for i in range(n)])


def integer(array, n):
    return sum(int(array[i]) << (64 * i) for i in range(n))


def size_of(value):
    return max(1, (value.bit_length() + 63) // 64)


def bach_bound(value):
    """ceil(2 * (L * 0.69315)^2) in exact integer arithmetic."""
    bits = value.bit_length()
    return -(-2 * bits * bits * 13863 * 13863 // (20000 * 20000))


def strong_probable_prime(n, a):
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(a, d, n)
    if x in (1, n - 1):
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def reference_is_prime(n):
    """Independent Python version of the same ERH-conditional Miller test."""
    if n < 4:
        return n >= 2
    if n % 2 == 0:
        return False
    last = min(bach_bound(n), n - 2)
    return all(strong_probable_prime(n, a) for a in range(2, last + 1))


def trial_division(n):
    if n < 2:
        return False
    return all(n % f for f in range(2, math.isqrt(n) + 1))


def c_mod_exp(base, exponent, mod, n=None, exp_n=None):
    n = n or size_of(max(base, mod))
    exp_n = exp_n or size_of(exponent)
    out = (U * n)()
    rc = LIB.crypto_mod_exp(out, limbs(base, n), limbs(exponent, exp_n), exp_n, limbs(mod, n), n)
    return rc, c.get_errno(), integer(out, n)


def c_is_prime(value, n=None, max_bases=0):
    n = n or size_of(value)
    result = c.c_int(-7)
    rc = LIB.crypto_is_prime_grh(c.byref(result), limbs(value, n), n, max_bases)
    return rc, c.get_errno(), result.value


class ModularExponentiationTests(unittest.TestCase):
    def test_random_against_pow(self):
        rng = random.Random(20261122)
        for n in (1, 2, 3, 4, 7, 17, 64):
            for _ in range(6):
                mod = rng.getrandbits(64 * n) | 1 | (1 << (64 * n - 1))
                base = rng.getrandbits(64 * n)  # need not be reduced
                exponent = rng.getrandbits(rng.choice((1, 63, 64 * n)))
                rc, err, out = c_mod_exp(base, exponent, mod, n)
                self.assertEqual((rc, err), (0, 0))
                self.assertEqual(out, pow(base, exponent, mod), (n, exponent.bit_length()))

    def test_exponent_width_differs_from_modulus_width(self):
        mod = (1 << 127) - 1
        exponent = (1 << 300) + 12345
        rc, err, out = c_mod_exp(5, exponent, mod, n=2, exp_n=5)
        self.assertEqual((rc, err, out), (0, 0, pow(5, exponent, mod)))

    def test_edge_values(self):
        mod = 1000000009
        self.assertEqual(c_mod_exp(7, 0, mod)[2], 1)
        self.assertEqual(c_mod_exp(0, 5, mod)[2], 0)
        self.assertEqual(c_mod_exp(mod, 3, mod)[2], 0)
        self.assertEqual(c_mod_exp(2, 10, 3)[2], 1)

    def test_domain_errors_leave_output_unchanged(self):
        for mod in (0, 1, 1000000008):
            out = (U * 1)(42)
            rc = LIB.crypto_mod_exp(out, limbs(3, 1), limbs(5, 1), 1, limbs(mod, 1), 1)
            self.assertEqual((rc, c.get_errno(), out[0]), (-1, errno.EDOM, 42))


class GrhPrimalityTests(unittest.TestCase):
    def test_small_values_match_trial_division(self):
        for v in range(0, 20000):
            rc, err, result = c_is_prime(v)
            self.assertEqual((rc, err), (0, 0), v)
            self.assertEqual(result, int(trial_division(v)), v)

    def test_random_odd_values_match_python_reference(self):
        rng = random.Random(7)
        for bits in (20, 40, 64, 65, 96, 128):
            for _ in range(25):
                v = rng.getrandbits(bits) | 1 | (1 << (bits - 1))
                rc, err, result = c_is_prime(v)
                self.assertEqual((rc, err), (0, 0))
                self.assertEqual(result, int(reference_is_prime(v)), v)

    def test_known_primes(self):
        for v in ((1 << 61) - 1, (1 << 64) - 59, (1 << 89) - 1, (1 << 107) - 1,
                  (1 << 127) - 1, (1 << 64) + 13):
            self.assertEqual(c_is_prime(v)[:3], (0, 0, 1), v)

    def test_prime_padded_with_zero_limbs(self):
        self.assertEqual(c_is_prime((1 << 89) - 1, n=5)[:3], (0, 0, 1))

    def test_strong_pseudoprimes_are_composite(self):
        # Each passes the strong test for every prime base up to last_base (re-checked here so
        # a wrong constant cannot pass silently). The Python reference finds a
        # witness, which proves each composite unconditionally.
        cases = {
            3215031751: 7,
            3825123056546413051: 23,
            318665857834031151167461: 37,
            3317044064679887385961981: 37,
        }
        for v, last_base in cases.items():
            prime_bases = [a for a in range(2, last_base + 1) if trial_division(a)]
            self.assertTrue(all(strong_probable_prime(v, a) for a in prime_bases), v)
            self.assertFalse(reference_is_prime(v), v)
            self.assertEqual(c_is_prime(v)[:3], (0, 0, 0), v)

    def test_carmichael_numbers_are_composite(self):
        for v in (561, 1105, 1729, 2465, 2821, 6601, 8911, 41041, 825265, 321197185,
                  5394826801, 232250619601, 9746347772161):
            self.assertFalse(reference_is_prime(v), v)
            self.assertEqual(c_is_prime(v)[:3], (0, 0, 0), v)

    def test_semiprimes_of_large_primes_are_composite(self):
        p, q = (1 << 61) - 1, (1 << 89) - 1
        self.assertEqual(c_is_prime(p * q)[:3], (0, 0, 0))

    def test_witness_bound(self):
        for v in (5, 2047, (1 << 61) - 1, (1 << 127) - 1, (1 << 1024) - 105):
            n = size_of(v)
            self.assertEqual(LIB.crypto_grh_witness_bound(limbs(v, n), n), bach_bound(v))
            self.assertGreaterEqual(bach_bound(v), 2 * math.log(v) ** 2)

    def test_work_cap_refuses_without_answering(self):
        v = (1 << 127) - 1
        needed = bach_bound(v) - 1  # bases 2..bound
        rc, err, result = c_is_prime(v, max_bases=needed - 1)
        self.assertEqual((rc, err, result), (-1, errno.ERANGE, -7))
        self.assertEqual(c_is_prime(v, max_bases=needed)[:3], (0, 0, 1))

    def test_argument_errors(self):
        result = c.c_int(-7)
        self.assertEqual(LIB.crypto_is_prime_grh(c.byref(result), limbs(7, 1), 0, 0), -1)
        self.assertEqual((c.get_errno(), result.value), (errno.EINVAL, -7))
        self.assertEqual(LIB.crypto_is_prime_grh(None, limbs(7, 1), 1, 0), -1)
        self.assertEqual(c.get_errno(), errno.EINVAL)


if __name__ == '__main__':
    unittest.main()
