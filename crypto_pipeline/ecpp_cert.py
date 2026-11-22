# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:8a952a6a-eadd-4815-8716-24c3723c75d4

"""Builds ECPP primality certificates for crypto_ecpp.h.

This module is NOT trusted: crypto_ecpp_verify accepts a chain only if every
condition holds, so a mistake here can only cause a rejection.

It uses complex multiplication with the nine discriminants of class number 1
(D = -3, -4, -7, -8, -11, -19, -43, -67, -163), whose j-invariants are
integers. For a prime N with (D/N) = 1, Cornacchia's algorithm solves
4N = t^2 + |D| v^2; the curves with that j-invariant then have N + 1 - u
points for the traces u of the units times (t + v sqrt(D)) / 2. An order m is
usable when removing its small prime factors leaves a probable prime q with
(floor(N^(1/4)) + 2)^2 <= q < N. The search descends depth-first until q < 2^64.

Class number 1 alone does not certify every prime: some N have no usable
order at any depth. provable_prime() draws primes until one works; certify()
returns None on failure. Arbitrary primes need more discriminants or an
external prover (docs/engineering/CRYPTO-ECPP.md, slice 2).

Usage: python3 ecpp_cert.py BITS  prints a random provable prime and its chain
length.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass

from pocklington_cert import probable_prime

U64 = 1 << 64
CM_DISCRIMINANTS = ((-3, 0), (-4, 1728), (-7, -3375), (-8, 8000), (-11, -32768),
                    (-19, -884736), (-43, -884736000), (-67, -147197952000),
                    (-163, -262537412640768000))
SMOOTH_BOUND = 1 << 15


def _primorial(bound: int) -> int:
    sieve = bytearray([1]) * bound
    sieve[0:2] = b"\x00\x00"
    for i in range(2, math.isqrt(bound) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return math.prod(i for i in range(bound) if sieve[i])


PRIMORIAL = _primorial(SMOOTH_BOUND)


@dataclass
class Step:
    n: int
    a: int
    x: int
    y: int
    k: int
    q: int

    @property
    def b(self) -> int:
        return (self.y * self.y - self.x ** 3 - self.a * self.x) % self.n


def q_bound(n: int) -> int:
    """Smallest q the checker accepts: (floor(n^(1/4)) + 2)^2."""
    return (math.isqrt(math.isqrt(n)) + 2) ** 2


# ---- field helpers (N prime) ---------------------------------------------

def sqrt_mod(a: int, p: int) -> int | None:
    """Tonelli-Shanks; None when a is a non-residue."""
    a %= p
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        return None
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return r


def cornacchia4(d: int, p: int) -> tuple[int, int] | None:
    """Solves t^2 + |d| v^2 = 4p for prime p and d < 0, d = 0 or 1 mod 4."""
    x0 = sqrt_mod(d, p)
    if x0 is None:
        return None
    if (x0 - d) % 2:
        x0 = p - x0
    a, b, limit = 2 * p, x0, math.isqrt(4 * p)
    while b > limit:
        a, b = b, a % b
    c, rem = divmod(4 * p - b * b, -d)
    if rem or math.isqrt(c) ** 2 != c:
        return None
    return b, math.isqrt(c)


def traces(d: int, t: int, v: int) -> list[int]:
    if d == -3:
        return [t, -t, (t + 3 * v) // 2, -(t + 3 * v) // 2, (t - 3 * v) // 2, -(t - 3 * v) // 2]
    if d == -4:
        return [t, -t, 2 * v, -2 * v]
    return [t, -t]


# ---- affine curve arithmetic (N prime; used only to build certificates) ----

def affine_add(p1, p2, a: int, n: int):
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    (x1, y1), (x2, y2) = p1, p2
    if x1 == x2 and (y1 + y2) % n == 0:
        return None
    if p1 == p2:
        lam = (3 * x1 * x1 + a) * pow(2 * y1, -1, n) % n
    else:
        lam = (y2 - y1) * pow(x2 - x1, -1, n) % n
    x3 = (lam * lam - x1 - x2) % n
    return x3, (lam * (x1 - x3) - y1) % n


def affine_mul(k: int, p, a: int, n: int):
    r = None
    for bit in bin(k)[2:]:
        r = affine_add(r, r, a, n)
        if bit == "1":
            r = affine_add(r, p, a, n)
    return r


def random_point(a: int, b: int, n: int, rng: random.Random):
    while True:
        x = rng.randrange(n)
        y = sqrt_mod(x ** 3 + a * x + b, n)
        if y is not None and y != 0:
            return x, y


# ---- reference projective law (the checker's formula, for differential tests)

def rcb_add(p1, p2, a: int, b3: int, n: int):
    """Renes-Costello-Batina 2016 Algorithm 1, step for step, as in crypto_ecpp.c."""
    (X1, Y1, Z1), (X2, Y2, Z2) = p1, p2
    t0, t1, t2 = X1 * X2 % n, Y1 * Y2 % n, Z1 * Z2 % n
    t3 = (X1 + Y1) * (X2 + Y2) % n
    t3 = (t3 - t0 - t1) % n
    t4 = (X1 + Z1) * (X2 + Z2) % n
    t4 = (t4 - t0 - t2) % n
    t5 = (Y1 + Z1) * (Y2 + Z2) % n
    t5 = (t5 - t1 - t2) % n
    Z3 = (b3 * t2 + a * t4) % n
    X3 = (t1 - Z3) % n
    Z3 = (t1 + Z3) % n
    Y3 = X3 * Z3 % n
    t1 = (3 * t0 + a * t2) % n
    t2 = a * (t0 - a * t2) % n
    t4 = (b3 * t4 + t2) % n
    Y3 = (Y3 + t1 * t4) % n
    X3 = (t3 * X3 - t5 * t4) % n
    Z3 = (t5 * Z3 + t3 * t1) % n
    return X3, Y3, Z3


def projective_mul(k: int, x: int, y: int, a: int, b: int, n: int):
    r, p, b3 = (0, 1 % n, 0), (x % n, y % n, 1 % n), 3 * b % n
    for bit in bin(k)[2:] if k else "":
        r = rcb_add(r, r, a, b3, n)
        if bit == "1":
            r = rcb_add(r, p, a, b3, n)
    return r


# ---- certificate search ----------------------------------------------------

def _strip_smooth(m: int) -> int:
    while (g := math.gcd(m, PRIMORIAL)) > 1:
        m //= g
    return m


def _candidates(n: int) -> list[tuple[int, int, int]]:
    """(q, m, j) for every usable order m = k q, smallest q first."""
    out, bound = [], q_bound(n)
    for d, j in CM_DISCRIMINANTS:
        if pow(d % n, (n - 1) // 2, n) != 1:
            continue
        tv = cornacchia4(d, n)
        if tv is None:
            continue
        for u in traces(d, *tv):
            m = n + 1 - u
            q = _strip_smooth(m)
            if bound <= q < n and probable_prime(q):
                out.append((q, m, j))
    return sorted(set(out))


def _curve_with_order(n: int, m: int, j: int, rng: random.Random, tries: int = 60):
    """(a, b, P) with m P = O for a curve of j-invariant j, or None."""
    for _ in range(tries):
        g = rng.randrange(1, n)
        if j == 0:
            a, b = 0, g
        elif j == 1728:
            a, b = g, 0
        else:
            c = j * pow(1728 - j, -1, n) % n
            a, b = 3 * c * g * g % n, 2 * c * g ** 3 % n
        if (4 * a ** 3 + 27 * b * b) % n == 0:
            continue
        p = random_point(a, b, n, rng)
        if affine_mul(m, p, a, n) is None:
            return a, b, p
    return None


def certify(n: int, rng: random.Random | None = None, budget: int = 400) -> list[Step] | None:
    """An ECPP chain for prime n (n >= 2^64), or None if none was found."""
    rng = rng or random.Random()
    state = {"left": budget}

    def chain(n_: int) -> list[Step] | None:
        if n_ < U64:
            return [] if probable_prime(n_) else None  # exact below 2^64
        state["left"] -= 1
        if state["left"] < 0:
            return None
        for q, m, j in _candidates(n_):
            rest = chain(q)
            if rest is None:
                continue
            curve = _curve_with_order(n_, m, j, rng)
            if curve is None:
                continue
            a, b, p = curve
            k = m // q
            for _ in range(20):
                if affine_mul(k, p, a, n_) is not None:
                    return [Step(n_, a, p[0], p[1], k, q)] + rest
                p = random_point(a, b, n_, rng)
        return None

    if n < U64 or not probable_prime(n):
        return None
    return chain(n)


def provable_prime(bits: int, rng: random.Random) -> tuple[int, list[Step]]:
    """A random bits-bit prime (bits > 64) with its ECPP chain."""
    while True:
        n = rng.getrandbits(bits) | 1 | (1 << (bits - 1))
        if n % 3 == 0 or not probable_prime(n):
            continue
        steps = certify(n, rng)
        if steps:
            return n, steps


def main(argv: list[str]) -> int:
    bits = int(argv[1]) if len(argv) > 1 else 256
    n, steps = provable_prime(bits, random.Random())
    print(n)
    print(f"{len(steps)} step(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
