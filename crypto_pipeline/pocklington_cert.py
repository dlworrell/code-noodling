# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:ddd81355-aea9-44fa-a019-72a714a4039f

"""Builds Pocklington primality certificates for crypto_cert.h.

This module is NOT trusted: the C checker (crypto_pocklington_verify) accepts a
certificate only if every condition holds, so a mistake here can only cause a
rejection. It provides:

- Certificate: a tree of nodes, each giving N and the factored part F of N - 1
  as (q, exponent, base, child) entries, child = -1 for q < 2^64;
- certify(n, factorization): a certificate from a known factorization of part
  of n - 1 (for example Proth numbers k * 2^m + 1, or Mersenne primes);
- provable_prime(bits, rng): a random prime with its certificate, built
  bottom-up in the style of Maurer's method (N = 2 R q + 1 with q > sqrt(N)/2).

Usage: python3 pocklington_cert.py BITS  prints a random provable prime and the
size of its certificate.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field

U64 = 1 << 64
SMALL_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def probable_prime(n: int) -> bool:
    """Strong test to the first 12 prime bases: exact below 3.18e23, a
    probable-prime filter above. Used only to choose candidates."""
    if n < 2:
        return False
    for p in SMALL_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in SMALL_BASES + (41, 43, 47, 53, 59, 61, 67, 71):
        if a % n == 0:
            continue  # n is one of the extra bases: already shown prime by 2..37
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@dataclass
class Factor:
    q: int
    exponent: int
    base: int
    child: int  # node index proving q, or -1 when q < 2^64


@dataclass
class Node:
    n: int
    factors: list[Factor] = field(default_factory=list)


@dataclass
class Certificate:
    nodes: list[Node] = field(default_factory=list)
    root: int = 0

    def add(self, node: Node) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1


def find_base(n: int, q: int, start: int = 2) -> int:
    """Smallest a >= start satisfying both Pocklington conditions for q."""
    for a in range(start, min(n - 1, start + 100000)):
        if pow(a, n - 1, n) == 1 and math.gcd(pow(a, (n - 1) // q, n) - 1, n) == 1:
            return a
    raise ValueError(f"no Pocklington base found for q={q}; is n prime?")


def certify(n: int, factorization: dict[int, int], children: dict[int, Certificate] | None = None,
            cert: Certificate | None = None) -> Certificate:
    """Certificate for n from {q: exponent} with prod q^e dividing n - 1 and
    (prod q^e)^2 > n. Each q >= 2^64 needs a certificate in `children`."""
    children = children or {}
    cert = cert or Certificate()
    f = math.prod(q ** e for q, e in factorization.items())
    if (n - 1) % f or f * f <= n:
        raise ValueError("factored part must divide n - 1 and exceed sqrt(n)")
    node = Node(n)
    for q, e in sorted(factorization.items()):
        child = -1
        if q >= U64:
            child = graft(cert, children[q])
        node.factors.append(Factor(q, e, find_base(n, q), child))
    cert.root = cert.add(node)
    return cert


def graft(cert: Certificate, sub: Certificate) -> int:
    """Copies sub's nodes into cert and returns the index of sub's root."""
    offset = len(cert.nodes)
    for node in sub.nodes:
        cert.nodes.append(Node(node.n, [Factor(f.q, f.exponent, f.base,
                                               f.child + offset if f.child >= 0 else -1)
                                        for f in node.factors]))
    return sub.root + offset


def random_small_prime(bits: int, rng: random.Random) -> int:
    while True:
        v = rng.getrandbits(bits) | 1 | (1 << (bits - 1))
        if probable_prime(v):  # exact below 2^64
            return v


def provable_prime(bits: int, rng: random.Random) -> tuple[int, Certificate | None]:
    """A random bits-bit prime and its certificate (None for primes < 2^62,
    which the checker proves directly as 64-bit leaves)."""
    if bits <= 62:
        return random_small_prime(bits, rng), None
    q_bits = bits // 2 + 2
    q, q_cert = provable_prime(q_bits, rng)
    lo = (1 << (bits - 1)) // (2 * q) + 1
    hi = ((1 << bits) - 1) // (2 * q)
    while True:
        r = rng.randrange(lo, hi + 1)
        n = 2 * r * q + 1
        if n.bit_length() != bits or not probable_prime(n):
            continue
        children = {q: q_cert} if q_cert is not None else {}
        try:
            return n, certify(n, {2: 1, q: 1}, children)
        except ValueError:
            continue  # a probable prime that is not prime: try another R


def main(argv: list[str]) -> int:
    bits = int(argv[1]) if len(argv) > 1 else 512
    n, cert = provable_prime(bits, random.Random())
    print(n)
    print(f"{len(cert.nodes) if cert else 0} node(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
