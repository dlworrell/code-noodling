# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:8268cc06-105c-450c-b56a-69b86bc8da97

"""Untrusted generator for mixed Pocklington/ECPP certificates (crypto_mixed.h).

Each prime is proven by whichever method works for it:

- Pocklington, when n - 1 = (small primes) * R with R = 1 or R prime. Then
  n - 1 is fully factored, F = n - 1 > sqrt(n), and R (if >= 2^64) is proven
  by its own node.
- ECPP (class-number-1 curves, ecpp_cert.py), whose q is proven by its own node.

With alternate=True the preferred method switches at each depth, so chains mix
both kinds. Nothing here is trusted: crypto_mixed_verify checks the result.
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field

import ecpp_cert as ec
import pocklington_cert as pc

U64 = 1 << 64
POCKLINGTON, ECPP = "pocklington", "ecpp"
TRIAL_BOUND = 1 << 16


def _small_primes(bound: int) -> list[int]:
    sieve = bytearray([1]) * bound
    sieve[0:2] = b"\x00\x00"
    for i in range(2, math.isqrt(bound) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i, v in enumerate(sieve) if v]


SMALL_PRIMES = _small_primes(TRIAL_BOUND)


@dataclass
class MixedNode:
    kind: str
    pocklington: pc.Node | None = None   # factor.child indexes MixedCertificate.nodes
    ecpp: ec.Step | None = None
    q_child: int = -1                    # ECPP: node proving step.q, or -1 for a 64-bit leaf

    @property
    def n(self) -> int:
        return self.pocklington.n if self.kind == POCKLINGTON else self.ecpp.n


@dataclass
class MixedCertificate:
    nodes: list[MixedNode] = field(default_factory=list)
    root: int = 0

    def add(self, node: MixedNode) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def edges(self) -> set[tuple[str, str]]:
        """(parent kind, child kind) for every link between nodes."""
        out = set()
        for node in self.nodes:
            if node.kind == POCKLINGTON:
                children = [f.child for f in node.pocklington.factors if f.child >= 0]
            else:
                children = [node.q_child] if node.q_child >= 0 else []
            out.update((node.kind, self.nodes[c].kind) for c in children)
        return out


def split_n_minus_1(n: int) -> dict[int, int] | None:
    """{q: e} with prod q^e = n - 1, if n - 1 is (primes < 2^16) * (1 or one prime)."""
    m, factors = n - 1, {}
    for p in SMALL_PRIMES:
        if p * p > m:
            break
        while m % p == 0:
            factors[p] = factors.get(p, 0) + 1
            m //= p
    if m > 1:
        if not pc.probable_prime(m):
            return None
        factors[m] = factors.get(m, 0) + 1
    return factors


def certify(n: int, rng: random.Random | None = None, alternate: bool = True,
            first: str = POCKLINGTON, budget: int = 400) -> MixedCertificate | None:
    """A mixed certificate for prime n >= 2^64, or None if none was found."""
    rng = rng or random.Random()
    cert, memo, state = MixedCertificate(), {}, {"left": budget}

    def prove(m: int, depth: int) -> int | None:
        """Node index proving m, -1 for a 64-bit leaf, or None."""
        if m < U64:
            return -1 if pc.probable_prime(m) else None  # exact below 2^64
        if m in memo:
            return memo[m]
        state["left"] -= 1
        if state["left"] < 0:
            return None
        order = [first, ECPP if first == POCKLINGTON else POCKLINGTON]
        if alternate and depth % 2:
            order.reverse()
        for method in order:
            index = (by_pocklington if method == POCKLINGTON else by_ecpp)(m, depth)
            if index is not None:
                memo[m] = index
                return index
        return None

    def by_pocklington(m: int, depth: int) -> int | None:
        factors = split_n_minus_1(m)
        if factors is None:
            return None
        node = pc.Node(m)
        for q, e in sorted(factors.items()):
            child = prove(q, depth + 1) if q >= U64 else -1
            if child is None:
                return None
            node.factors.append(pc.Factor(q, e, pc.find_base(m, q), child))
        return cert.add(MixedNode(POCKLINGTON, pocklington=node))

    def by_ecpp(m: int, depth: int) -> int | None:
        if m % 3 == 0:
            return None
        for q, order, j in ec._candidates(m):  # noqa: SLF001 (same package)
            child = prove(q, depth + 1)
            if child is None:
                continue
            curve = ec._curve_with_order(m, order, j, rng)  # noqa: SLF001
            if curve is None:
                continue
            a, b, p = curve
            k = order // q
            for _ in range(20):
                if ec.affine_mul(k, p, a, m) is not None:
                    return cert.add(MixedNode(ECPP, ecpp=ec.Step(m, a, p[0], p[1], k, q), q_child=child))
                p = ec.random_point(a, b, m, rng)
        return None

    if n < U64 or not pc.probable_prime(n):
        return None
    root = prove(n, 0)
    if root is None or root < 0:
        return None
    cert.root = root
    return cert


def main(argv: list[str]) -> int:
    bits = int(argv[1]) if len(argv) > 1 else 192
    rng = random.Random()
    while True:
        n = rng.getrandbits(bits) | 1 | (1 << (bits - 1))
        if not pc.probable_prime(n):
            continue
        cert = certify(n, rng)
        if cert is not None:
            print(n)
            kinds = ", ".join(f"{p}->{c}" for p, c in sorted(cert.edges())) or "none"
            print(f"{len(cert.nodes)} node(s); links: {kinds}", file=sys.stderr)
            return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
