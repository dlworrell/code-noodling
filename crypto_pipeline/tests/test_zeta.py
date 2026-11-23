# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:e1a01bd4-4ae6-4f86-ad7d-710479b666ef

"""Full-height checks of the zeta-zeros verification (crypto_zeta.h).

Runs ./zeta_verify (built by `make all`) to T = 10^4 with both methods and
compares the results with published values (OEIS, retrieved 2026-11-23):

- A072080: N(100) = 29, N(1000) = 649, N(10^4) = 10142;
- A114856: the bad Gram points (-1)^n Z(g_n) < 0 begin 126, 134, 195, 211,
  232, 254, 288, 367 (the full list up to n = 10143, 840 entries, was
  compared locally; see docs/engineering/CRYPTO-ZETA-ZEROS.md);
- A013629: zeros 6709 and 6710 (Lehmer's pair) both have floor 7005;
- A058303 and A065434: the first two zeros.

Both runs are proofs: Euler-Maclaurin everywhere (about 4 minutes), and
Riemann-Siegel with Gabcke's bound (Dissertation 1979, Satz 3.2.2 (b)) for
t >= 200 (seconds). They must agree.
"""
import json
import subprocess
import unittest
from pathlib import Path

SCOPE_ROOT = Path(__file__).resolve().parent.parent
BINARY = SCOPE_ROOT / 'zeta_verify'

FIRST_BAD_GRAM = [126, 134, 195, 211, 232, 254, 288, 367]
FIRST_ZEROS = [14.134725141734693790, 21.022039638771554993]


def run(*args):
    proc = subprocess.run([str(BINARY), '--json', *args], capture_output=True, text=True,
                          check=False, timeout=1800)
    return proc.returncode, json.loads(proc.stdout)


class ZetaZerosTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.exists():
            raise unittest.SkipTest('zeta_verify not built; run make all')
        cls.em_rc, cls.em = run('--brackets', '10000', '100', '1000')
        cls.rs_rc, cls.rs = run('--rs', '--brackets', '10000', '100', '1000')

    def check_result(self, rc, res, used_riemann_siegel):
        self.assertEqual(rc, 0, res['reason'])
        self.assertEqual(res['status'], 0)
        self.assertEqual(res['used_riemann_siegel'], used_riemann_siegel)
        self.assertEqual(res['zeros_to_height'], 10142)
        self.assertEqual(res['checkpoints'], [[100, 29], [1000, 649]])
        # Turing: every zero up to t0 is a counted sign change.
        self.assertGreaterEqual(res['t0'], 10000)
        self.assertGreater(res['t1'], res['t0'])
        self.assertEqual(res['sign_changes_to_t0'], res['turing_bound'])
        self.assertLess(res['turing_upper'], res['turing_bound'] + 1)
        # Gram's law first fails at g_126.
        self.assertEqual(res['bad_gram'][:len(FIRST_BAD_GRAM)], FIRST_BAD_GRAM)

    def test_euler_maclaurin_proof(self):
        self.check_result(self.em_rc, self.em, 0)
        self.assertEqual(self.em['rs_fallbacks'], 0)
        self.assertEqual(self.em['evaluations'], self.em['em_evaluations'])

    def test_riemann_siegel_proof(self):
        self.check_result(self.rs_rc, self.rs, 1)

    def test_methods_agree(self):
        self.assertEqual(self.em['bad_gram'], self.rs['bad_gram'])
        self.assertEqual(self.em['sign_changes_to_t0'], self.rs['sign_changes_to_t0'])

    def test_brackets(self):
        br = self.em['brackets']
        for lo, hi in br:
            self.assertLess(lo, hi)
        for (_, hi), (lo, _) in zip(br, br[1:]):
            self.assertLessEqual(hi, lo)  # disjoint and ordered
        for k, gamma in enumerate(FIRST_ZEROS):
            self.assertTrue(br[k][0] < gamma < br[k][1])
        # Lehmer's pair: zeros 6709 and 6710 (1-based) in two brackets
        # inside [7005, 7006).
        for k in (6708, 6709):
            lo, hi = br[k]
            self.assertTrue(7005 <= lo and hi < 7006, (lo, hi))
        self.assertLess(br[6709][0] - br[6708][0], 0.1)

    def test_negative_usage(self):
        proc = subprocess.run([str(BINARY), '0'], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 1)
        proc = subprocess.run([str(BINARY), 'x'], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 2)


if __name__ == '__main__':
    unittest.main()
