"""Differential arithmetic tests using Python integers as the independent oracle."""
import ctypes as c
import errno
import math
from pathlib import Path
import random
import unittest

U = c.c_uint64
P = c.POINTER(U)
MASK = (1 << 64) - 1
LIB = c.CDLL(str(Path(__file__).with_name('libcryptomult.so')), use_errno=True)
for name in ('secure_comba_multiply', 'karatsuba_multiply', 'ntt_polynomial_multiply',
             'ct_mod_inverse_multiprecision', 'crypto_mod_inverse'):
    f = getattr(LIB, name)
    f.argtypes = [P, P, P, c.c_size_t]
    f.restype = c.c_int if name == 'crypto_mod_inverse' else None
LIB.crypto_multiply_reduce.argtypes = [P]*4 + [c.c_size_t, c.c_int]
LIB.crypto_multiply_reduce.restype = None
LIB.crypto_mod_multiply.argtypes = [P]*4 + [c.c_size_t]
LIB.crypto_mod_multiply.restype = c.c_int
LIB.secure_montgomery_reduction.argtypes = [P, P, c.c_size_t, U]
LIB.secure_montgomery_reduction.restype = None
LIB.ct_modular_reduce_final.argtypes = [P, P, c.c_size_t]
LIB.ct_modular_reduce_final.restype = None
LIB.compute_montgomery_inverse.argtypes = [U]
LIB.compute_montgomery_inverse.restype = U
LIB.crypto_blind.argtypes = [P]*4 + [c.c_size_t]
LIB.crypto_blind.restype = c.c_int
LIB.execute_unblinding.argtypes = [P]*4 + [c.c_size_t]
LIB.execute_unblinding.restype = None


def limbs(x, n):
    return (U*n)(*((x >> (64*i)) & MASK for i in range(n)))


def integer(a):
    return sum(int(v) << (64*i) for i, v in enumerate(a))


class ArithmeticTests(unittest.TestCase):
    def test_ntt_parameters(self):
        # Verify primes and root order independently of the C transform.
        for p,factors in ((998244353,(2,7,17)),(1004535809,(2,479))):
            self.assertTrue(all(p % d for d in range(2,math.isqrt(p)+1)))
            self.assertEqual((p-1)%8192,0)
            for q in factors: self.assertNotEqual(pow(3,(p-1)//q,p),1)
        self.assertLess(4096*65535**2,998244353*1004535809)

    def test_full_products(self):
        rng = random.Random(20261013)
        for n in (1, 2, 4, 15, 16, 17, 18, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024):
            R = 1 << (64*n)
            cases = [(0, R-1), (1, R-1), (R-1, R-1), (R//2+1, R//2-1)]
            cases += [(rng.randrange(R), rng.randrange(R)) for _ in range(8)]
            for a, b in cases:
                for name in ('secure_comba_multiply', 'karatsuba_multiply', 'ntt_polynomial_multiply'):
                    out = limbs(0, 2*n)
                    getattr(LIB, name)(out, limbs(a,n), limbs(b,n), n)
                    self.assertEqual(c.get_errno(), 0)
                    self.assertEqual(integer(out), a*b, (name,n,a,b))

    def test_montgomery(self):
        rng = random.Random(57)
        for n in (1, 2, 4, 17, 31, 32, 33, 63, 64, 65, 1023, 1024):
            R = 1 << (64*n)
            for m in (3, R-1, R//2+1, rng.randrange(3,R)|1):
                rinv = pow(R, -1, m)
                inv = LIB.compute_montgomery_inverse(m & MASK)
                self.assertEqual(((m & MASK)*inv) & MASK, MASK)
                for a,b in ((0,m-1), (m-1,m-1), (1,2), (rng.randrange(m),rng.randrange(m))):
                    for mode in (0,1):
                        out = limbs(0,2*n)
                        LIB.crypto_multiply_reduce(out,limbs(a,n),limbs(b,n),limbs(m,n),n,mode)
                        self.assertEqual(c.get_errno(),0)
                        self.assertEqual(integer(out),(a*b*rinv)%m,(n,m,mode))
                for T in (0, m*R-1, rng.randrange(m*R)):
                    out = limbs(T,2*n)
                    LIB.secure_montgomery_reduction(out,limbs(m,n),n,inv)
                    self.assertEqual(c.get_errno(),0)
                    self.assertEqual(integer(out),(T*rinv)%m)

    def test_final_subtraction(self):
        # Borrow into a UINT64_MAX limb used to wrap mod[i]+borrow to zero.
        m = (MASK << 64) | 2
        for r in (0, 1, m-1, m, m+1, (1<<128)-1):
            out = limbs(r,2)
            LIB.ct_modular_reduce_final(out,limbs(m,2),2)
            self.assertEqual(integer(out),r % m)

    def test_ordinary_modular_product(self):
        rng = random.Random(998)
        for n in (1,2,4,17,32,33,64,1024):
            R=1<<(64*n)
            for m in (2,3,R-2,R-1):
                for a,b in ((R-1,R-1),(rng.randrange(R),rng.randrange(R))):
                    out=limbs(0,n)
                    self.assertEqual(LIB.crypto_mod_multiply(out,limbs(a,n),limbs(b,n),limbs(m,n),n),0)
                    self.assertEqual(integer(out),(a*b)%m)

    def test_inverse(self):
        rng=random.Random(789)
        cases=[(a,m,1) for m in range(3,100,2) for a in range(0,2*m)]
        for n in (1,2,4,17,32,33,64):
            R=1<<(64*n)
            cases += [(rng.randrange(R),rng.randrange(3,R)|1,n) for _ in range(20)]
        cases += [(2,(1<<65536)-1,1024),(3,11,1024)]
        for a,m,n in cases:
            out=limbs(123,n)
            rc=LIB.crypto_mod_inverse(out,limbs(a,n),limbs(m,n),n)
            if math.gcd(a,m)==1:
                self.assertEqual(rc,0,(a,m,n))
                self.assertEqual(integer(out),pow(a,-1,m),(a,m,n))
            else:
                self.assertEqual(rc,-1)
                self.assertEqual(c.get_errno(),errno.EDOM)
                self.assertEqual(integer(out),123)
        out=limbs(0,1)
        LIB.ct_mod_inverse_multiprecision(out,limbs(3,1),limbs(11,1),1)
        self.assertEqual(integer(out),4)

    def test_blinding_roundtrip(self):
        for n,m in ((1,3),(1,15),(4,11),(4,(1<<256)-189)):
            a=(1<<(64*n))-1
            for _ in range(8):
                blinded,factor,out=limbs(0,n),limbs(0,n),limbs(0,n)
                self.assertEqual(LIB.crypto_blind(blinded,factor,limbs(a,n),limbs(m,n),n),0)
                k=integer(factor)
                self.assertTrue(0<k<m)
                self.assertEqual(math.gcd(k,m),1)
                self.assertEqual(integer(blinded),(a*k)%m)
                LIB.execute_unblinding(out,blinded,factor,limbs(m,n),n)
                self.assertEqual(c.get_errno(),0)
                self.assertEqual(integer(out),a%m)

    def test_domain_errors_preserve_output(self):
        for m in (0,1,2,4):
            out=limbs(123,2)
            LIB.crypto_multiply_reduce(out,limbs(1,1),limbs(1,1),limbs(m,1),1,0)
            self.assertEqual(c.get_errno(),errno.EDOM)
            self.assertEqual(integer(out),123)
            self.assertEqual(LIB.crypto_mod_inverse(out,limbs(1,1),limbs(m,1),1),-1)
            self.assertEqual(integer(out),123)
        out=limbs(123,2)
        LIB.crypto_multiply_reduce(out,limbs(11,1),limbs(1,1),limbs(11,1),1,0)
        self.assertEqual(c.get_errno(),errno.EDOM)
        self.assertEqual(integer(out),123)
        for T,inv in ((11<<64,LIB.compute_montgomery_inverse(11)),(123,0)):
            out=limbs(T,2)
            LIB.secure_montgomery_reduction(out,limbs(11,1),1,inv)
            self.assertEqual(c.get_errno(),errno.EDOM)
            self.assertEqual(integer(out),T)
        out=limbs(123,1)
        LIB.execute_unblinding(out,limbs(4,1),limbs(3,1),limbs(15,1),1)
        self.assertEqual(c.get_errno(),errno.EDOM)
        self.assertEqual(integer(out),123)

    def test_python_binding(self):
        from linking_ctypes import secure_multiply_wrapper
        m=(1<<256)-189
        for a,b in ((1<<230,1<<199),(-42,7),(m+13,2*m+9)):
            self.assertEqual(secure_multiply_wrapper(a,b,m),(a*b)%m)
        self.assertEqual(secure_multiply_wrapper(99,123,100),77)
        with self.assertRaises(ValueError): secure_multiply_wrapper(1,2,1)


if __name__=='__main__':
    unittest.main()
