"""Independent mathematical tests for the public-context Montgomery candidate."""
import ctypes as c
import errno
import os
from pathlib import Path
import random
import unittest

U=c.c_uint64
P=c.POINTER(U)
class Context(c.Structure):
    _fields_=[('limbs',c.c_size_t),('inverse',U),('modulus',U*1024)]
lib=c.CDLL(os.environ.get('CT_LIBRARY',str(Path(__file__).with_name('libcryptomult.so'))),use_errno=True)
lib.crypto_montgomery_ctx_init.argtypes=[c.POINTER(Context),P,c.c_size_t]
lib.crypto_montgomery_ctx_init.restype=c.c_int
lib.crypto_montgomery_mul_ct.argtypes=[P,c.c_size_t,P,c.c_size_t,P,c.c_size_t,c.POINTER(Context)]
lib.crypto_montgomery_mul_ct.restype=c.c_int

def limbs(x,n): return (U*n)(*((x>>(64*i))&((1<<64)-1) for i in range(n)))
def integer(a): return sum(int(v)<<(64*i) for i,v in enumerate(a))

class MontgomeryCandidateTests(unittest.TestCase):
    def test_oracle(self):
        rng=random.Random(48177)
        for n in (1,2,4,8,16,31,32,33,64,1024):
            R=1<<(64*n)
            for m in (3,R-1,R//2+1,rng.randrange(3,R)|1):
                ctx=Context()
                self.assertEqual(lib.crypto_montgomery_ctx_init(c.byref(ctx),limbs(m,n),n),0)
                rinv=pow(R,-1,m)
                cases=[(0,0),(0,m-1),(1,1),(m-1,m-1),(m-1,1)]
                cases += [(rng.randrange(m),rng.randrange(m)) for _ in range(10)]
                for a,b in cases:
                    out=limbs(0,n)
                    self.assertEqual(lib.crypto_montgomery_mul_ct(out,n,limbs(a,n),n,limbs(b,n),n,c.byref(ctx)),0)
                    self.assertEqual(c.get_errno(),0)
                    self.assertEqual(integer(out),(a*b*rinv)%m,(n,m,a,b))

    def test_public_setup_errors_preserve_context(self):
        ctx=Context()
        self.assertEqual(lib.crypto_montgomery_ctx_init(c.byref(ctx),limbs(11,1),1),0)
        saved=bytes(ctx)
        for m in (0,1,2,4):
            self.assertEqual(lib.crypto_montgomery_ctx_init(c.byref(ctx),limbs(m,1),1),-1)
            self.assertEqual(c.get_errno(),errno.EDOM)
            self.assertEqual(bytes(ctx),saved)
        for n in (0,1025,c.c_size_t(-1).value):
            self.assertEqual(lib.crypto_montgomery_ctx_init(c.byref(ctx),limbs(11,1),n),-1)
            self.assertEqual(c.get_errno(),errno.EINVAL)
            self.assertEqual(bytes(ctx),saved)

if __name__=='__main__': unittest.main()
