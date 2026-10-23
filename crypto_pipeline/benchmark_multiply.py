# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:a1436158-ff0f-452a-8d73-c4b5f360a45f

"""Compare full-product algorithms locally; build with make shared first."""
import ctypes as c
from pathlib import Path
import random
import statistics
import time

lib = c.CDLL(str(Path(__file__).with_name('libcryptomult.so')))
U = c.c_uint64
names = ('secure_comba_multiply', 'karatsuba_multiply', 'ntt_polynomial_multiply')
for name in names:
    f = getattr(lib, name)
    f.argtypes = [c.POINTER(U)]*3 + [c.c_size_t]
    f.restype = None
rng = random.Random(20261013)
print('limbs,Comba_us,Karatsuba_us,NTT_us')
for n in (4,16,32,64,128,256,512,1024):
    a,b = [(U*n)(*(rng.getrandbits(64) for _ in range(n))) for _ in range(2)]
    out = (U*(2*n))()
    times=[]
    for name in names:
        fn=getattr(lib,name)
        fn(out,a,b,n)
        samples=[]
        for _ in range(7):
            start=time.perf_counter_ns()
            for _ in range(10): fn(out,a,b,n)
            samples.append((time.perf_counter_ns()-start)/10000)
        times.append(statistics.median(samples))
    print(str(n)+','+','.join(f'{t:.2f}' for t in times))
