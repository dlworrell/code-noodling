"""Ordinary modular multiplication binding; experimental, variable-time code."""
import ctypes
from pathlib import Path

limb_t = ctypes.c_uint64
size_t = ctypes.c_size_t
crypto_lib = ctypes.CDLL(str(Path(__file__).with_name('libcryptomult.so')), use_errno=True)
crypto_lib.crypto_mod_multiply.argtypes = [ctypes.POINTER(limb_t)] * 4 + [size_t]
crypto_lib.crypto_mod_multiply.restype = ctypes.c_int


def secure_multiply_wrapper(param_a, param_b, param_mod):
    """Return (a*b)%mod; legacy name does not imply constant-time execution."""
    if not all(isinstance(x, int) for x in (param_a, param_b, param_mod)):
        raise TypeError('operands and modulus must be integers')
    if param_mod <= 1:
        raise ValueError('modulus must be greater than 1')
    size = (param_mod.bit_length() + 63) // 64
    if size > 1024:
        raise ValueError('modulus exceeds 1024 limbs')
    array = limb_t * size
    mask = (1 << 64) - 1
    def limbs(x):
        return array(*((x >> (64*i)) & mask for i in range(size)))
    a, b, mod = limbs(param_a % param_mod), limbs(param_b % param_mod), limbs(param_mod)
    result = array()
    if crypto_lib.crypto_mod_multiply(result, a, b, mod, size) != 0:
        raise OSError(ctypes.get_errno(), 'modular multiplication failed')
    return sum(int(result[i]) << (64*i) for i in range(size))
