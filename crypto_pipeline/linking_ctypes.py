import ctypes
import os

# Load the compiled, side-channel safe C-library
lib_path = os.path.abspath("./libcryptomult.so")
crypto_lib = ctypes.CDLL(lib_path)

# Define the data structure types for stability
limb_t = ctypes.c_uint64
size_t = ctypes.c_size_t

# Prototype: void crypto_multiply_reduce(limb_t *res, const limb_t *a, const limb_t *b, const limb_t *mod, size_t size, int is_constant_time)
crypto_lib.crypto_multiply_reduce.argtypes = [
    ctypes.POINTER(limb_t), # result
    ctypes.POINTER(limb_t), # a
    ctypes.POINTER(limb_t), # b
    ctypes.POINTER(limb_t), # mod
    size_t,                 # size
    ctypes.c_int            # is_constant_time
]

def secure_multiply_wrapper(param_a, param_b, param_mod):
    # Map Python data primitives to 64-bit multi-precision limb arrays
    size = 4  # e.g., 256-bit operations
    ArrayType = limb_t * size
    
    c_a = ArrayType(param_a & 0xFFFFFFFFFFFFFFFF, (param_a >> 64) & 0xFFFFFFFFFFFFFFFF) # etc...
    c_b = ArrayType(param_b & 0xFFFFFFFFFFFFFFFF, (param_b >> 64) & 0xFFFFFFFFFFFFFFFF)
    c_mod = ArrayType(param_mod & 0xFFFFFFFFFFFFFFFF, (param_mod >> 64) & 0xFFFFFFFFFFFFFFFF)
    c_res = (limb_t * (size * 2))() # Output allocation space

    # Force execution through the constant-time Comba + Montgomery safe fallback path
    crypto_lib.crypto_multiply_reduce(c_res, c_a, c_b, c_mod, size, 1)
    
    # Reassemble limbs cleanly back into standard python integers
    output_val = 0
    for i in reversed(range(size)):
        output_val = (output_val << 64) | c_res[i]
    return output_val