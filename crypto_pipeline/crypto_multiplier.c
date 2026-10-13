#if !defined(_WIN32) && !defined(_WIN64)
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif
#include "crypto_multiplier.h"
#include <errno.h>
#include <string.h>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <bcrypt.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static int valid_limb_count(size_t n) { return n > 0 && n <= CRYPTO_MAX_LIMBS; }
static void wipe(void *p, size_t bytes) {
    volatile unsigned char *q = p;
    while (bytes--) *q++ = 0;
}
static int compare(const limb_t *a, const limb_t *b, size_t n) {
    for (size_t i = n; i-- > 0;) {
        if (a[i] != b[i]) return a[i] > b[i] ? 1 : -1;
    }
    return 0;
}
static int is_zero(const limb_t *a, size_t n) {
    limb_t bits = 0;
    for (size_t i = 0; i < n; ++i) bits |= a[i];
    return bits == 0;
}
static int is_one(const limb_t *a, size_t n) {
    if (a[0] != 1) return 0;
    for (size_t i = 1; i < n; ++i) if (a[i]) return 0;
    return 1;
}
static int valid_modulus(const limb_t *m, size_t n) {
    return !is_zero(m, n) && !is_one(m, n);
}
/* Returns borrow; supports r==a. Widen BEFORE adding borrow to b[i]. */
static limb_t subtract(limb_t *r, const limb_t *a, const limb_t *b, size_t n) {
    limb_t borrow = 0;
    for (size_t i = 0; i < n; ++i) {
        dlimb_t sub = (dlimb_t)b[i] + borrow;
        limb_t next = (dlimb_t)a[i] < sub;
        r[i] = (limb_t)((dlimb_t)a[i] - sub);
        borrow = next;
    }
    return borrow;
}
/* Value high*R+r is known to be <2*m. Retain the overflow bit. */
static void final_subtract(limb_t *r, const limb_t *m, size_t n, limb_t high) {
    limb_t temp[n];
    limb_t borrow = subtract(temp, r, m, n);
    limb_t mask = (limb_t)0 - (limb_t)((high != 0) | (borrow == 0));
    for (size_t i = 0; i < n; ++i) r[i] = (mask & temp[i]) | (~mask & r[i]);
}
void ct_modular_reduce_final(limb_t *r, const limb_t *mod, size_t size) {
    if (!r || !mod || !valid_limb_count(size)) { errno = EINVAL; return; }
    if (!valid_modulus(mod, size)) { errno = EDOM; return; }
    final_subtract(r, mod, size, 0);
    errno = 0;
}
limb_t compute_montgomery_inverse(limb_t mod0) {
    if (!(mod0 & 1)) { errno = EDOM; return 0; }
    limb_t inv = 1;
    /* Each Newton iteration doubles the correct low bits: 1 -> 64. */
    for (unsigned i = 0; i < 6; ++i) inv *= 2 - mod0 * inv;
    errno = 0;
    return (limb_t)0 - inv;
}

void secure_comba_multiply(limb_t *result, const limb_t *a, const limb_t *b, size_t size) {
    if (!result || !a || !b || !valid_limb_count(size)) {
        errno = EINVAL;
        return;
    }
    limb_t c0 = 0, c1 = 0, c2 = 0;
    memset(result, 0, 2 * size * sizeof(limb_t));

    for (size_t i = 0; i < 2 * size - 1; i++) {
        size_t start = (i >= size) ? (i - size + 1) : 0;
        size_t end = (i < size) ? i : (size - 1);

        for (size_t j = start; j <= end; j++) {
            size_t k = i - j;
            dlimb_t prod = (dlimb_t)a[j] * b[k];

            dlimb_t sum = (dlimb_t)c0 + (prod & 0xFFFFFFFFFFFFFFFFULL);
            c0 = (limb_t)sum;
            
            sum = (dlimb_t)c1 + (prod >> 64) + (sum >> 64);
            c1 = (limb_t)sum;
            
            c2 += (limb_t)(sum >> 64);
        }

        result[i] = c0;
        c0 = c1;
        c1 = c2;
        c2 = 0;
    }
    result[2 * size - 1] = c0;
    errno = 0;
}


/* REDC(T), with R=2^(64*n). Contract: odd m>1 and 0<=T<m*R. */
void secure_montgomery_reduction(limb_t *product, const limb_t *mod, size_t size, limb_t inv) {
    if (!product || !mod || !valid_limb_count(size)) { errno = EINVAL; return; }
    if (!(mod[0] & 1) || !valid_modulus(mod, size) ||
        mod[0] * inv != UINT64_MAX || compare(product + size, mod, size) >= 0) {
        errno = EDOM; return;
    }
    limb_t high = 0;
    for (size_t i = 0; i < size; ++i) {
        limb_t u = product[i] * inv;
        dlimb_t carry = 0;
        for (size_t j = 0; j < size; ++j) {
            dlimb_t sum = (dlimb_t)u * mod[j] + product[i+j] + carry;
            product[i+j] = (limb_t)sum;
            carry = sum >> 64;
        }
        /* Fixed loop bounds; carry beyond the allocation is kept separately. */
        for (size_t k = i + size; k < 2 * size; ++k) {
            dlimb_t sum = (dlimb_t)product[k] + carry;
            product[k] = (limb_t)sum;
            carry = sum >> 64;
        }
        high += (limb_t)carry;
    }
    memmove(product, product + size, size * sizeof(limb_t));
    memset(product + size, 0, size * sizeof(limb_t));
    final_subtract(product, mod, size, high);
    errno = 0;
}

/* Sum-form Karatsuba: keep h+1 limbs in BOTH midpoint sums. */
static void karatsuba_recursive(limb_t *r, const limb_t *a, const limb_t *b, size_t n) {
    if (n <= 16) { secure_comba_multiply(r, a, b, n); return; }
    if (n & 1) {
        /* Pad one zero limb; retain recursive Karatsuba for odd lengths. */
        limb_t pa[n+1], pb[n+1], pr[2*(n+1)];
        memcpy(pa, a, n * sizeof(limb_t));
        memcpy(pb, b, n * sizeof(limb_t));
        pa[n] = 0; pb[n] = 0;
        karatsuba_recursive(pr, pa, pb, n+1);
        memcpy(r, pr, 2*n * sizeof(limb_t));
        return;
    }
    size_t h = n / 2, k = h + 1;
    limb_t sa[k], sb[k], cross[2*k];
    limb_t ca = 0, cb = 0;
    for (size_t i = 0; i < h; ++i) {
        dlimb_t x = (dlimb_t)a[i] + a[h+i] + ca;
        dlimb_t y = (dlimb_t)b[i] + b[h+i] + cb;
        sa[i] = (limb_t)x; ca = (limb_t)(x >> 64);
        sb[i] = (limb_t)y; cb = (limb_t)(y >> 64);
    }
    sa[h] = ca; sb[h] = cb;
    karatsuba_recursive(r, a, b, h);
    karatsuba_recursive(r+n, a+h, b+h, h);
    karatsuba_recursive(cross, sa, sb, k);
    /* Two separate subtractions preserve borrows of up to two in the sum. */
    for (size_t part = 0; part < 2; ++part) {
        limb_t borrow = 0;
        for (size_t i = 0; i < 2*k; ++i) {
            dlimb_t sub = (dlimb_t)(i < n ? r[part*n+i] : 0) + borrow;
            limb_t next = (dlimb_t)cross[i] < sub;
            cross[i] = (limb_t)((dlimb_t)cross[i] - sub);
            borrow = next;
        }
    }
    limb_t carry = 0;
    for (size_t i = 0; i < 2*n-h; ++i) {
        dlimb_t sum = (dlimb_t)r[h+i] + (i < 2*k ? cross[i] : 0) + carry;
        r[h+i] = (limb_t)sum;
        carry = (limb_t)(sum >> 64);
    }
}
void karatsuba_multiply(limb_t *result, const limb_t *a, const limb_t *b, size_t size) {
    if (!result || !a || !b || !valid_limb_count(size)) { errno = EINVAL; return; }
    karatsuba_recursive(result, a, b, size);
    errno = 0;
}
static uint32_t mod_power(uint32_t a, uint32_t e, uint32_t p) {
    uint64_t value = 1, base = a;
    while (e) {
        if (e & 1) value = value * base % p;
        base = base * base % p;
        e >>= 1;
    }
    return (uint32_t)value;
}
/* Iterative radix-2 NTT. Both primes admit primitive root 3 and length<=8192. */
static void transform(uint32_t *a, size_t length, uint32_t p, int inverse) {
    for (size_t i = 1, j = 0; i < length; ++i) {
        size_t bit = length >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) { uint32_t temp = a[i]; a[i] = a[j]; a[j] = temp; }
    }
    for (size_t width = 2; width <= length; width <<= 1) {
        uint32_t root = mod_power(3, (p-1)/(uint32_t)width, p);
        if (inverse) root = mod_power(root, p-2, p);
        for (size_t i = 0; i < length; i += width) {
            uint64_t w = 1;
            for (size_t j = 0; j < width/2; ++j) {
                uint32_t u = a[i+j];
                uint32_t v = (uint32_t)(w * a[i+j+width/2] % p);
                a[i+j] = u+v >= p ? u+v-p : u+v;
                a[i+j+width/2] = u >= v ? u-v : u+p-v;
                w = w * root % p;
            }
        }
    }
    if (inverse) {
        uint32_t inv_length = mod_power((uint32_t)length, p-2, p);
        for (size_t i = 0; i < length; ++i) a[i] = (uint32_t)((uint64_t)a[i] * inv_length % p);
    }
}
/* Base-2^16 convolution, exact CRT reconstruction, then carry propagation.
 * Max coefficient: 4096*(65535^2) < p1*p2, so CRT is unambiguous.
 * This is a practical bounded NTT, not the Harvey-van der Hoeven algorithm. */
void ntt_polynomial_multiply(limb_t *result, const limb_t *a, const limb_t *b, size_t n) {
    if (!result || !a || !b || !valid_limb_count(n)) { errno = EINVAL; return; }
    const uint32_t primes[2] = {998244353u, 1004535809u};
    size_t digits = 4*n, length = 1;
    while (length < 2*digits) length <<= 1;
    uint32_t fa[length], fb[length], first[2*digits];
    uint32_t inv_p1 = mod_power(primes[0], primes[1]-2, primes[1]);
    memset(result, 0, 2*n * sizeof(limb_t));
    for (size_t pass = 0; pass < 2; ++pass) {
        uint32_t p = primes[pass];
        for (size_t i = 0; i < length; ++i) {
            fa[i] = i < digits ? (uint32_t)((a[i/4] >> (16*(i%4))) & 65535u) : 0;
            fb[i] = i < digits ? (uint32_t)((b[i/4] >> (16*(i%4))) & 65535u) : 0;
        }
        transform(fa, length, p, 0); transform(fb, length, p, 0);
        for (size_t i = 0; i < length; ++i) fa[i] = (uint32_t)((uint64_t)fa[i] * fb[i] % p);
        transform(fa, length, p, 1);
        if (pass == 0) {
            memcpy(first, fa, 2*digits * sizeof(uint32_t));
        } else {
            uint64_t carry = 0;
            for (size_t i = 0; i < 2*digits; ++i) {
                uint64_t delta = ((uint64_t)fa[i] + p - first[i]) % p;
                uint64_t coefficient = first[i] + (uint64_t)primes[0] * (delta * inv_p1 % p);
                uint64_t value = coefficient + carry;
                result[i/4] |= (limb_t)(value & 65535u) << (16*(i%4));
                carry = value >> 16;
            }
        }
    }
    errno = 0;
}
void crypto_multiply_reduce(limb_t *result, const limb_t *a, const limb_t *b,
                            const limb_t *mod, size_t size, int is_constant_time) {
    if (!result || !a || !b || !mod || !valid_limb_count(size)) { errno = EINVAL; return; }
    if (!(mod[0] & 1) || !valid_modulus(mod, size) ||
        compare(a, mod, size) >= 0 || compare(b, mod, size) >= 0) { errno = EDOM; return; }
    limb_t inv = compute_montgomery_inverse(mod[0]);
    if (is_constant_time || size < 64) secure_comba_multiply(result, a, b, size);
    else karatsuba_multiply(result, a, b, size);
    /* All dispatch sizes have the same Montgomery semantics. */
    secure_montgomery_reduction(result, mod, size, inv);
}

/* Binary long division. Invariant: r<m; after shifting r<2*m. */
static void limb_remainder(limb_t *r, const limb_t *a, size_t an, const limb_t *m, size_t n) {
    memset(r, 0, n * sizeof(limb_t));
    for (size_t bit = an * 64; bit-- > 0;) {
        limb_t carry = (a[bit / 64] >> (bit % 64)) & 1;
        for (size_t j = 0; j < n; ++j) {
            limb_t next = r[j] >> 63;
            r[j] = (r[j] << 1) | carry;
            carry = next;
        }
        final_subtract(r, m, n, carry);
    }
}
int crypto_mod_multiply(limb_t *result, const limb_t *a, const limb_t *b,
                        const limb_t *mod, size_t n) {
    if (!result || !a || !b || !mod || !valid_limb_count(n)) { errno = EINVAL; return -1; }
    if (!valid_modulus(mod, n)) { errno = EDOM; return -1; }
    limb_t product[2*n];
    karatsuba_recursive(product, a, b, n);
    limb_remainder(result, product, 2*n, mod, n);
    wipe(product, sizeof(product));
    errno = 0; return 0;
}
static void shift_right(limb_t *a, size_t n, limb_t high) {
    for (size_t i = n; i-- > 0;) {
        limb_t next = a[i] & 1;
        a[i] = (a[i] >> 1) | (high << 63);
        high = next;
    }
}
/* For odd m, half of odd x is (x+m)/2. Retain the addition overflow. */
static void half_mod(limb_t *x, const limb_t *m, size_t n) {
    limb_t carry = 0;
    if (x[0] & 1) {
        for (size_t i = 0; i < n; ++i) {
            dlimb_t sum = (dlimb_t)x[i] + m[i] + carry;
            x[i] = (limb_t)sum; carry = (limb_t)(sum >> 64);
        }
    }
    shift_right(x, n, carry);
}
static void subtract_mod(limb_t *x, const limb_t *y, const limb_t *m, size_t n) {
    if (subtract(x, x, y, n)) {
        limb_t carry = 0;
        for (size_t i = 0; i < n; ++i) {
            dlimb_t sum = (dlimb_t)x[i] + m[i] + carry;
            x[i] = (limb_t)sum; carry = (limb_t)(sum >> 64);
        }
    }
}
/* Variable-time binary extended GCD. u=input*x1 (mod m), v=input*x2 (mod m). */
int crypto_mod_inverse(limb_t *result, const limb_t *input, const limb_t *mod, size_t n) {
    if (!result || !input || !mod || !valid_limb_count(n)) { errno = EINVAL; return -1; }
    if (!(mod[0] & 1) || !valid_modulus(mod, n)) { errno = EDOM; return -1; }
    limb_t u[n], v[n], x1[n], x2[n];
    limb_remainder(u, input, n, mod, n);
    memcpy(v, mod, n * sizeof(limb_t));
    memset(x1, 0, n * sizeof(limb_t));
    memset(x2, 0, n * sizeof(limb_t));
    x1[0] = 1;
    int ok = 0;
    while (!is_zero(u, n) && !is_zero(v, n)) {
        if (is_one(u, n) || is_one(v, n)) { ok = 1; break; }
        while (!(u[0] & 1)) { shift_right(u, n, 0); half_mod(x1, mod, n); }
        while (!(v[0] & 1)) { shift_right(v, n, 0); half_mod(x2, mod, n); }
        if (compare(u, v, n) >= 0) { subtract(u, u, v, n); subtract_mod(x1, x2, mod, n); }
        else { subtract(v, v, u, n); subtract_mod(x2, x1, mod, n); }
    }
    /* One may have become 1 immediately before the other became zero. */
    if (is_one(u, n) || is_one(v, n)) ok = 1;
    if (ok) memcpy(result, is_one(u, n) ? x1 : x2, n * sizeof(limb_t));
    wipe(u, sizeof(u)); wipe(v, sizeof(v)); wipe(x1, sizeof(x1)); wipe(x2, sizeof(x2));
    errno = ok ? 0 : EDOM; return ok ? 0 : -1;
}
/* Deprecated compatibility name: this routine is NOT constant-time. */
void ct_mod_inverse_multiprecision(limb_t *r, const limb_t *a, const limb_t *m, size_t n) {
    (void)crypto_mod_inverse(r, a, m, n);
}
int secure_get_random_bytes(uint8_t *buffer, size_t length) {
    if (!buffer || length > CRYPTO_MAX_LIMBS * sizeof(limb_t)) { errno = EINVAL; return -1; }
#if defined(_WIN32) || defined(_WIN64)
    if (BCryptGenRandom(NULL, buffer, (ULONG)length, BCRYPT_USE_SYSTEM_PREFERRED_RANDOM) != 0) {
        errno = EIO; return -1;
    }
#else
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    size_t done = 0;
    while (done < length) {
        ssize_t got = read(fd, buffer + done, length - done);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) { int err = got < 0 ? errno : EIO; close(fd); errno = err; return -1; }
        done += (size_t)got;
    }
    close(fd);
#endif
    errno = 0; return 0;
}
/* Return both a*k mod m and k; uniform rejection sampling among units mod m. */
int crypto_blind(limb_t *blinded, limb_t *factor, const limb_t *a, const limb_t *m, size_t n) {
    if (!blinded || !factor || !a || !m || !valid_limb_count(n)) { errno = EINVAL; return -1; }
    if (!(m[0] & 1) || !valid_modulus(m, n)) { errno = EDOM; return -1; }
    limb_t k[n], inverse[n], value[n];
    size_t used = n;
    while (used > 1 && m[used-1] == 0) --used;
    limb_t mask = m[used-1];
    for (unsigned shift = 1; shift < 64; shift *= 2) mask |= mask >> shift;
    int status = -1, error = EAGAIN;
    memset(k, 0, n * sizeof(limb_t));
    for (unsigned attempt = 0; attempt < 128; ++attempt) {
        if (secure_get_random_bytes((uint8_t *)k, used * sizeof(limb_t))) { error = errno; break; }
        k[used-1] &= mask;
        if (is_zero(k, n) || compare(k, m, n) >= 0) continue;
        if (crypto_mod_inverse(inverse, k, m, n)) continue;
        (void)crypto_mod_multiply(value, a, k, m, n);
        memcpy(blinded, value, n * sizeof(limb_t));
        memcpy(factor, k, n * sizeof(limb_t));
        status = 0; error = 0; break;
    }
    wipe(k, sizeof(k)); wipe(inverse, sizeof(inverse)); wipe(value, sizeof(value));
    errno = error; return status;
}
/* The old ABI cannot return k or specify m. Fail instead of discarding k. */
int execute_blinded_multiplication(limb_t *out, const limb_t *a, size_t n) {
    if (!out || !a || !valid_limb_count(n)) { errno = EINVAL; return -1; }
    errno = ENOTSUP; return -1;
}
void execute_unblinding(limb_t *out, const limb_t *blinded, const limb_t *k,
                        const limb_t *m, size_t n) {
    if (!out || !blinded || !k || !m || !valid_limb_count(n)) { errno = EINVAL; return; }
    limb_t inverse[n];
    if (crypto_mod_inverse(inverse, k, m, n)) { wipe(inverse, sizeof(inverse)); return; }
    (void)crypto_mod_multiply(out, blinded, inverse, m, n);
    wipe(inverse, sizeof(inverse));
}
