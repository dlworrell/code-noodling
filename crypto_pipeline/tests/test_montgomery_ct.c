#include "crypto_montgomery_ct.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#ifdef CT_TAINT_TEST
#include <valgrind/memcheck.h>
#define SECRET(p, bytes) ((void)VALGRIND_MAKE_MEM_UNDEFINED(p, bytes))
#define PUBLIC(p, bytes) ((void)VALGRIND_MAKE_MEM_DEFINED(p, bytes))
#else
#define SECRET(p, bytes) ((void)0)
#define PUBLIC(p, bytes) ((void)0)
#endif
static limb_t a[CRYPTO_MAX_LIMBS], b[CRYPTO_MAX_LIMBS], m[CRYPTO_MAX_LIMBS];
static limb_t result[CRYPTO_MAX_LIMBS+2];
static crypto_montgomery_ctx ctx;
#ifdef CT_NEGATIVE_CONTROL
static volatile unsigned control_sink;
#endif
int main(void) {
#ifdef CT_TAINT_TEST
    assert(RUNNING_ON_VALGRIND); /* Do not report a taint check without instrumentation. */
#endif
    const size_t sizes[] = {1, 2, 4, 16, 32, 64, 1024};
    for (size_t k = 0; k < sizeof(sizes)/sizeof(sizes[0]); ++k) {
        size_t n = sizes[k];
        for (size_t i = 0; i < n; ++i) m[i] = UINT64_MAX;
        assert(crypto_montgomery_ctx_init(&ctx, m, n) == 0);
        for (unsigned pattern = 0; pattern < 3; ++pattern) {
            for (size_t i = 0; i < n; ++i) {
                a[i] = pattern == 0 ? 0 : (pattern == 1 ? UINT64_MAX : UINT64_C(0xaaaaaaaaaaaaaaaa));
                b[i] = pattern == 0 ? 0 : (pattern == 1 ? UINT64_MAX : UINT64_C(0x5555555555555555));
            }
            if (pattern == 1) { a[0]--; b[0]--; } /* m-1 */
            result[0] = result[n+1] = UINT64_C(0xdeadbeefcafef00d);
            SECRET(a, n*sizeof(limb_t)); SECRET(b, n*sizeof(limb_t));
#ifdef CT_NEGATIVE_CONTROL
            /* Deliberately secret-dependent control flow: the harness must catch it. */
            for (limb_t j = 0; j < (a[0] & 255); ++j) control_sink++;
#endif
            int rc = crypto_montgomery_mul_ct(result+1, n, a, n, b, n, &ctx);
            /* Only declassify after the call; no poisoned branch is suppressed. */
            PUBLIC(a, n*sizeof(limb_t)); PUBLIC(b, n*sizeof(limb_t));
            PUBLIC(result+1, n*sizeof(limb_t));
            assert(rc == 0 && errno == 0);
            assert(result[0] == UINT64_C(0xdeadbeefcafef00d));
            assert(result[n+1] == UINT64_C(0xdeadbeefcafef00d));
            if (pattern < 2) {
                assert(result[1] == pattern); /* R==1 mod m; (m-1)^2==1 */
                for (size_t i = 1; i < n; ++i) assert(result[i+1] == 0);
            }
        }
        /* The same input pointer may supply both read-only operands. */
        assert(crypto_montgomery_mul_ct(result+1, n, a, n, a, n, &ctx) == 0);
        result[1] = 42;
        assert(crypto_montgomery_mul_ct(result+1, n-1, a, n, b, n, &ctx) == -1);
        assert(errno == EINVAL && result[1] == 42);
        assert(crypto_montgomery_mul_ct(result+1, n, a, n-1, b, n, &ctx) == -1);
        assert(result[1] == 42);
        assert(crypto_montgomery_mul_ct(result+1, n, a, n, b, n-1, &ctx) == -1);
        assert(result[1] == 42);
    }
    assert(crypto_montgomery_mul_ct(NULL, 1, a, 1, b, 1, &ctx) == -1);
    assert(crypto_montgomery_mul_ct(result, 1, NULL, 1, b, 1, &ctx) == -1);
    assert(crypto_montgomery_mul_ct(result, 1, a, 1, NULL, 1, &ctx) == -1);
    assert(crypto_montgomery_mul_ct(result, 1, a, 1, b, 1, NULL) == -1);
    assert(crypto_montgomery_ctx_init(&ctx, m, 0) == -1);
    assert(crypto_montgomery_ctx_init(&ctx, m, SIZE_MAX) == -1);
    limb_t bad = 2;
    assert(crypto_montgomery_ctx_init(&ctx, &bad, 1) == -1 && errno == EDOM);
    puts("Montgomery candidate: bounds, carry, and instrumented-path checks completed.");
    return 0;
}
