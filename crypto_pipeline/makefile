# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:29ab7c65-6568-4fd6-ac92-09581239a29b

CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -fno-strict-aliasing -fno-omit-frame-pointer -fwrapv -fno-aggressive-loop-optimizations
LDFLAGS = 
# The zeta-zeros slice needs MPFR (and GMP); nothing else links them.
MPFR_LIBS = -lmpfr -lgmp -lm

# Target binaries
TARGET = test_suite

# Object files
OBJS = crypto_multiplier.o test_suite.o

all: $(TARGET) test_prime test_cert test_cert_depth test_ecpp test_mixed test_mixed_depth test_zeta zeta_verify

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

crypto_multiplier.o: crypto_multiplier.c crypto_multiplier.h
	$(CC) $(CFLAGS) -c crypto_multiplier.c

test_suite.o: tests/test_suite.c crypto_multiplier.h
	$(CC) $(CFLAGS) -I. -c tests/test_suite.c

crypto_prime.o: crypto_prime.c crypto_prime.h crypto_multiplier.h
	$(CC) $(CFLAGS) -c crypto_prime.c

test_prime: crypto_multiplier.o crypto_prime.o tests/test_prime.c
	$(CC) $(CFLAGS) -I. -o test_prime tests/test_prime.c crypto_multiplier.o crypto_prime.o $(LDFLAGS)

crypto_bigint.o: crypto_bigint.c crypto_bigint.h crypto_multiplier.h
	$(CC) $(CFLAGS) -c crypto_bigint.c

crypto_cert.o: crypto_cert.c crypto_cert.h crypto_checks_internal.h crypto_bigint.h crypto_prime.h crypto_multiplier.h
	$(CC) $(CFLAGS) -c crypto_cert.c

crypto_ecpp.o: crypto_ecpp.c crypto_ecpp.h crypto_checks_internal.h crypto_bigint.h crypto_cert.h crypto_multiplier.h
	$(CC) $(CFLAGS) -c crypto_ecpp.c

crypto_mixed.o: crypto_mixed.c crypto_mixed.h crypto_checks_internal.h crypto_bigint.h crypto_cert.h crypto_ecpp.h crypto_multiplier.h
	$(CC) $(CFLAGS) -c crypto_mixed.c

CERT_OBJS = crypto_multiplier.o crypto_prime.o crypto_bigint.o crypto_cert.o

test_cert: $(CERT_OBJS) tests/test_cert.c
	$(CC) $(CFLAGS) -I. -o test_cert tests/test_cert.c $(CERT_OBJS) $(LDFLAGS)

# The Pocklington walk's depth rule at a small limit: its own copy of crypto_cert.c.
CERT_DEPTH_FLAGS = -DCRYPTO_CERT_MAX_DEPTH=2u
CERT_DEPTH_SRCS = crypto_multiplier.c crypto_prime.c crypto_bigint.c crypto_cert.c
test_cert_depth: $(CERT_DEPTH_SRCS) crypto_cert.h crypto_checks_internal.h tests/test_cert_depth.c
	$(CC) $(CFLAGS) $(CERT_DEPTH_FLAGS) -I. -o test_cert_depth tests/test_cert_depth.c $(CERT_DEPTH_SRCS) $(LDFLAGS)

test_ecpp: $(CERT_OBJS) crypto_ecpp.o tests/test_ecpp.c
	$(CC) $(CFLAGS) -I. -o test_ecpp tests/test_ecpp.c $(CERT_OBJS) crypto_ecpp.o $(LDFLAGS)

test_mixed: $(CERT_OBJS) crypto_ecpp.o crypto_mixed.o tests/test_mixed.c
	$(CC) $(CFLAGS) -I. -o test_mixed tests/test_mixed.c $(CERT_OBJS) crypto_ecpp.o crypto_mixed.o $(LDFLAGS)

# The walk's depth rule at a small limit: its own copy of crypto_mixed.c.
DEPTH_FLAGS = -DCRYPTO_MIXED_MAX_DEPTH=2u
test_mixed_depth: $(CERT_OBJS) crypto_ecpp.o crypto_mixed.c crypto_mixed.h crypto_checks_internal.h tests/test_mixed_depth.c
	$(CC) $(CFLAGS) $(DEPTH_FLAGS) -I. -o test_mixed_depth tests/test_mixed_depth.c crypto_mixed.c $(CERT_OBJS) crypto_ecpp.o $(LDFLAGS)

crypto_zeta_ball.o: crypto_zeta_ball.c crypto_zeta_ball.h
	$(CC) $(CFLAGS) -c crypto_zeta_ball.c

crypto_zeta.o: crypto_zeta.c crypto_zeta.h crypto_zeta_ball.h
	$(CC) $(CFLAGS) -c crypto_zeta.c

ZETA_OBJS = crypto_zeta_ball.o crypto_zeta.o

test_zeta: $(ZETA_OBJS) tests/test_zeta.c
	$(CC) $(CFLAGS) -I. -o test_zeta tests/test_zeta.c $(ZETA_OBJS) $(LDFLAGS) $(MPFR_LIBS)

zeta_verify: $(ZETA_OBJS) zeta_verify.c
	$(CC) $(CFLAGS) -I. -o zeta_verify zeta_verify.c $(ZETA_OBJS) $(LDFLAGS) $(MPFR_LIBS)
	
shared: crypto_multiplier.c crypto_multiplier.h crypto_montgomery_ct.c crypto_montgomery_ct.h crypto_prime.c crypto_prime.h crypto_bigint.c crypto_bigint.h crypto_cert.c crypto_cert.h crypto_ecpp.c crypto_ecpp.h crypto_mixed.c crypto_mixed.h crypto_checks_internal.h
	$(CC) $(CFLAGS) -fPIC -shared -o libcryptomult.so crypto_multiplier.c crypto_montgomery_ct.c crypto_prime.c crypto_bigint.c crypto_cert.c crypto_ecpp.c crypto_mixed.c
	
clean:
	rm -f *.o $(TARGET) test_prime test_cert test_cert_depth test_ecpp test_mixed test_mixed_depth test_zeta zeta_verify test_bounds test_prime_sanitize test_cert_sanitize test_cert_depth_sanitize test_ecpp_sanitize test_mixed_sanitize test_mixed_depth_sanitize test_zeta_sanitize libcryptomult.so

.PHONY: all shared clean

# Dedicated bounds checks, instrumented independently of optimized objects.
bounds-sanitize: crypto_multiplier.c crypto_multiplier.h tests/test_bounds.c
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -I. crypto_multiplier.c tests/test_bounds.c -o test_bounds
	./test_bounds
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -I. crypto_multiplier.c crypto_prime.c tests/test_prime.c -o test_prime_sanitize
	./test_prime_sanitize
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -I. crypto_multiplier.c crypto_prime.c crypto_bigint.c crypto_cert.c tests/test_cert.c -o test_cert_sanitize
	./test_cert_sanitize
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all $(CERT_DEPTH_FLAGS) -I. $(CERT_DEPTH_SRCS) tests/test_cert_depth.c -o test_cert_depth_sanitize
	./test_cert_depth_sanitize
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -I. crypto_multiplier.c crypto_prime.c crypto_bigint.c crypto_cert.c crypto_ecpp.c tests/test_ecpp.c -o test_ecpp_sanitize
	./test_ecpp_sanitize
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -I. crypto_multiplier.c crypto_prime.c crypto_bigint.c crypto_cert.c crypto_ecpp.c crypto_mixed.c tests/test_mixed.c -o test_mixed_sanitize
	./test_mixed_sanitize
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all $(DEPTH_FLAGS) -I. crypto_multiplier.c crypto_prime.c crypto_bigint.c crypto_cert.c crypto_ecpp.c crypto_mixed.c tests/test_mixed_depth.c -o test_mixed_depth_sanitize
	./test_mixed_depth_sanitize
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -I. crypto_zeta_ball.c crypto_zeta.c tests/test_zeta.c -o test_zeta_sanitize $(MPFR_LIBS)
	./test_zeta_sanitize

.PHONY: bounds-sanitize
