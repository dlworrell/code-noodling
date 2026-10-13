CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -fno-strict-aliasing -fno-omit-frame-pointer -fwrapv -fno-aggressive-loop-optimizations
LDFLAGS = 

# Target binaries
TARGET = test_suite

# Object files
OBJS = crypto_multiplier.o test_suite.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

crypto_multiplier.o: crypto_multiplier.c crypto_multiplier.h
	$(CC) $(CFLAGS) -c crypto_multiplier.c

test_suite.o: test_suite.c crypto_multiplier.h
	$(CC) $(CFLAGS) -c test_suite.c
	
shared: crypto_multiplier.c crypto_multiplier.h crypto_montgomery_ct.c crypto_montgomery_ct.h
	$(CC) $(CFLAGS) -fPIC -shared -o libcryptomult.so crypto_multiplier.c crypto_montgomery_ct.c
	
clean:
	rm -f *.o $(TARGET) test_bounds libcryptomult.so

.PHONY: all clean

# Dedicated bounds checks, instrumented independently of optimized objects.
bounds-sanitize: crypto_multiplier.c crypto_multiplier.h test_bounds.c
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all crypto_multiplier.c test_bounds.c -o test_bounds
	./test_bounds

.PHONY: bounds-sanitize
