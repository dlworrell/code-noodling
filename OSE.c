//Optimized Sieve of Eratosthenes in C with 
//Wheel Factorization, Segmented Sieving, and
//Parallelism:

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <pthread.h>

#define WHEEL_SIZE 2
#define WHEEL_FIRST_PRIME 3
#define WHEEL_PRIMES {2,3}

typedef struct {
    int start;
    int end;
    unsigned char *primes;
    int prime_count;
} segment_t;

typedef struct {
    segment_t *segments;
    int num_segments;
} thread_data_t;

void mark_primes(segment_t *segment, int *small_primes, int num_small_primes) {
    int segment_size = segment->end - segment->start + 1;
    int num_bits = (segment_size - 1) / WHEEL_SIZE + 1;
    unsigned char *primes = segment->primes;
    memset(primes, 0xFF, sizeof(unsigned char) * num_bits);
    // Mark 1 as composite
    primes[0] &= 0xFE;
    // Sieve with small primes up to sqrt(end)
    int sqrt_end = (int) sqrt(segment->end);
    for (int i = 0; i < num_small_primes; i++) {
        int p = small_primes[i];
        if (p > sqrt_end) {
            break;
        }
        int p_squared = p * p;
        int q = (segment->start + p - 1) / p * p;
        if (q < p_squared) {
            q = p_squared;
        }
        int j = (q - segment->start) / WHEEL_SIZE * WHEEL_SIZE - segment->start;
        for (; j < segment_size; j += p * WHEEL_SIZE) {
            primes[j / WHEEL_SIZE] &= ~(1 << (j / WHEEL_SIZE % 8));
        }
    }
    // Sieve with primes in wheel
    int wheel_skip[WHEEL_SIZE] = {4, 2};
    int j = 0;
    for (int i = WHEEL_FIRST_PRIME; i <= sqrt_end; i += wheel_skip[j++ % WHEEL_SIZE]) {
        if (primes[(i - segment->start) / WHEEL_SIZE]) {
            int p_squared = i * i;
            int j = (p_squared - segment->start) / WHEEL_SIZE * WHEEL_SIZE - segment->start;
            for (; j < segment_size; j += i * WHEEL_SIZE) {
                primes[j / WHEEL_SIZE] &= ~(1 << (j / WHEEL_SIZE % 8));
            }
        }
    }
    // Count primes in segment
    int count = 0;
    for (int i = 0; i < num_bits; i++) {
        count += __builtin_popcount(primes[i]);
    }
    segment->prime_count = count;
}

void *mark_primes_thread(void *arg) {
    thread_data_t *data = (thread_data_t *) arg;
    for (int i = 0; i < data->num_segments; i++) {
        mark_primes(&data->segments[i], WHEEL_PRIMES, sizeof(WHEEL_PRIMES) / sizeof(int));
    }
    return NULL;
}

void print_primes(segment_t *segment) {
    int segment_size = segment->end - segment->start + 1;
    int num_bits = (segment_size - 1) / W
