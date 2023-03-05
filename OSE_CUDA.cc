//CUDA Sieve 

#include <iostream>
#include <cuda.h>

#define BLOCK_SIZE 1024
#define MAX_PRIMES 1000000

__global__ void sieve(int* is_prime, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (idx > n) return;
    if (is_prime[idx]) {
        for (int i = idx * idx; i <= n; i += idx) {
            is_prime[i] = 0;
        }
    }
}

int main() {
    int n = MAX_PRIMES;
    int* is_prime_host = new int[n + 1];
    for (int i = 0; i <= n; i++) {
        is_prime_host[i] = 1;
    }

    int* is_prime_device;
    cudaMalloc(&is_prime_device, (n + 1) * sizeof(int));
    cudaMemcpy(is_prime_device, is_prime_host, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sieve<<<num_blocks, BLOCK_SIZE>>>(is_prime_device, n);

    cudaMemcpy(is_prime_host, is_prime_device, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 2; i <= n; i++) {
        if (is_prime_host[i]) {
            std::cout << i << " ";
        }
    }

    std::cout << std::endl;

    delete[] is_prime_host;
    cudaFree(is_prime_device);

    return 0;
}
