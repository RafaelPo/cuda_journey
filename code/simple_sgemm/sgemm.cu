#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono> // Include chrono for CPU timing

// Function to compute the ceiling of a division (used for grid size calculation)
inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

// CPU implementation of SGEMM (Single-precision General Matrix Multiply)
void sgemm_cpu(float* A, float* B, float* C, float alpha, float beta, int dim) {
    for (int i = 0; i < dim; ++i) {          // Row index of A and C
        for (int j = 0; j < dim; ++j) {      // Column index of B and C
            float sum = 0.0f;
            for (int k = 0; k < dim; ++k) {  // Common dimension
                sum += A[i * dim + k] * B[k * dim + j];
            }
            C[i * dim + j] = alpha * sum + beta * C[i * dim + j]; // Scale result and add scaled previous value
        }
    }
}

// CUDA kernel for matrix multiplication
__global__ void sgemm_kernel(float* matrix_a, float* matrix_b, float* matrix_c, float alpha, float beta, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Compute row index of output matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Compute column index of output matrix

    if (row >= dim || col >= dim) return; // Ensure threads stay within matrix bounds
    
    float tmp = 0;
    for (int i = 0; i < dim; ++i) { // Compute dot product of row and column
        tmp += (matrix_a[row * dim + i] * matrix_b[i * dim + col]);
    }
    
    matrix_c[row * dim + col] = alpha * tmp + beta * matrix_c[row * dim + col]; // Scale and accumulate
}

int main() {
    const int dim = 256; 
    const int N = dim * dim;
    const int size = N * sizeof(float); // Total size of matrices in bytes

    // Scaling factors for SGEMM operation
    float alpha = 1.3;
    float beta = 0.50;

    // Allocate host memory for matrices
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    float *h_C2 = new float[N]; // Used to store CPU computation result

    // Initialize host matrices
    for(int i = 0; i < N; i++) {
        h_A[i] = 2;
        h_B[i] = 1;
        h_C[i] = i;
        h_C2[i] = i;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice); // Ensure C is initialized on GPU

    // CUDA event timers for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time for GPU execution
    cudaEventRecord(start);

    // Define CUDA grid and block dimensions
    dim3 tpb(16, 16); // Threads per block (16x16 = 256 threads per block)
    dim3 blocks(cdiv(dim, tpb.x), cdiv(dim, tpb.y)); // Number of blocks needed

    // Launch the CUDA kernel
    sgemm_kernel<<<blocks, tpb>>>(d_A, d_B, d_C, alpha, beta, dim);

    // Record stop time for GPU execution
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Ensure kernel execution is finished before measuring time

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Compute elapsed time for GPU execution
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Copy result back from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Measure CPU execution time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    sgemm_cpu(h_A, h_B, h_C2, alpha, beta, dim);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<float, std::milli> cpu_time = stop_cpu - start_cpu;
    std::cout << "CPU execution time: " << cpu_time.count() << " ms" << std::endl;

    // Compare results between CPU and GPU computation
    std::cout << "Comparison" << std::endl;
    float diff = 0.;
    for (int i = 0; i < N; ++i) {
        diff += abs(h_C[i] - h_C2[i]); // Sum of absolute differences
    }
    std::cout << "Total diff: " << diff << std::endl;
    std::cout << "Average diff: " << diff / N << std::endl;

    // Sanity check
    for (int j=0; j < 10; ++j) {
        float expected_value_at_j = alpha * 2 * dim + beta * j;
        std::cout << h_C[j] - expected_value_at_j << std::endl;
    }
    
    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C2;

    return 0; 
}
