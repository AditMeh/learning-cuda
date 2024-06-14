#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <limits>

#define BLOCK_SIZE 32
#define MAX_RUNS 2

__global__ void matmulkernel(float *d_A, float *d_B, float *result, int WIDTH)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < WIDTH) && (col < WIDTH))
    {
        float ret = 0;
        for (int i = 0; i < WIDTH; i++)
        {
            ret += d_A[row * WIDTH + i] * d_B[i * WIDTH + col];
        }
        result[row * WIDTH + col] = ret;
    }
}

void setrand(float *arr, int len)
{
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            arr[i * len + j] = float(rand() % 9);
        }
    }
}

float invoke_kernel(float *h_A, float *h_B, float *h_C, int size)
{
    float *d_A, *d_B, *d_C;
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms = 0;

    cudaMalloc((void **)&d_A, size * sizeof(float));
    cudaMalloc((void **)&d_B, size * sizeof(float));
    cudaMalloc((void **)&d_C, size * sizeof(float));

    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    int side_length = std::sqrt(size);
    int grid_side_length = (side_length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(grid_side_length, grid_side_length);

    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmulkernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, side_length);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return ms;
}

int main(int argc, const char *argv[])
{
    for (int i = 1; i < 25; i++)
    {
        float avg_runtime = 0.0;
        float min_runtime = std::numeric_limits<float>::max();

        int size = (BLOCK_SIZE * i) * (BLOCK_SIZE * i);

        int side_length = std::sqrt(size);
        int grid_side_length = (side_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
        printf("Threads launched: %d, Matrix Dim: (%d, %d)\n", (int)(pow(grid_side_length, 2) * pow(BLOCK_SIZE, 2)), side_length, side_length);

        for (int run = 0; run < MAX_RUNS; run++)
        {
            float h_A[size];
            float h_B[size];
            float h_C[size] = {0};
            // int PRINT_FLAG = (int)(*argv[1] - '0');

            // Make the first and second vector and put it on the GPU
            setrand(h_A, BLOCK_SIZE * i);
            setrand(h_B, BLOCK_SIZE * i);

            cudaDeviceSynchronize();

            float ms = invoke_kernel(h_A, h_B, h_C, size);

            avg_runtime += (float)ms;
            if (min_runtime > ms)
            {
                min_runtime = ms;
            }
        }

        std::cout << "Avg Execution time: " << avg_runtime / (MAX_RUNS - 1.0) << " ms" << std::endl;
        std::cout << "Min Execution time: " << min_runtime << " ms" << std::endl;

    }
    return 0;
}