#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <limits>

#define BLOCK_SIZE 32
#define MAX_RUNS 2

__global__ void Tiled_MatmulKernel(float *d_A, float *d_B, float *result, int WIDTH)
{

    // Shared within-block memory
    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];

    // This thread needs to compute the value at (row,col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Final value that this thread needs to compute
    float Pvalue = 0;

    // Matrix for this element involves dot product between the row'th row of d_A, and the col'th col of d_B
    // WE compute the dot products little by little for each BLOCK_SIZE chunk and sum em together in Pvalue
    // Block-by-block iterator.
    for (int ph = 0; ph < WIDTH / BLOCK_SIZE; ++ph)
    {

        // We're looking at the ph'th chunk of the row'th col of d_A
        // 1. row*WIDTH takes us to the start of the row'th row.
        // 2. ph*BLOCK_SIZE takes us to the start of the current chunk we're looking at
        // 3. threadIdx.x moves up to the current thread in the current block
        Mds[threadIdx.y][threadIdx.x] = d_A[row * WIDTH + ph * BLOCK_SIZE + threadIdx.x];

        // Once this is done Mds will hold all of all elements of M within this current block

        // We're looking at the ph'th chunk of the col'th row of d_A
        // 1. ph*BLOCK_SIZE takes us to the start of the current col chunk we're looking at, is a row index
        // 2. threadIdx.y offsets to where the current thread is within the block
        // 3. * WIDTH takes us to the start of the row at which the column element we need to access it
        // 4. + col then offsets within the row to go to the column we're operating on.
        Nds[threadIdx.y][threadIdx.x] = d_B[(ph * BLOCK_SIZE + threadIdx.y) * WIDTH + col];

        // Once this is done Nds will hold all of all elements of M within this current block

        __syncthreads();

        // Now do within-block matrix multiplication for the current thread
        // Aggregate and store in Pvalue
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        }
        __syncthreads();
    }
    result[row * WIDTH + col] = Pvalue;
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

    Tiled_MatmulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, side_length);

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