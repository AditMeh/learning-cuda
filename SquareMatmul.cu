#include <stdio.h>
#include <iostream>
#include <thread>
#include <chrono>

#define BLOCK_SIZE 4
#define GRID_SIZE 1

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
        // printf("(%i, %i) %f\n", row, col, ret);
    }
}

void printvec(float *arr, int len)
{
    for (int j = 0; j < len; j++)
    {
        printf("[");
        for (int i = 0; i < len; i++)
        {
            printf("%f ", arr[j * len + i]);
        }
        printf("]");

        printf("\n");
    }
    printf("\n");
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

int main(int argc, const char *argv[])
{
    float h_A[(BLOCK_SIZE * 1) * (BLOCK_SIZE * 1)];       // Row major
    float h_B[(BLOCK_SIZE * 1) * (BLOCK_SIZE * 1)];       // Row major
    float h_C[(BLOCK_SIZE * 1) * (BLOCK_SIZE * 1)] = {0}; // Row major

    // Make the first and second vector and put it on the GPU
    setrand(h_A, BLOCK_SIZE * 1);
    setrand(h_B, BLOCK_SIZE * 1);

    int PRINT_FLAG = (int)(*argv[1] - '0');

    if (PRINT_FLAG)
    {
        printvec(h_A, BLOCK_SIZE * 1);
        printvec(h_B, BLOCK_SIZE * 1);
    }
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, (BLOCK_SIZE * 1) * (BLOCK_SIZE * 1) * sizeof(float));
    cudaMalloc((void **)&d_B, (BLOCK_SIZE * 1) * (BLOCK_SIZE * 1) * sizeof(float));
    cudaMalloc((void **)&d_C, (BLOCK_SIZE * 1) * (BLOCK_SIZE * 1) * sizeof(float));

    cudaMemcpy(d_A, h_A, (BLOCK_SIZE * 1) * (BLOCK_SIZE * 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, (BLOCK_SIZE * 1) * (BLOCK_SIZE * 1) * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock((BLOCK_SIZE * 1), (BLOCK_SIZE * 1));
    dim3 dimGrid(1, 1);

    matmulkernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, (BLOCK_SIZE * 1));
    cudaMemcpy(h_C, d_C, (BLOCK_SIZE * 1) * (BLOCK_SIZE * 1) * sizeof(float), cudaMemcpyDeviceToHost);
        

    if (PRINT_FLAG)
    {
        printvec(h_C, BLOCK_SIZE * 1);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}