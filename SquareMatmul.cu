#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cmath>

#define BLOCK_SIZE 16

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

void invoke_kernel(float *h_A, float *h_B, float *h_C, int size)
{
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, size * sizeof(float));
    cudaMalloc((void **)&d_B, size * sizeof(float));
    cudaMalloc((void **)&d_C, size * sizeof(float));

    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    int side_length = std::sqrt(size);
    int grid_side_length = (side_length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 dimBlock(size, size);
    dim3 dimGrid(grid_side_length, grid_side_length);
    printf("%d %d\n", grid_side_length, side_length);

    matmulkernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, side_length);
    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, const char *argv[])
{

    for (int i = 1; i < 100; i++)
    {
        int size = (BLOCK_SIZE * i) * (BLOCK_SIZE * i);
        float h_A[size];
        float h_B[size];
        float h_C[size] = {0};
        int PRINT_FLAG = (int)(*argv[1] - '0');

        // Make the first and second vector and put it on the GPU
        setrand(h_A, BLOCK_SIZE * i);
        setrand(h_B, BLOCK_SIZE * i);

        auto start = std::chrono::high_resolution_clock::now();

        invoke_kernel(h_A, h_B, h_C, size);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    }
    return 0;
}