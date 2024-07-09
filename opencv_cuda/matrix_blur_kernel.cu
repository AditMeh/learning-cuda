#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void matrix_blur_kernel(unsigned char *img, unsigned char *out, int h, int w, int window_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;
    
    // (i * frame.cols + j) * frame.channels() + k
    if (row < h && col < w)
    {
        int accum = 0;
        int pixels = 0;
        for (int i = -window_size; i <= window_size; i++)
        {
            for (int j = -window_size; j <= window_size; j++)
            {
                int curr_row = row + i;
                int curr_col = col + j;

                if (curr_row >= 0 && curr_row < h && curr_col >= 0 && curr_col < w && channel < 3)
                {
                    accum += img[(curr_row * w + curr_col) * 3 + channel];
                    pixels += 1;
                }
            }
        }

        out[(row * w + col) * 3 + channel] = (unsigned char)(accum / pixels);
    }
}

float invoke_kernel(unsigned char *img, int h, int w, int window_size, int size)
{
    
    unsigned char *d_orig; 
    unsigned char *d_blur;

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms = 0;

    cudaMalloc((void **)&d_orig, size * sizeof(u_char));
    cudaMalloc((void **)&d_blur, size * sizeof(u_char));

    cudaMemcpy(d_orig, img, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 3);
    dim3 dimGrid((h + BLOCK_SIZE - 1) / BLOCK_SIZE, (w + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);


    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrix_blur_kernel<<<dimGrid, dimBlock>>>(d_orig, d_blur, h, w, window_size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(img, d_blur, size * sizeof(u_char), cudaMemcpyDeviceToHost);

    cudaFree(d_orig);
    cudaFree(d_blur);
    return ms;
}

