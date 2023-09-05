#include <stdio.h>

__global__ void helloGPU(){
    int rank = threadIdx.x;
    printf("Hello from thread with rank %d\n", rank);
}

int main() {
    dim3 dimBlock(10);
    dim3 dimGrid(1);

    //launch
    helloGPU<<<dimGrid, dimBlock>>>();
    cudaDeviceSynchronize();
    return 0;
}