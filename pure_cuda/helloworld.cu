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

    // CPU basically just dips after the kernel is run,
    // if we don't syncronize and wait for kernel to complete, there is no guarantee
    // the printf buffer on the device will be sent to standard output
    
    // Sources: 
    // https://stackoverflow.com/questions/58531349/cuda-kernel-printf-produces-no-output-in-terminal-works-in-profiler
    // https://stackoverflow.com/questions/19193468/why-do-we-need-cudadevicesynchronize-in-kernels-with-device-printf
    
    cudaDeviceSynchronize();
    return 0;
}