#include <stdio.h>

__global__ void matmulkernel(float*vec1, float* vec2, float *result){
    int i = threadIdx.x;
    result[i] = vec1[i] + vec2[i]; // vec addition
}

void setrand(float *arr, int len){
    for (int i =0; i< len; i++) {
        // arr[i] = float((double)rand()/(double)(RAND_MAX/9));
        arr[i] = float(rand() % 10);

    }
}

void printvec(float *arr, int len){
    printf("[");
    for (int i =0; i < len; i++) {
        printf("%f ", arr[i]);
    }
    printf("]");

    printf("\n\n");
}

int main() {

    float* vec1d; float* vec2d; float* resultd; // the d means device
    int len = 5;

    // Make the first and second vector and put it on the GPU
    float vec1[len];
    setrand(vec1, len);
    cudaMalloc((void **) &vec1d, len * sizeof(float));
    cudaMemcpy((void*)vec1d, vec1, len * sizeof(float), cudaMemcpyHostToDevice);

    float vec2[len];
    setrand(vec2, len);
    cudaMalloc((void **) &vec2d, len * sizeof(float));
    cudaMemcpy(vec2d, vec2, len * sizeof(float), cudaMemcpyHostToDevice);

    printvec(vec1, len);
    printvec(vec2, len);


    float result[len];
    cudaMalloc((void **) &resultd, len * sizeof(float));

    dim3 dimBlock(len);
    dim3 dimGrid(1);
    
    matmulkernel<<<dimGrid, dimBlock>>>(vec1d, vec2d, resultd);
    cudaMemcpy(result, resultd, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(vec1d); cudaFree(vec2d); cudaFree(resultd);

    printvec(result, len);
    cudaDeviceSynchronize();
    return 0;
}