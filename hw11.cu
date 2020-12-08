#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void mtxMult(float *A, float *B, float *C, int N){

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float temp = 0;

    if(row < N && col < N){
        // compute for each thread
        for (int i = 0; i < N; i++) {
            temp += A[row * N + i] * B[i * N + col];
        }
    }
    C[row * N + col] = temp;
}

int main(){
    // Block and Tile Size
    int N = 64;
    int T = 8;
    size_t memSize = N * N * sizeof(int);
    
    printf("Running on N = %d\n", N);
    // Allocate host memory
    float* h_A;
    float* h_B;
    float* C_CPU;
    float* C_GPU;
    h_A = (float *) malloc(memSize);
    h_B = (float *) malloc(memSize);
    C_CPU = (float *) malloc(memSize);
    C_GPU = (float *) malloc(memSize);

    // Allocate device memory
    float* d_A; 
    float* d_B;
    float* d_C;
    cudaMalloc((void **) &d_A, memSize);
    cudaMalloc((void **) &d_B, memSize);
    cudaMalloc((void **) &d_C, memSize);

    // Initialize matrices on host
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            h_A[i*N+j] = 1;
            h_B[i*N+j] = 1;
            C_CPU[i*N+j] = 0;
        }
    }

    // Copy host array to device array
    cudaMemcpy(d_A, h_A, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_CPU, memSize, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 dimGrid(N/T);
    dim3 dimBlock(T);
    mtxMult<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, N);

    // device to host copy
    cudaMemcpy(C_GPU, d_C, memSize, cudaMemcpyDeviceToHost );

    // Run program sequentially
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            float temp = 0;
            for (int k = 0; k < N; k++){
                temp += h_A[i*N + k] * h_B[k * N+j];
            }
            C_CPU[i * N + j] = temp;
            printf("C_CPU[%d] = %f and C_GPU[%d] = %f\n", i * N + j, C_CPU[i * N + j], i * N + j, C_GPU[i * N + j]);
        }
    }

    /*
    printf("First 20 C CPU: ");
    for (int i = 0; i < 20; i++){
        printf("%f ", C_CPU[i]);
    }
    printf("\n");

    printf("First 20 C GPU: ");
    for (int i = 0; i < 20; i++){
        printf("%f ", C_GPU[i]);
    }
    printf("\n");

    printf("Verifying program correctness.... ");
    // verify the data returned to the host is correct
    for (int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            assert(C_CPU[i*N + j] == C_GPU[i*N + j]);
        }
    }
    printf("Everthing checks out!\n");*/

    // free host memory
    free(h_A);
    free(h_B);
    free(C_CPU);
    free(C_GPU);

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
} // qsub hw11.sh -q UI-GPU -I ngpus=1