#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h> 

/*
    mtxMult - perform matrix multiplication
    @params arrays A, B, C and dimension size N
    @return - void 
*/
__global__ void mtxMult(float *A, float *B, float *C, int N){

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float temp = 0;

    // compute for each thread
    for (int i = 0; i < N; i++) {
        temp += A[row * N + i] * B[i * N + col];
    }
    
    C[row * N + col] = temp;
}

/*
    setup - sets up the host and device arrays for calculation
    @params dim - dimensions of the arrays, isSmall - boolean to check whether the dimensions are small
    @return - void 
*/
void setup(int dim, bool isSmall){
    // Block and Tile Size
    int N = dim;
    int T = 1;
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

    clock_t begin = clock();
    // Copy host array to device array
    cudaMemcpy(d_A, h_A, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_CPU, memSize, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 dimGrid(N, N);
    dim3 dimBlock(T, T);
    mtxMult<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, N);

    // device to host copy
    cudaMemcpy(C_GPU, d_C, memSize, cudaMemcpyDeviceToHost );
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%dx%d Execution on CUDA: %lf seconds\n", N, N, time_spent);

    if(isSmall){
        // Run program sequentially
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                float temp = 0;
                for (int k = 0; k < N; k++){
                    temp += h_A[i*N + k] * h_B[k * N+j];
                }
                C_CPU[i * N + j] = temp;
            }
        }
    
        printf("Verifying program correctness.... ");
        // verify the data returned to the host is correct
        for (int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                assert(C_CPU[i*N + j] == C_GPU[i*N + j]);
            }
        }
        printf("Everthing checks out!\n");
    }

    // free host memory
    free(h_A);
    free(h_B);
    free(C_CPU);
    free(C_GPU);

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(){
    // test run on 64 dim
    printf("Running 64x64...\n");
    setup(64, true);

    // 4096*4096
    printf("Running 4096x4096...\n");
    //setup(4096, false);

    // 8192*8192
    printf("Running 8192x8192...\n");
    //setup(8192, false);

    return 0;
} // qsub hw11.sh -q UI-GPU -I ngpus=1