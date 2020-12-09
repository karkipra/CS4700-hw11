#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h> 

/*
    References
    - no sharing method: Help from Andrew's office hours 
    - shared method: https://stackoverflow.com/questions/18815489/cuda-tiled-matrix-matrix-multiplication-with-shared-memory-and-matrix-size-whic/18856054 
*/

/*
    mtxMult - perform matrix multiplication
    @params arrays A, B, C and dimension size N
    @return - void 
*/
__global__ void sharedMtxMult(float *A, float *B, float *C, int N) {
    // Block Size
    int block_size = 8;

    __shared__ float tile_a[block_size][block_size];
    __shared__ float tile_b[block_size][block_size];

    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;
    float tmp = 0;
    int idx;

    for (int i = 0; i < gridDim.x; i++) {
        // A index
        idx = row * N + i * block_size + threadIdx.x;
        if(idx < N*N) {
            tile_a[threadIdx.y][threadIdx.x] = A[idx];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }

        // B index
        idx = (i * block_size + threadIdx.y) * N + col;
        if(idx < N*N) {
            tile_b[threadIdx.y][threadIdx.x] = B[idx];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        // update the temp (value of c)
        for (int N = 0; N < block_size; N++) {
            tmp += tile_a[threadIdx.y][N] * tile_b[N][threadIdx.x];
        }
        __syncthreads();
    }
    // Finally put the values in C
    if(row < N && col < N) {
        C[row * N + col] = tmp;
    }
}

/*
    mtxMult - perform matrix multiplication without sharing
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

    clock_t begin = clock();
    // Copy host array to device array
    cudaMemcpy(d_A, h_A, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_CPU, memSize, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 dimGrid(N, N);
    dim3 dimBlock(T, T);
    sharedMtxMult<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, N);

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
    // test run on 1024 dim
    printf("Running 1024x1024...\n");
    setup(1024, true);

    // 4096*4096
    printf("Running 4096x4096...\n");
    //setup(4096, false);

    // 8192*8192
    printf("Running 8192x8192...\n");
    //setup(8192, false);

    return 0;
} // qsub hw11.sh -q UI-GPU -I ngpus=1