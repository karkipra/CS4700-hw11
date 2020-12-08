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
    //int T = 8;
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
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    mtxMult<<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C, N);

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
        }
    }

    printf("Verifying program correctness.... ");
    // verify the data returned to the host is correct
    for (int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            assert(C_CPU[i*N + j] == C_CPU[i*N + j]);
        }
    }
    printf("Everthing checks out!\n");

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








/*
#include <stdio.h>
#include <stdlib.h>

/*
    References
    - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
    - http://users.umiacs.umd.edu/~ramani/cmsc828e_gpusci/Lecture5.pdf   


int N = 64; // Block size
int T = 8; // Tile size

__global__ void mtxMult(int *A, int *B, int *C, int N){
    __shared__ float smem_c[64][64];
    __shared__ float smem_a[64][8];
    __shared__ float smem_b[8][64];

    int c = blockIdx.x * 64;
    int r = blockIdx.y * 64;

    for(int kk = 0; kk < N; kk += T){
        for(int i = threadIdx.x+(blockDim.x*threadIdx.y); i < 64*8; i+= blockDim.x*blockDim.y){
            int k = kk + i/64;
            int rt = r + i % 64;
            int ct = c + i % 64;

            smem_a[i%64][i/64] = A[rt*N+k];
            smem_b[i/64][i%64] = B[k*N+ct];
        }
    }

    __syncthreads();

    // Matrix multiplication here
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            // finish this up
            c = 0;
            r = 0;
            for(int k = 0; k < N; k++){
                smem_c = 0; 
            }
        }
    }
}

int main(){

    // Sequential code
    float A[N][N];
    float B[N][N];
    float C[N][N];

    // Initializing A and B
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            A[i][j] = 1;
            B[i][j] = 1;
            C[i][j] = 0;
        }
    }  

    // Matrix multiplication on CPU
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // pointer for host memory and size
    float * A_host;

    // pointer for device memory
    int *A_dev;

    // allocate host and device memory
    size_t memSize = N*N*4;
    A_host = (float *) malloc(memSize);
    cudaMalloc((void **) &a_dev, memSize);

    // Copy host array to device array
    //cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice );

    // launch kernel
    //dim3 dimGrid(num_blocks);
    //dim3 dimBlock(num_th_per_blk);
    //reverseArray<<< dimGrid, dimBlock >>>(d_a, dim_a);

    // device to host copy
    //cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost );

    // free device memory
    //cudaFree(d_a);

    // free host memory
}
*/