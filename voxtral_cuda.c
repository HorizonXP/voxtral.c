#include "voxtral_cuda.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t g_handle;
static int g_init = 0;
static int g_available = 0;

static void vox_cuda_init(void) {
    if (g_init) return;
    g_init = 1;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) return;
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) return;
    g_available = 1;
}

int vox_cuda_available(void) {
    vox_cuda_init();
    return g_available;
}

static int vox_cuda_gemm_rowmajor(float *C, const float *A, const float *B,
                                  int M, int K, int N, int b_is_transposed) {
    if (!vox_cuda_available()) return 0;

    size_t bytes_a = (size_t)M * K * sizeof(float);
    size_t bytes_b = (size_t)K * N * sizeof(float);
    if (b_is_transposed) bytes_b = (size_t)N * K * sizeof(float);
    size_t bytes_c = (size_t)M * N * sizeof(float);

    float *dA = NULL, *dB = NULL, *dC = NULL;
    if (cudaMalloc((void **)&dA, bytes_a) != cudaSuccess) return 0;
    if (cudaMalloc((void **)&dB, bytes_b) != cudaSuccess) {
        cudaFree(dA);
        return 0;
    }
    if (cudaMalloc((void **)&dC, bytes_c) != cudaSuccess) {
        cudaFree(dA);
        cudaFree(dB);
        return 0;
    }

    if (cudaMemcpy(dA, A, bytes_a, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dB, B, bytes_b, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return 0;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status;

    if (!b_is_transposed) {
        status = cublasSgemm(g_handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             dB, N,
                             dA, K,
                             &beta,
                             dC, N);
    } else {
        status = cublasSgemm(g_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             dB, K,
                             dA, K,
                             &beta,
                             dC, N);
    }

    int ok = (status == CUBLAS_STATUS_SUCCESS) &&
             (cudaMemcpy(C, dC, bytes_c, cudaMemcpyDeviceToHost) == cudaSuccess);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return ok;
}

int vox_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    return vox_cuda_gemm_rowmajor(C, A, B, M, K, N, 0);
}

int vox_cuda_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
    return vox_cuda_gemm_rowmajor(C, A, B, M, K, N, 1);
}

#else

int vox_cuda_available(void) { return 0; }
int vox_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    (void)C; (void)A; (void)B; (void)M; (void)K; (void)N;
    return 0;
}
int vox_cuda_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
    (void)C; (void)A; (void)B; (void)M; (void)K; (void)N;
    return 0;
}

#endif
