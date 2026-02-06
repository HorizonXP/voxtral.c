#include "voxtral_cuda.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string.h>

static cublasHandle_t g_handle;
static cudaStream_t g_stream;
static int g_init = 0;
static int g_available = 0;
static char g_device_name[256] = "unavailable";

static float *g_dA = NULL;
static float *g_dB = NULL;
static float *g_dC = NULL;
static size_t g_cap_a = 0;
static size_t g_cap_b = 0;
static size_t g_cap_c = 0;

static int ensure_buffer(float **buf, size_t *cap, size_t needed_bytes) {
    if (*cap >= needed_bytes) return 1;
    if (*buf != NULL) cudaFree(*buf);
    *buf = NULL;
    *cap = 0;
    if (cudaMalloc((void **)buf, needed_bytes) != cudaSuccess) return 0;
    *cap = needed_bytes;
    return 1;
}

static void vox_cuda_init(void) {
    if (g_init) return;
    g_init = 1;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) return;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        strncpy(g_device_name, prop.name, sizeof(g_device_name) - 1);
        g_device_name[sizeof(g_device_name) - 1] = '\0';
    }

    if (cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking) != cudaSuccess) return;
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) return;
    if (cublasSetStream(g_handle, g_stream) != CUBLAS_STATUS_SUCCESS) return;

    g_available = 1;
}

int vox_cuda_available(void) {
    vox_cuda_init();
    return g_available;
}

const char *vox_cuda_device_name(void) {
    vox_cuda_init();
    return g_device_name;
}

static int vox_cuda_gemm_rowmajor(float *C, const float *A, const float *B,
                                  int M, int K, int N, int b_is_transposed) {
    if (!vox_cuda_available()) return 0;

    size_t bytes_a = (size_t)M * K * sizeof(float);
    size_t bytes_b = b_is_transposed ? (size_t)N * K * sizeof(float)
                                     : (size_t)K * N * sizeof(float);
    size_t bytes_c = (size_t)M * N * sizeof(float);

    if (!ensure_buffer(&g_dA, &g_cap_a, bytes_a) ||
        !ensure_buffer(&g_dB, &g_cap_b, bytes_b) ||
        !ensure_buffer(&g_dC, &g_cap_c, bytes_c)) {
        return 0;
    }

    if (cudaMemcpyAsync(g_dA, A, bytes_a, cudaMemcpyHostToDevice, g_stream) != cudaSuccess ||
        cudaMemcpyAsync(g_dB, B, bytes_b, cudaMemcpyHostToDevice, g_stream) != cudaSuccess) {
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
                             g_dB, N,
                             g_dA, K,
                             &beta,
                             g_dC, N);
    } else {
        status = cublasSgemm(g_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             g_dB, K,
                             g_dA, K,
                             &beta,
                             g_dC, N);
    }

    if (status != CUBLAS_STATUS_SUCCESS) return 0;

    if (cudaMemcpyAsync(C, g_dC, bytes_c, cudaMemcpyDeviceToHost, g_stream) != cudaSuccess) {
        return 0;
    }

    return (cudaStreamSynchronize(g_stream) == cudaSuccess);
}

int vox_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    return vox_cuda_gemm_rowmajor(C, A, B, M, K, N, 0);
}

int vox_cuda_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
    return vox_cuda_gemm_rowmajor(C, A, B, M, K, N, 1);
}

#else

int vox_cuda_available(void) { return 0; }
const char *vox_cuda_device_name(void) { return "disabled"; }
int vox_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    (void)C; (void)A; (void)B; (void)M; (void)K; (void)N;
    return 0;
}
int vox_cuda_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
    (void)C; (void)A; (void)B; (void)M; (void)K; (void)N;
    return 0;
}

#endif
