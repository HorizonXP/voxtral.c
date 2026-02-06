#ifndef VOXTRAL_CUDA_H
#define VOXTRAL_CUDA_H

int vox_cuda_available(void);
int vox_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N);
int vox_cuda_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);
const char *vox_cuda_device_name(void);

#endif
