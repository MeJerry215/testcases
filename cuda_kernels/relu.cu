#include "common.h"

__global__ void relu_kernel(float* x, float* y, int32_t n_elem) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elem) y[idx] = x[idx] < 0 ? 0 : x[idx];
}



int main(int argc, char** argv) {
    const size_t max_elem_cnts = 1024 * 1024 * 32;
    const size_t elem_cnts[] = {
        // 2048,
        // 4096,
        // 8192,
        16384
        // 1024 * 512,
        // 1024 * 1024
    };

    float *hx = (float*)malloc(bytes_of<float>(max_elem_cnts));
    float *hy = (float*)malloc(bytes_of<float>(max_elem_cnts));
    float *dx = nullptr;
    float *dy = nullptr;
    CHECK_CALL_ERROR(cudaMalloc(&dx, bytes_of<float>(max_elem_cnts)));
    CHECK_CALL_ERROR(cudaMalloc(&dy, bytes_of<float>(max_elem_cnts)));
    gen_random<float>(hx, max_elem_cnts);
    CHECK_CALL_ERROR(cudaMemcpy(dx, hx, max_elem_cnts, cudaMemcpyHostToDevice));
    for (size_t elem_cnt: elem_cnts) {
        // dim3 blockSize(min(size_t(1024), elem_cnt));
        // dim3 gridSize((elem_cnt + blockSize.x - 1) / blockSize.x);
        // cout << "elem_cnt " << elem_cnt << " block size " << blockSize.x << " grid size " << gridSize.x << endl;
        // warm up
        // relu_kernel<<<gridSize, blockSize>>>(dx, dy);
        // EVT_BEG(1)
        // relu_kernel<<<gridSize, blockSize>>>(dx, dy);
        // CHECK_CUDA_ERROR();
        // EVT_END(1)
    }

}

