#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>
#include <stdint.h>

using std::vector;
using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;
using std::random_device;
using std::mt19937;
using std::uniform_int_distribution;
using std::srand;
using std::rand;
using std::is_same;
using std::min;
using std::max;

#define CHECK_CUDA_ERROR() \
    do { \
        cudaError_t result = cudaDeviceSynchronize(); \
        if (result != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(result)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CALL_ERROR(call) \
    do { \
        cudaError_t result = call; \
        if (result != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(result)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define PRO_BEG   \
    {   \
        auto start = std::chrono::high_resolution_clock::now(); \

#define PRO_END \
    auto end = std::chrono::high_resolution_clock::now();\
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
    double elapsed = duration.count() / 1000.0;  \
    cout << "Elapsed time: " << fixed << setprecision(4) << elapsed << " milliseconds" << endl;  \
    }

#define EVT_BEG(repeat_time) \
    {   \
        cudaEvent_t start, stop;    \
        cudaEventCreate(&start);    \
        cudaEventCreate(&stop);     \
        cudaEventRecord(start, 0);      \
        for(int repeat_cnt = 0; repeat_cnt < repeat_time; repeat_cnt++) {


#define EVT_END(repeat_time) \
    }                           \
    cudaEventRecord(stop, 0);   \
    cudaEventSynchronize(stop); \
    float milliseconds = 0.0f;  \
    cudaEventElapsedTime(&milliseconds, start, stop);   \
    cout << "repeat " << repeat_time << " times, total time " << milliseconds   \
        << " avg time " << fixed << setprecision(4) << milliseconds / repeat_time << " milliseconds" << endl;  \
    }


template<typename T>
size_t bytes_of()
{
    return sizeof(T);
}

template <typename T, typename... Args>
size_t bytes_of(size_t num, Args... args)
{
    return num * bytes_of<T>(args...);
}

size_t num_of_elems() {
    return 1;
}

template<typename... Args>
size_t num_of_elems(size_t num, Args... args) {
    return num *  bytes_of(args...);
}

enum DTYPE {
    DOUBLE = 0,
    FLOTAT64 = 0,
    FLOAT32 = 1,
    FLOAT16 = 2,
    HALF = 2,
    INT64 = 3,
    INT32 = 4,
    INT8 = 5,
    UINT64 = 6,
    UINT32 = 7,
    UINT8 = 8,
    UNK_TYPE
};

template<typename T>
DTYPE checkType()
{
    if (is_same<T, double>::value)
        return DOUBLE;
    else if (is_same<T, float>::value)
        return FLOAT32;
    else if (is_same<T, half>::value)
        return HALF;
    else if (is_same<T, int64_t>::value)
        return INT64;
    else if (is_same<T, int32_t>::value)
        return INT32;
    else if (is_same<T, int8_t>::value)
        return INT8;
    else if (is_same<T, uint64_t>::value)
        return UINT64;
    else if (is_same<T, uint32_t>::value)
        return UINT32;
    else if (is_same<T, uint8_t>::value)
        return UINT8;
    else
        return UNK_TYPE;
}

template<typename T>
T random_value()
{
    DTYPE type = checkType<T>();

    switch (type) {
        case DOUBLE: {
            T rand_val = static_cast<T>(rand()) / RAND_MAX * 2.0 - 1.0;
            return rand_val;
        }

        case FLOAT32: {
            T rand_val = static_cast<T>(rand()) / RAND_MAX * 2.0 - 1.0;
            return rand_val;
        }

        case HALF: {
            T rand_val = static_cast<T>(rand()) / RAND_MAX * 2.0 - 1.0;
            return rand_val;
        }

        case INT64: return 0;

        case INT32: return 0;

        case INT8: return 0;

        case UINT64: return 0;

        case UINT32: return 0;

        case UINT8: return 0;

        default:
            return 0;
    }
}

template<typename T>
void gen_random(T *data, size_t n_elems) {
    assert(data != nullptr);
    while (n_elems--)
    {
        data[n_elems] = random_value<T>();
    }
}

