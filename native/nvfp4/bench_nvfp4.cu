#include "nvfp4.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(expr) do { \
    const cudaError_t error = (expr); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(error)); \
        std::exit(1); \
    } \
} while (0)

static int argument(int argc, char ** argv, const char * name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
    }
    return fallback;
}

template <typename Launch>
static float time_ms(Launch launch, int iterations) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 10; ++i) CUDA_CHECK(launch());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) CUDA_CHECK(launch());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    return elapsed / iterations;
}

int main(int argc, char ** argv) {
    const int rows = argument(argc, argv, "--rows", 4096);
    const int cols = argument(argc, argv, "--cols", 4096);
    const int iterations = argument(argc, argv, "--iterations", 100);
    if (rows <= 0 || cols <= 0 || cols % 16 != 0 || iterations <= 0) {
        std::fprintf(stderr, "rows/iterations must be positive and cols must be a positive multiple of 16\n");
        return 2;
    }

    const size_t packed_count = static_cast<size_t>(rows) * cols / 2;
    const size_t scale_count = static_cast<size_t>(rows) * cols / 16;
    const size_t fp16_count = static_cast<size_t>(rows) * cols;
    std::vector<uint8_t> packed(packed_count);
    std::vector<uint8_t> scales(scale_count);
    std::vector<__half> fp16(fp16_count);
    std::vector<__half> input(cols);
    for (int col = 0; col < cols; ++col) input[col] = __float2half((col % 13 - 6) / 6.0f);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < cols / 16; ++block) scales[row * (cols / 16) + block] = 0x38;
        for (int col = 0; col < cols; col += 2) {
            const uint8_t low = static_cast<uint8_t>((row + col) % 16);
            const uint8_t high = static_cast<uint8_t>((row + col + 1) % 16);
            packed[row * (cols / 2) + col / 2] = static_cast<uint8_t>(low | (high << 4));
            fp16[static_cast<size_t>(row) * cols + col] = __float2half(autoluce::nvfp4::decode_e2m1(low));
            fp16[static_cast<size_t>(row) * cols + col + 1] = __float2half(autoluce::nvfp4::decode_e2m1(high));
        }
    }

    uint8_t * d_packed = nullptr;
    uint8_t * d_scales = nullptr;
    __half * d_fp16 = nullptr;
    __half * d_input = nullptr;
    float * d_nvfp4_output = nullptr;
    float * d_fp16_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_packed, packed_count));
    CUDA_CHECK(cudaMalloc(&d_scales, scale_count));
    CUDA_CHECK(cudaMalloc(&d_fp16, fp16_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_input, cols * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_nvfp4_output, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fp16_output, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_packed, packed.data(), packed_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, scales.data(), scale_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fp16, fp16.data(), fp16_count * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), cols * sizeof(__half), cudaMemcpyHostToDevice));

    const float nvfp4_ms = time_ms([&] {
        return autoluce::nvfp4::launch_gemv(d_packed, d_scales, 1.0f, d_input, d_nvfp4_output, rows, cols);
    }, iterations);
    const float fp16_ms = time_ms([&] {
        return autoluce::nvfp4::launch_fp16_gemv(d_fp16, d_input, d_fp16_output, rows, cols);
    }, iterations);

    std::vector<float> nvfp4_output(rows);
    std::vector<float> fp16_output(rows);
    CUDA_CHECK(cudaMemcpy(nvfp4_output.data(), d_nvfp4_output, rows * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fp16_output.data(), d_fp16_output, rows * sizeof(float), cudaMemcpyDeviceToHost));
    float max_abs_error = 0.0f;
    for (int row = 0; row < rows; ++row) {
        max_abs_error = std::max(max_abs_error, std::abs(nvfp4_output[row] - fp16_output[row]));
    }
    const double bytes = static_cast<double>(packed_count + scale_count + cols * sizeof(__half));
    const double gib_s = bytes / (1024.0 * 1024.0 * 1024.0) / (nvfp4_ms / 1000.0);
    std::printf(
        "{\"backend\":\"cuda\",\"format\":\"nvfp4-e2m1-e4m3\",\"rows\":%d,\"cols\":%d,"
        "\"iterations\":%d,\"nvfp4_ms\":%.6f,\"fp16_ms\":%.6f,\"speedup_vs_fp16\":%.6f,"
        "\"nvfp4_effective_gib_s\":%.6f,\"max_abs_error\":%.9g}\n",
        rows, cols, iterations, nvfp4_ms, fp16_ms, fp16_ms / nvfp4_ms, gib_s, max_abs_error);

    CUDA_CHECK(cudaFree(d_fp16_output));
    CUDA_CHECK(cudaFree(d_nvfp4_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_fp16));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_packed));
    return max_abs_error <= 2e-4f ? 0 : 1;
}
