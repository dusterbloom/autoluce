#include "nvfp4.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr) do { \
    const cudaError_t error = (expr); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(error)); \
        std::exit(1); \
    } \
} while (0)

using autoluce::nvfp4::decode_e2m1;
using autoluce::nvfp4::decode_e4m3;

int main() {
    const float expected_e2m1[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    for (int code = 0; code < 16; ++code) {
        if (decode_e2m1(static_cast<uint8_t>(code)) != expected_e2m1[code]) {
            std::fprintf(stderr, "E2M1 decode mismatch for code %d\n", code);
            return 1;
        }
    }
    if (decode_e4m3(0x38) != 1.0f || decode_e4m3(0x30) != 0.5f ||
        decode_e4m3(0x40) != 2.0f || decode_e4m3(0x7e) != 448.0f ||
        !std::isnan(decode_e4m3(0x7f))) {
        std::fprintf(stderr, "E4M3 scale oracle failed\n");
        return 1;
    }

    constexpr int rows = 37;
    constexpr int cols = 272;
    constexpr float global_scale = 0.75f;
    std::vector<uint8_t> packed(rows * cols / 2);
    std::vector<uint8_t> scales(rows * cols / 16);
    std::vector<__half> input(cols);
    std::vector<float> expected(rows, 0.0f);
    std::vector<float> actual(rows, 0.0f);

    for (int col = 0; col < cols; ++col) {
        input[col] = __float2half((static_cast<float>(col % 17) - 8.0f) / 8.0f);
    }
    const uint8_t scale_codes[] = {0x30, 0x34, 0x38, 0x3c, 0x40};
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < cols / 16; ++block) {
            scales[row * (cols / 16) + block] = scale_codes[(row + block) % 5];
        }
        for (int col = 0; col < cols; col += 2) {
            const uint8_t low = static_cast<uint8_t>((row * 3 + col) % 16);
            const uint8_t high = static_cast<uint8_t>((row * 5 + col + 1) % 16);
            packed[row * (cols / 2) + col / 2] = static_cast<uint8_t>(low | (high << 4));
        }
        for (int col = 0; col < cols; ++col) {
            const uint8_t byte = packed[row * (cols / 2) + col / 2];
            const uint8_t code = (col & 1) ? byte >> 4 : byte & 0x0f;
            const float scale = decode_e4m3(scales[row * (cols / 16) + col / 16]);
            expected[row] += decode_e2m1(code) * scale * global_scale * __half2float(input[col]);
        }
    }

    uint8_t * d_packed = nullptr;
    uint8_t * d_scales = nullptr;
    __half * d_input = nullptr;
    float * d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_packed, packed.size()));
    CUDA_CHECK(cudaMalloc(&d_scales, scales.size()));
    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output, actual.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(autoluce::nvfp4::launch_gemv(
        d_packed, d_scales, global_scale, d_input, d_output, rows, cols));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), d_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs_error = 0.0f;
    for (int row = 0; row < rows; ++row) {
        max_abs_error = std::max(max_abs_error, std::abs(expected[row] - actual[row]));
    }
    if (max_abs_error > 2e-4f) {
        std::fprintf(stderr, "NVFP4 CUDA oracle mismatch: max_abs_error=%g\n", max_abs_error);
        return 1;
    }
    if (autoluce::nvfp4::launch_gemv(d_packed, d_scales, global_scale, d_input, d_output, rows, cols - 1)
        != cudaErrorInvalidValue) {
        std::fprintf(stderr, "invalid column count was not rejected\n");
        return 1;
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_packed));
    std::printf("{\"status\":\"pass\",\"rows\":%d,\"cols\":%d,\"max_abs_error\":%.9g}\n",
                rows, cols, max_abs_error);
    return 0;
}
