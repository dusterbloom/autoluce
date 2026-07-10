#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace autoluce::nvfp4 {

// NVFP4 E2M1 finite values. The sign is bit 3; bits 2:1 are exponent and bit 0
// is the mantissa. This explicit table is also the independent CPU oracle.
__host__ __device__ inline float decode_e2m1(uint8_t code) {
    constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = values[code & 0x7u];
    return (code & 0x8u) ? -value : value;
}

// NVIDIA E4M3 finite-number encoding: bias 7, subnormals at exponent zero,
// and 0x7f/0xff reserved for NaN. NVFP4 stores one such scale per 16 values.
__host__ __device__ inline float decode_e4m3(uint8_t code) {
    const int sign = (code & 0x80u) ? -1 : 1;
    const int exponent = (code >> 3) & 0x0f;
    const int mantissa = code & 0x07;
    if (exponent == 0x0f && mantissa == 0x07) {
        return NAN;
    }
    if (exponent == 0) {
        return sign * ldexpf(static_cast<float>(mantissa), -9);
    }
    return sign * ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

cudaError_t launch_gemv(
    const uint8_t * packed_weights,
    const uint8_t * e4m3_scales,
    float global_scale,
    const __half * input,
    float * output,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

cudaError_t launch_fp16_gemv(
    const __half * weights,
    const __half * input,
    float * output,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

}  // namespace autoluce::nvfp4
