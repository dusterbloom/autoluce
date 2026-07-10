#include "nvfp4.cuh"

namespace autoluce::nvfp4 {
namespace {

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__inline__ __device__ float block_sum(float value) {
    __shared__ float warps[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        warps[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < blockDim.x / 32 ? warps[lane] : 0.0f;
    return warp_sum(value);
}

__global__ void nvfp4_gemv_kernel(
    const uint8_t * packed_weights,
    const uint8_t * e4m3_scales,
    float global_scale,
    const __half * input,
    float * output,
    int cols) {
    const int row = blockIdx.x;
    const size_t packed_row = static_cast<size_t>(row) * (cols / 2);
    const size_t scale_row = static_cast<size_t>(row) * (cols / 16);
    float sum = 0.0f;
    const int blocks = cols / 16;
    for (int block = threadIdx.x; block < blocks; block += blockDim.x) {
        const float scale = decode_e4m3(e4m3_scales[scale_row + block]) * global_scale;
        const size_t byte_offset = packed_row + static_cast<size_t>(block) * 8;
        const int input_offset = block * 16;
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const uint8_t packed = packed_weights[byte_offset + pair];
            const __half2 values = reinterpret_cast<const __half2 *>(input + input_offset)[pair];
            const float2 inputs = __half22float2(values);
            sum = fmaf(decode_e2m1(packed & 0x0fu) * scale, inputs.x, sum);
            sum = fmaf(decode_e2m1(packed >> 4) * scale, inputs.y, sum);
        }
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        output[row] = sum;
    }
}

__global__ void fp16_gemv_kernel(
    const __half * weights,
    const __half * input,
    float * output,
    int cols) {
    const int row = blockIdx.x;
    const size_t weight_row = static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        sum = fmaf(__half2float(weights[weight_row + col]), __half2float(input[col]), sum);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        output[row] = sum;
    }
}

}  // namespace

cudaError_t launch_gemv(
    const uint8_t * packed_weights,
    const uint8_t * e4m3_scales,
    float global_scale,
    const __half * input,
    float * output,
    int rows,
    int cols,
    cudaStream_t stream) {
    if (!packed_weights || !e4m3_scales || !input || !output || rows <= 0 || cols <= 0 || cols % 16 != 0) {
        return cudaErrorInvalidValue;
    }
    nvfp4_gemv_kernel<<<rows, 256, 0, stream>>>(
        packed_weights, e4m3_scales, global_scale, input, output, cols);
    return cudaGetLastError();
}

cudaError_t launch_fp16_gemv(
    const __half * weights,
    const __half * input,
    float * output,
    int rows,
    int cols,
    cudaStream_t stream) {
    if (!weights || !input || !output || rows <= 0 || cols <= 0) {
        return cudaErrorInvalidValue;
    }
    fp16_gemv_kernel<<<rows, 256, 0, stream>>>(weights, input, output, cols);
    return cudaGetLastError();
}

}  // namespace autoluce::nvfp4
