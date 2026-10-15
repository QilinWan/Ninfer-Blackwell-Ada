#include "ninfer/ops/rope.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int run_case(int axes, int heads, int tokens, int first_position, float factor, bool graph) {
    constexpr int dim               = 256;
    constexpr int rotary            = 64;
    const int stride                = dim * heads + 8;
    constexpr std::uint16_t padding = 0x3f81;
    std::vector<std::uint16_t> input(static_cast<std::size_t>(tokens) * stride, padding);
    std::vector<int> positions(static_cast<std::size_t>(tokens) * axes);
    for (int axis = 0; axis < axes; ++axis) {
        for (int t = 0; t < tokens; ++t) {
            positions[axis * tokens + t] = first_position + 83 * axis + 3 * t;
        }
    }
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            for (int i = 0; i < dim; ++i) {
                input[t * stride + h * dim + i] =
                    f32_to_bf16(static_cast<float>(((t * 13 + h * 7 + i * 3) % 61) - 30) / 8.0F);
            }
        }
    }
    const auto coefficients = ops::make_text_yarn_scaling(factor, 262144);
    GuardedDeviceBuffer data(input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer pos(positions.size() * sizeof(int));
    data.copy_from_host(input.data(), data.bytes());
    pos.copy_from_host(positions.data(), pos.bytes());
    Tensor x(data.data(), DType::BF16, {dim, heads, tokens});
    x.nb[2] = static_cast<std::int64_t>(stride) * 2;
    Tensor p(pos.data(), DType::I32, {tokens, axes});
    cudaStream_t stream        = nullptr;
    cudaGraph_t captured       = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    }
    ops::rope(p, rotary, 1.0e7F, x, stream, &coefficients);
    if (graph) {
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        // Another Engine can use another transform while this graph remains alive.
        const auto other = ops::make_text_yarn_scaling(factor == 1.5F ? 4.0F : 1.5F, 262144);
        ops::rope(p, rotary, 1.0e7F, x, stream, &other);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        data.copy_from_host(input.data(), data.bytes());
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        cuda_synchronize();
    }
    const auto got = from_device<std::uint16_t>(data.data(), input.size());
    int failures   = 0;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            const int offset = t * stride + h * dim;
            for (int pair = 0; pair < rotary / 2; ++pair) {
                const int axis         = axes == 3 ? pair % 3 : 0;
                const double frequency = coefficients.enabled
                                             ? coefficients.inverse_frequency[pair]
                                             : std::pow(1.0e7, -2.0 * pair / rotary);
                const double angle  = static_cast<double>(positions[axis * tokens + t]) * frequency;
                const double a      = bf16_to_f32(input[offset + pair]);
                const double b      = bf16_to_f32(input[offset + pair + rotary / 2]);
                const double c      = std::cos(angle) * coefficients.attention_factor;
                const double s      = std::sin(angle) * coefficients.attention_factor;
                const double want_a = a * c - b * s;
                const double want_b = b * c + a * s;
                const double tolerance =
                    0.0069 * std::hypot(a, b) * coefficients.attention_factor + 3e-5;
                if (std::abs(bf16_to_f32(got[offset + pair]) - want_a) > tolerance ||
                    std::abs(bf16_to_f32(got[offset + pair + rotary / 2]) - want_b) > tolerance) {
                    ++failures;
                }
            }
            for (int i = rotary; i < dim; ++i) { failures += got[offset + i] != input[offset + i]; }
        }
        for (int i = dim * heads; i < stride; ++i) { failures += got[t * stride + i] != padding; }
    }
    failures += data.verify_guards("YaRN data");
    failures += pos.verify_guards("YaRN positions");
    if (graph) {
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    if (failures) { std::cerr << "YaRN mismatch axes=" << axes << " factor=" << factor << '\n'; }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (float factor : {1.5F, 2.0F, 4.0F}) {
        for (int axes : {1, 3}) {
            failures += run_case(axes, 24, 1, 393212, factor, false);
            failures += run_case(axes, 4, 17, 1048500, factor, true);
        }
    }
    // Interleaving independent parameter sets must never mutate another Engine's transform.
    failures += run_case(3, 4, 3, 8192, 1.5F, false);
    failures += run_case(3, 4, 3, 8192, 4.0F, false);
    failures += run_case(3, 4, 3, 8192, 1.5F, false);
    std::cout << (failures ? "FAIL" : "OK") << " YaRN FP64 rotation oracle\n";
    return failures ? 1 : 0;
}
