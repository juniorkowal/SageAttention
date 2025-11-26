/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Inspired by CUTLASS, https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/numeric_conversion.h
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
// #include <cuda/pipeline>
// #include <hip/pipeline>

// #if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
// #if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
// #define FP8_CAST_ENABLED
// #endif
// #endif

// #if defined(__CUDA_ARCH__)
// #define RUNTIME_ASSERT(x) __brkpt()
// #else
// #include <assert.h>
// #define RUNTIME_ASSERT(x) assert(0 && x)
// #endif

// __device__ __forceinline__ void unpack_half2_from_uint32_to_float(float* dest, uint32_t source) {
//   uint16_t h0 = source & 0xFFFF;
//   uint16_t h1 = (source >> 16) & 0xFFFF;
//   asm("cvt.f32.f16 %0, %1;" : "=f"(dest[0]) : "h"(h0));
//   asm("cvt.f32.f16 %0, %1;" : "=f"(dest[1]) : "h"(h1));
// }

__device__ __forceinline__ void unpack_half2_from_uint32_to_float(float* dest, uint32_t source) {
    uint16_t h0 = source & 0xFFFF;
    uint16_t h1 = (source >> 16) & 0xFFFF;

    __half half0 = *reinterpret_cast<__half*>(&h0);
    __half half1 = *reinterpret_cast<__half*>(&h1);

    dest[0] = __half2float(half0);
    dest[1] = __half2float(half1);
}

// __device__ __forceinline__ void floatx4_to_e4m3x4(uint32_t *dest, float *source0, float *source1)
// {
// #ifdef FP8_CAST_ENABLED
//   asm volatile( \
//       "{\n" \
//       ".reg .b16 lo;\n" \
//       ".reg .b16 hi;\n" \
//       "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n" \
//       "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n" \
//       "mov.b32 %0, {lo, hi};\n" \
//       "}" \
//       : "=r"(dest[0]) : "f"(source0[0]), "f"(source0[1]), "f"(source1[0]), "f"(source1[1]));
// #else
//   RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
// #endif
// }


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
namespace detail {

  struct vec16_t { float x, y, z, w; };

  template <bool PadZero, typename T>
  __device__ __forceinline__ void predicated_g2s_16B(T* smem_dst, const T* gmem_src, bool pred) {
    if (pred) {
      *reinterpret_cast<vec16_t*>(smem_dst) = *reinterpret_cast<const vec16_t*>(gmem_src);
    } else if constexpr (PadZero) {
      *reinterpret_cast<vec16_t*>(smem_dst) = vec16_t{0.f, 0.f, 0.f, 0.f};
    }
  }

  template<typename T>
  __device__ __forceinline__ void load_8xT_to_regs(const T* __restrict__ ptr, T (&dst)[8]) {
    static_assert(sizeof(T) == 2, "T must be 16-bit (half/bfloat16)");
    const uint4* __restrict__ src = reinterpret_cast<const uint4*>(ptr);
    *reinterpret_cast<uint4*>(&dst[0]) = *src;
  }

  __device__ __forceinline__ void store_8fp8(const uint32_t* __restrict__ fp8x4,
                                           int8_t* __restrict__ out) {
    *reinterpret_cast<uint2*>(out) = *reinterpret_cast<const uint2*>(fp8x4);
  }

  // ---- E4M3 打包：float32 -> uint8（rn、satfinite、非 subnormal）----
  // E4M3：1|4|3，bias=7；rn-even；satfinite；不产生 subnormal（下溢置0）
  __device__ __forceinline__ uint8_t float_to_e4m3_rn_satfinite_relaxed(float x) {
    const uint8_t POS_MAX_CODE = 0x76;
    const uint8_t NEG_MAX_CODE = 0xF6;
    const uint8_t NAN_CODE        = (uint8_t)((15 << 3) | 1);
    
    if (isnan(x)) {
    // canonical NaN (positive sign); any mant != 0 is NaN
    return NAN_CODE;
    }
    if (!isfinite(x)) return signbit(x) ? NEG_MAX_CODE : POS_MAX_CODE;
    if (x == 0.0f)    return 0u;

    const uint8_t s = signbit(x) ? 0x80 : 0x00;
    float ax = fabsf(x);

    // 最小正规值 = 2^(1-bias) = 2^-6
    if (ax < (1.0f / 64.0f)) return s | 0x00;  // 不做 subnormal：直接 0

    // 规格化 ax = m * 2^e，m∈[1,2)
    int   e;
    float m = frexpf(ax, &e);   // m∈[0.5,1)
    m *= 2.0f; e -= 1;          // m∈[1,2)

    // 量化 3 位尾数（去掉隐含 1）：mant = rn_even((m-1)*8)
    float mant_f  = (m - 1.0f) * 8.0f;  // ∈[0,8)
    float floor_v = floorf(mant_f);
    float frac_v  = mant_f - floor_v;

    int mant;
    if      (frac_v > 0.5f) mant = (int)floor_v + 1;
    else if (frac_v < 0.5f) mant = (int)floor_v;
    else                    mant = ((int)floor_v & 1) ? (int)floor_v + 1 : (int)floor_v;

    // 尾数进位：mant==8 -> mant=0, e+1
    if (mant == 8) { mant = 0; e += 1; }

    // 带偏置指数
    int eb = e + 7;

    // 下溢：不做 subnormal，直接 0
    if (eb <= 0) return s | 0x00;

    // ------- 关键改动：指数=15 的处理规则 -------
    // 仅当 eb==15 且 mant==7 时，判定为“上溢”（你定义的溢出码）；
    // 其他 eb==15 且 mant!=7 的情况，按“有效数”编码（非常规做法，需上下游一致解码）。
    if (eb >= 0xF) {
        // eb==15 且 mant==7 -> 上溢（饱和到“最大码”）
        // 这里沿用你原有的“最大有限值”码（exp=14,mant=7），或你也可以选择返回 (15,7) 本码
      return s ? NEG_MAX_CODE : POS_MAX_CODE;
    }

    // 常规（eb=1..14）
    uint8_t e_bits = (uint8_t)(eb & 0xF);
    uint8_t m_bits = (uint8_t)(mant & 0x7);
    return s | (e_bits << 3) | m_bits;
  }

  __device__ __forceinline__ void floatx4_to_e4m3x4(uint32_t* dest, float* s0, float* s1) {
    //   lo = cvt(..., s0[1], s0[0]); hi = cvt(..., s1[1], s1[0]); mov.b32 {lo,hi}
    // uint8_t b0 = c10::Float8_e4m3fnuz(
    //       __hip_cvt_float_to_fp8(s0[0], __HIP_SATFINITE, __HIP_E4M3_FNUZ),
    //       c10::Float8_e4m3fnuz::from_bits());
    uint8_t b0 = __hip_cvt_float_to_fp8(s0[0], __HIP_SATFINITE, __HIP_E4M3_FNUZ);  
    uint8_t b1 = __hip_cvt_float_to_fp8(s0[1], __HIP_SATFINITE, __HIP_E4M3_FNUZ);  
    uint8_t b2 = __hip_cvt_float_to_fp8(s1[0], __HIP_SATFINITE, __HIP_E4M3_FNUZ);  
    uint8_t b3 = __hip_cvt_float_to_fp8(s1[1], __HIP_SATFINITE, __HIP_E4M3_FNUZ);  

    *dest = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
  }

} // namespace detail
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~









// __device__ __forceinline__ void floatx4_to_e5m2x4(uint32_t *dest, float *source0, float *source1)
// {
// #ifdef FP8_CAST_ENABLED
//   asm volatile( \
//       "{\n" \
//       ".reg .b16 lo;\n" \
//       ".reg .b16 hi;\n" \
//       "cvt.rn.satfinite.e5m2x2.f32   lo, %2, %1;\n" \
//       "cvt.rn.satfinite.e5m2x2.f32   hi, %4, %3;\n" \
//       "mov.b32 %0, {lo, hi};\n" \
//       "}" \
//       : "=r"(dest[0]) : "f"(source0[0]), "f"(source1[1]), "f"(source1[0]), "f"(source1[1]));
// #else
//   RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
// #endif
// }

// __device__ __forceinline__ void halfx4_to_e4m3x4(uint32_t *dest, uint32_t *source0, uint32_t *source1)
// {
// #ifdef FP8_CAST_ENABLED
//   asm volatile( \
//       "{\n" \
//       ".reg .b16 lo;\n" \
//       ".reg .b16 hi;\n" \
//       "cvt.rn.satfinite.e4m3x2.f16x2   lo, %1;\n" \
//       "cvt.rn.satfinite.e4m3x2.f16x2   hi, %2;\n" \
//       "mov.b32 %0, {lo, hi};\n" \
//       "}" \
//       : "=r"(dest[0]) : "r"(source0[0]), "r"(source1[0]));
// #else
//   RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
// #endif
// }

// __device__ __forceinline__ void halfx4_to_e5m2x4(uint32_t *dest, uint32_t *source0, uint32_t *source1)
// {
// #ifdef FP8_CAST_ENABLED
//   asm volatile( \
//       "{\n" \
//       ".reg .b16 lo;\n" \
//       ".reg .b16 hi;\n" \
//       "cvt.rn.satfinite.e5m2x2.f16x2   lo, %1;\n" \
//       "cvt.rn.satfinite.e5m2x2.f16x2   hi, %2;\n" \
//       "mov.b32 %0, {lo, hi};\n" \
//       "}" \
//       : "=r"(dest[0]) : "r"(source0[0]), "r"(source1[0]));
// #else
//   RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
// #endif
// }

// __device__ __forceinline__ void e4m3x4_to_halfx4(uint32_t *dest0, uint32_t *dest1, uint32_t *source)
// {
// #ifdef FP8_CAST_ENABLED
//   asm volatile( \
//       "{\n" \
//       ".reg .b16 lo, hi;\n" \
//       "mov.b32 {lo, hi}, %2;\n" \
//       "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
//       "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
//       "}\n" : "=r"(dest0[0]), "=r"(dest1[0]) : "r"(source[0]));
// #else
//   RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
// #endif
// }

// __device__ __forceinline__ void e5m2x4_to_halfx4(uint32_t *dest0, uint32_t *dest1, uint32_t *source)
// {
// #ifdef FP8_CAST_ENABLED
//   asm volatile( \
//       "{\n" \
//       ".reg .b16 lo, hi;\n" \
//       "mov.b32 {lo, hi}, %2;\n" \
//       "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
//       "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
//       "}\n" : "=r"(dest0[0]), "=r"(dest1[0]) : "r"(source[0]));
// #else
//   RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
// #endif
// }

__device__ __forceinline__ int8_t float_to_int8_rn(float x) {
  static constexpr auto i8_min = static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max = static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate
  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // dst = std::clamp(dst, i8_min, i8_max);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
}