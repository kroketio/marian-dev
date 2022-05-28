/*
 * This file follows intgemm and is a means of retrofitting ruy into the intgemm based wiring in
 * `intgemm_interface.h`. ruy is an inference backend used in tensorflow and android deployment and
 * has an optimized ARM backend for the multiply operations required. Optimized code for quantize,
 * unquantize, transpose are added separately to connect the multiply library to marian.
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "ruy/platform.h"
#include "ruy/system_aligned_alloc.h"

#if RUY_PLATFORM_NEON
#include <arm_neon.h>
#endif

namespace marian {
namespace cpu {
namespace integer {

using Index = unsigned int;

// The following partitions a pure C++ slow implementation and a faster SIMD implementation using
// NEON intrinsics on ARM hardware. Ruy already has such a routing, but we add some preprocessing
// and postprocessing functions (quantize, transpose, unquantize) that are outside ruy's offerings
// and required in the fast matrix-multiplication workflow for machine-translation, that exists in
// marian.

enum class Path {
  kStandardCpp = 0,  // Pure C++
  kNeon = 1          // NEON Intrinsics (ARM)
};

#if RUY_PLATFORM_NEON
constexpr Path kHighestPath = Path::kNeon;
#else
constexpr Path kHighestPath = Path::kStandardCpp;
#endif

template <enum Path>
struct Preprocess;

/*
 * Naive implementation using standard C++ functions. Not optimized using SIMD operations.
 */
template <>
struct Preprocess<Path::kStandardCpp> {
  static void quantize(const float *input, int8_t *output, float scale, Index rows, Index width) {
    const Index size = rows * width;
    for(Index i = 0; i < size; i++) {
      // Round to nearest after multiplying with scale.
      float value = roundf(scale * input[i]);

      // Since float can store bigger values, we threshold anything that's gone
      // higher and can't fit in int8.
      value = std::max<float>(-127.0f, value);
      value = std::min<float>(127.0f, value);

      // Finally a static cast.
      output[i] = static_cast<int8_t>(value);
    };
  }

  template <class Scalar>
  static void transpose(const Scalar *input, Index rows, Index cols, Scalar *output) {
    for(Index i = 0; i < rows; i++) {
      for(Index j = 0; j < cols; j++) {
        output[j * rows + i] = input[i * cols + j];
      }
    }
  }

  struct UnquantizeAndAddBiasAndWrite {
    UnquantizeAndAddBiasAndWrite(float unquant_multiplier, const float *input_bias_prepared)
        : unquant_multiplier_(unquant_multiplier), input_bias_prepared_(input_bias_prepared) {}

    void operator()(const int32_t *input, Index rows_A, Index cols_B, float *output) const {
      for(Index i = 0; i < rows_A; i++) {
        for(Index j = 0; j < cols_B; j++) {
          Index idx = i * cols_B + j;
          output[idx] = (input[idx] * unquant_multiplier_) + input_bias_prepared_[j];
        }
      }
    }

  private:
    float unquant_multiplier_;
    const float *input_bias_prepared_;
  };

  struct UnquantizeAndWrite {
    explicit UnquantizeAndWrite(float unquant_multiplier)
        : unquant_multiplier_(unquant_multiplier) {}

    void operator()(const int32_t *input, Index rows_A, Index cols_B, float *output) const {
      for(Index i = 0; i < rows_A; i++) {
        for(Index j = 0; j < cols_B; j++) {
          Index idx = i * cols_B + j;
          output[idx] = (input[idx] * unquant_multiplier_);
        }
      }
    }

  private:
    float unquant_multiplier_;
  };
};

#if RUY_PLATFORM_NEON

/*
 * Optimized path using ARM NEON SIMD intrinsics. Currently only supports int8_t.
 * TODO: Expand support to 16-bit.
 */
template <>
struct Preprocess<Path::kNeon> {
  static void quantize(const float *input, int8_t *output, float scale, Index rows, Index width) {
    const float32x4_t *Input = reinterpret_cast<const float32x4_t *>(input);
    const float32x4_t *InputEnd = reinterpret_cast<const float32x4_t *>(input + rows * width);

    int8x8_t *Output = reinterpret_cast<int8x8_t *>(output);
    while(Input != InputEnd) {
      // Vector multiply by scalar
      // float32x4_t vmulq_n_f32(float32x4_t a, float32_t b);
      // VMUL.F32 q0,q0,d0[0]
      float32x4_t scaledFloat_lo = vmulq_n_f32(*Input++, scale);

      // Convert from float
      // int32x4_t  vcvtnq_s32_f32(float32x4_t a);
      // VCVT.S32.F32 q0, q0
      int32x4_t scaledInt_lo = vcvtnq_s32_f32(scaledFloat_lo);

      // Vector saturating narrow integer
      // int16x4_t  vqmovn_s32(int32x4_t a);   // VQMOVN.S32 d0,q0
      int16x4_t s16x4_lo = vqmovn_s32(scaledInt_lo);

      // Vector multiply by scalar
      // float32x4_t vmulq_n_f32(float32x4_t a, float32_t b);
      // VMUL.F32 q0,q0,d0[0]
      float32x4_t scaledFloat_hi = vmulq_n_f32(*Input++, scale);

      // Convert from float
      // int32x4_t  vcvtnq_s32_f32(float32x4_t a);
      // VCVT.S32.F32 q0, q0
      int32x4_t scaledInt_hi = vcvtnq_s32_f32(scaledFloat_hi);

      // Vector saturating narrow integer
      // int16x4_t  vqmovn_s32(int32x4_t a);
      // VQMOVN.S32 d0,q0
      int16x4_t s16x4_hi = vqmovn_s32(scaledInt_hi);

      // Combine two ints.
      // int16x8_t   vcombine_s16(int16x4_t low, int16x4_t high);
      int16x8_t s16x8 = vcombine_s16(s16x4_lo, s16x4_hi);

      // Vector saturating narrow integer
      int8x8_t s8x8 = vqmovn_s16(s16x8);

      *Output = s8x8;
      ++Output;
    };
  }

  // Specialization for int8_t
  static void transpose(const int8_t *input, Index rows, Index cols, int8_t *output) {
    constexpr Index tile_size = 16;
    // TODO(jerin): Enable
    // assert(rows % tile_size == 0 && cols & tile_size == 0);
    for(Index i = 0; i < rows; i += tile_size) {
      for(Index j = 0; j < cols; j += tile_size) {
        _transpose_16x16(input, i, j, rows, cols, output);
      }
    }
  }

  static void _transpose_16x16(const int8_t *src,
                               Index i,
                               Index j,
                               Index rows,
                               Index cols,
                               int8_t *dst) {
    // Implemented following the algorithm described in
    // https://stackoverflow.com/a/29587984/4565794
    //
    // permute n 32-bit rows
    // permute n 64-bit rows
    // ...
    // permute n simd_width/2-bit rows

    // clang-format off
    
    // Permute 8 8-bit rows.
    // Load int8x16x2 from memory into SIMD registers, transpose as 2x2 matrices.

    Index srcRowBegin = i*cols + j;
    int8x16x2_t r0 = vtrnq_s8(vld1q_s8(&src[ 0*cols + srcRowBegin]), vld1q_s8(&src[ 1*cols + srcRowBegin]));
    int8x16x2_t r1 = vtrnq_s8(vld1q_s8(&src[ 2*cols + srcRowBegin]), vld1q_s8(&src[ 3*cols + srcRowBegin]));
    int8x16x2_t r2 = vtrnq_s8(vld1q_s8(&src[ 4*cols + srcRowBegin]), vld1q_s8(&src[ 5*cols + srcRowBegin]));
    int8x16x2_t r3 = vtrnq_s8(vld1q_s8(&src[ 6*cols + srcRowBegin]), vld1q_s8(&src[ 7*cols + srcRowBegin]));
    int8x16x2_t r4 = vtrnq_s8(vld1q_s8(&src[ 8*cols + srcRowBegin]), vld1q_s8(&src[ 9*cols + srcRowBegin]));
    int8x16x2_t r5 = vtrnq_s8(vld1q_s8(&src[10*cols + srcRowBegin]), vld1q_s8(&src[11*cols + srcRowBegin]));
    int8x16x2_t r6 = vtrnq_s8(vld1q_s8(&src[12*cols + srcRowBegin]), vld1q_s8(&src[13*cols + srcRowBegin]));
    int8x16x2_t r7 = vtrnq_s8(vld1q_s8(&src[14*cols + srcRowBegin]), vld1q_s8(&src[15*cols + srcRowBegin]));


    // Permute 8 16-bit rows.
    // Next step is to treat the entries as int16x8x2 (via cast) and do
    // transpose for int16, which will now leave intra-2 pairs intact while
    // transposing inter 2-pairs into the right places.
    int16x8x2_t t0 = vtrnq_s16(vreinterpretq_s16_s8(r0.val[0]), vreinterpretq_s16_s8(r1.val[0]));
    int16x8x2_t t1 = vtrnq_s16(vreinterpretq_s16_s8(r2.val[0]), vreinterpretq_s16_s8(r3.val[0]));
    int16x8x2_t t2 = vtrnq_s16(vreinterpretq_s16_s8(r4.val[0]), vreinterpretq_s16_s8(r5.val[0]));
    int16x8x2_t t3 = vtrnq_s16(vreinterpretq_s16_s8(r6.val[0]), vreinterpretq_s16_s8(r7.val[0]));
    int16x8x2_t t4 = vtrnq_s16(vreinterpretq_s16_s8(r0.val[1]), vreinterpretq_s16_s8(r1.val[1]));
    int16x8x2_t t5 = vtrnq_s16(vreinterpretq_s16_s8(r2.val[1]), vreinterpretq_s16_s8(r3.val[1]));
    int16x8x2_t t6 = vtrnq_s16(vreinterpretq_s16_s8(r4.val[1]), vreinterpretq_s16_s8(r5.val[1]));
    int16x8x2_t t7 = vtrnq_s16(vreinterpretq_s16_s8(r6.val[1]), vreinterpretq_s16_s8(r7.val[1]));

    // Permute 8 32-bit rows.
    int32x4x2_t x0 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[0]), vreinterpretq_s32_s16(t1.val[0]));
    int32x4x2_t x1 = vtrnq_s32(vreinterpretq_s32_s16(t4.val[0]), vreinterpretq_s32_s16(t5.val[0]));
    int32x4x2_t x2 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[1]), vreinterpretq_s32_s16(t1.val[1]));
    int32x4x2_t x3 = vtrnq_s32(vreinterpretq_s32_s16(t4.val[1]), vreinterpretq_s32_s16(t5.val[1]));

    int32x4x2_t x4 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[0]), vreinterpretq_s32_s16(t3.val[0]));
    int32x4x2_t x5 = vtrnq_s32(vreinterpretq_s32_s16(t6.val[0]), vreinterpretq_s32_s16(t7.val[0]));
    int32x4x2_t x6 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[1]), vreinterpretq_s32_s16(t3.val[1]));
    int32x4x2_t x7 = vtrnq_s32(vreinterpretq_s32_s16(t6.val[1]), vreinterpretq_s32_s16(t7.val[1]));

    // There is no permute 8 64-bit rows available. 
    // Instead we follow extracting low and high and placing them into the right places.
    Index dstRowBegin = j*rows + i;
    vst1q_s8(&dst[ 0*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x0.val[0]),  vget_low_s32(x4.val[0])))); 
    vst1q_s8(&dst[ 1*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x1.val[0]),  vget_low_s32(x5.val[0]))));
    vst1q_s8(&dst[ 2*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x2.val[0]),  vget_low_s32(x6.val[0]))));
    vst1q_s8(&dst[ 3*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x3.val[0]),  vget_low_s32(x7.val[0]))));
    vst1q_s8(&dst[ 4*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x0.val[1]),  vget_low_s32(x4.val[1]))));
    vst1q_s8(&dst[ 5*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x1.val[1]),  vget_low_s32(x5.val[1]))));
    vst1q_s8(&dst[ 6*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x2.val[1]),  vget_low_s32(x6.val[1]))));
    vst1q_s8(&dst[ 7*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x3.val[1]),  vget_low_s32(x7.val[1]))));

    vst1q_s8(&dst[ 8*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x0.val[0]), vget_high_s32(x4.val[0]))));
    vst1q_s8(&dst[ 9*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x1.val[0]), vget_high_s32(x5.val[0]))));
    vst1q_s8(&dst[10*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x2.val[0]), vget_high_s32(x6.val[0]))));
    vst1q_s8(&dst[11*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x3.val[0]), vget_high_s32(x7.val[0]))));
    vst1q_s8(&dst[12*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x0.val[1]), vget_high_s32(x4.val[1]))));
    vst1q_s8(&dst[13*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x1.val[1]), vget_high_s32(x5.val[1]))));
    vst1q_s8(&dst[14*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x2.val[1]), vget_high_s32(x6.val[1]))));
    vst1q_s8(&dst[15*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x3.val[1]), vget_high_s32(x7.val[1]))));

    // clang-format on
  }

  struct UnquantizeAndAddBiasAndWrite {
    UnquantizeAndAddBiasAndWrite(float unquant_multiplier, const float *input_bias_prepared)
        : unquant_multiplier_(unquant_multiplier), input_bias_prepared_(input_bias_prepared) {}

    void operator()(const int32_t *input, Index rows_A, Index cols_B, float *output) const {
      // Set all registers in lane from same scalar value.
      float32x4_t multiplier = vdupq_n_f32(unquant_multiplier_);
      const int32x4_t *Input = reinterpret_cast<const int32x4_t *>(input);
      const int32x4_t *InputEnd = reinterpret_cast<const int32x4_t *>(input + rows_A * cols_B);
      float32x4_t *Output = reinterpret_cast<float32x4_t *>(output);

      while(Input != InputEnd) {
        // Bias cycles every column for addition.
        const float32x4_t *Bias = reinterpret_cast<const float32x4_t *>(input_bias_prepared_);

        // InputEnd needs to be determined to end the while loop below.
        const int32x4_t *RowEnd = reinterpret_cast<const int32x4_t *>(
            reinterpret_cast<const int32_t *>(Input) + cols_B);

        while(Input != RowEnd) {
          // Operation happening for 4-elements together:
          // output = [int32_t]input * [float]quant_mult + [float]bias;
          float32x4_t floatInput = vcvtq_f32_s32(*Input++);
          float32x4_t unquantized = vmulq_f32(floatInput, multiplier);
          *Output++ = vaddq_f32(unquantized, *Bias++);
        }
      }
    }

  private:
    float unquant_multiplier_;
    const float *input_bias_prepared_;
  };

  struct UnquantizeAndWrite {
    explicit UnquantizeAndWrite(float unquant_multiplier)
        : unquant_multiplier_(unquant_multiplier) {}

    void operator()(const int32_t *input, Index rows_A, Index cols_B, float *output) const {
      // Set all registers in lane from same scalar value.
      float32x4_t multiplier = vdupq_n_f32(unquant_multiplier_);
      const int32x4_t *Input = reinterpret_cast<const int32x4_t *>(input);
      const int32x4_t *InputEnd = reinterpret_cast<const int32x4_t *>(input + rows_A * cols_B);
      float32x4_t *Output = reinterpret_cast<float32x4_t *>(output);

      while(Input != InputEnd) {
        // Bias cycles every column for addition.

        // InputEnd needs to be determined to end the while loop below.
        const int32x4_t *RowEnd = reinterpret_cast<const int32x4_t *>(
            reinterpret_cast<const int32_t *>(Input) + cols_B);

        while(Input != RowEnd) {
          // Operation happening for 4-elements together:
          // output = [int32_t]input * [float]quant_mult + [float]bias;
          float32x4_t floatInput = vcvtq_f32_s32(*Input++);
          float32x4_t unquantized = vmulq_f32(floatInput, multiplier);
          *Output++ = unquantized;
        }
      }
    }

  private:
    float unquant_multiplier_;
  };
};

#endif

/*
 * The following nomenclature comes from intgemm. The current state of code is to keep the
 * intgemm_interface.h diff minimal. There are possibly better abstractions.
 */
struct IntgemmViaRuy {
  // Convert compile time errors into run-time ABORTS. This allows bringing in only int8_t and
  // select functions that are required to create a path which will run while not achieving
  // parity with intgemm.
  template <class T>
  struct IntBase {
    using Type = T;
    static void Quantize(const float *, Type *, float, Index) { ABORT("Quantize unsupported"); }

    static void PrepareA(const float *input,
                         Type *output,
                         float quant_mult,
                         Index rows,
                         Index cols) {
      ABORT("PrepareA Unsupported");
    }

    static void PrepareB(const float *, Type *, float, Index, Index) {
      ABORT("PrepareB Unsupported");
    }
    static void PrepareBQuantizedTransposed(const Type *, Type *, Index, Index) {
      ABORT("PrepareBQuantizedTransposed Unsupported");
    }
    static void PrepareBTransposed(const float *, Type *, float, Index, Index) {
      ABORT("PrepareBTransposed Unsupported");
    }
    static void SelectColumnsB(const Type *, Type *, Index, const Index *, const Index *) {
      ABORT("SelectColumnsB Unsupported");
    }

    template <class Callback>
    static void Multiply(const Type *A_prepared,
                         const Type *B_prepared,
                         float *output,
                         Index rows_A,
                         Index width,
                         Index cols_B,
                         Callback callback) {
      ABORT("Multiply (A*B) Unsupported");
    }
  };

  // Intgemm nomenclature expects Int8. Missing functions are ABORTs.
  struct Int8 : IntBase<int8_t> {
    using Type = int8_t;
    static void PrepareBQuantizedTransposed(const Type *input,
                                            Type *output,
                                            Index rows,
                                            Index cols) {
      std::memcpy(output, input, /*count=*/sizeof(Type) * (rows * cols));
    }

    static void PrepareBTransposed(const float *input,
                                   Type *output,
                                   float quant_mult,
                                   Index rows,
                                   Index cols) {
      Preprocess<kHighestPath>::quantize(input, output, quant_mult, rows, cols);
    }

    static void PrepareA(const float *input,
                         int8_t *output,
                         float quant_mult,
                         Index rows,
                         Index cols) {
      Preprocess<kHighestPath>::quantize(input, output, quant_mult, rows, cols);
    }

    static void SelectColumnsB(const Type *input,
                               Type *output,
                               Index width,
                               const Index *cols,
                               const Index *cols_end) {
      // B_prepared is expected to be col-major, for our implementation via ruy. If
      // col-major we can memcpy the respective column entries as they're
      // sequential. There are width=rows entries.
      Index num_cols = static_cast<Index>(std::distance(cols, cols_end));
      for(Index c = 0; c < num_cols; ++c) {
        std::memcpy(&(output[c * width]), &(input[cols[c] * width]), width);
      }
    }

    // We don't have callback an no-op capability here yet. Multiply is kept similar to Mozilla
    // specification and there are overloads with and without bias to avoid an if inside. This
    // method corresponds to the one with bias.
    // output = A*B + bias
    template <class Callback>
    static void Multiply(const Type *input_A_prepared,
                         const Type *input_B_prepared,
                         float *output,
                         Index rows_A,
                         Index width,
                         Index cols_B,
                         Callback callback) {
      // It is expected that somehow we have managed to call all prepare by the time
      // we are here, with inputs (prepared) in int8_t. All that's left to do is use
      // ruy for multiply and then start with the reverse ops to get to fp32.

      // Use ruy to multiply.
      // The following is adapted from
      // https://github.com/google/ruy/blob/878283640de7946a43053e8ebf4f15114fbc9156/example/example.cc#L129-L152

      ruy::Context context;
      ruy::Matrix<std::int8_t> lhs;
      ruy::MakeSimpleLayout(rows_A, width, ruy::Order::kRowMajor, lhs.mutable_layout());
      lhs.set_data(input_A_prepared);

      // PRINT_MATRIX_DEBUG(input_A_prepared, rows_A, width, Order::RowMajor);

      ruy::Matrix<std::int8_t> rhs;
      ruy::MakeSimpleLayout(width, cols_B, ruy::Order::kColMajor, rhs.mutable_layout());
      rhs.set_data(input_B_prepared);

      // PRINT_MATRIX_DEBUG(input_B_prepared, width, cols_B, Order::ColMajor);

      ruy::Matrix<std::int32_t> dst;
      ruy::MakeSimpleLayout(rows_A, cols_B, ruy::Order::kRowMajor, dst.mutable_layout());

      std::int32_t *dest_ptr = reinterpret_cast<std::int32_t *>(output);
      dst.set_data(dest_ptr);

      // When Dst is int32, mul_params is unused.
      ruy::MulParams<std::int32_t, std::int32_t> mul_params;
      ruy::Mul(lhs, rhs, mul_params, &context, &dst);

      callback(dest_ptr, rows_A, cols_B, output);
    }
  };

  // Int16 support is currently missing.
  struct Int16 : IntBase<int16_t> {
    using Type = int16_t;
  };

  template <class T>
  static T MaxAbsolute(const T *begin, const T *end) {
    T result = 0;
    for(auto p = begin; p < end; ++p) {
      result = std::max(result, std::abs(*p));
    }
    return result;
  }
};

}  // namespace integer
}  // namespace cpu
}  // namespace marian
