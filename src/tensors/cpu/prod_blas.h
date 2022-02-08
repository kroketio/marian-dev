#pragma once
#if MKL_FOUND
    #include <mkl.h>
#elif BLAS_FOUND
    #include <cblas.h>
#elif USE_ONNX_SGEMM
    #include "3rd_party/onnxjs/src/wasm-ops/gemm.h"
#endif

inline void sgemm(bool transA,
                  bool transB,
                  int rows_a,
                  int rows_b,
                  int width,
                  float alpha,
                  float* a,
                  int lda,
                  float* b,
                  int ldb,
                  float beta,
                  float* c,
                  int ldc) {
#if BLAS_FOUND
        cblas_sgemm(CblasRowMajor,
                    transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans,
                    rows_a,
                    rows_b,
                    width,
                    alpha,
                    a,
                    lda,
                    b,
                    ldb,
                    beta,
                    c,
                    ldc);
#elif USE_ONNX_SGEMM
        gemm_f32_imp(transA, transB, rows_a, rows_b, width, alpha, a, b, beta, c);
#else
    transA; transB; rows_a; rows_b; width; alpha; a; lda; b; ldb; beta; c; ldc; // make compiler happy
    ABORT("Marian must be compiled with a BLAS library");
#endif
}
