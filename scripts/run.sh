#!/bin/bash

NDK=/mnt/Storage/jphilip/android-ndk-r23b
ABI=arm64-v8a
MINSDK_VERSION=28
CUSTOM_MODULE_PATH=/mnt/Storage/jphilip/marian-android/openblas-install/lib/cmake/openblas
ANDROID_PLATFORM=28

OTHER_ANDROID_ARGS=(
    -DANDROID_ARM_NEON=TRUE
)

OTHER_MARIAN_ARGS=(
    -DCOMPILE_CUDA=off
    -DCOMPILE_CPU=on
    -DCMAKE_HAVE_THREADS_LIBRARY=1
    -DCMAKE_USE_WIN32_THREADS_INIT=0
    -DCMAKE_USE_PTHREADS_INIT=1
    -DTHREADS_PREFER_PTHREAD_FLAG=ON
    -DBUILD_ARCH=armv8-a
    -DUSE_INTGEMM=off
    -DUSE_SIMDE=on
    -DUSE_RUY=on
    -DUSE_ONNX_SGEMM=on # For time being.
    -DCOMPILE_WITHOUT_EXCEPTIONS=on # Apparently this can reduce the binary size, let's see.
)
# Additionally list variables finally configured.
cmake -L \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DCMAKE_MODULE_PATH=$CUSTOM_MODULE_PATH \
    -DANDROID_TOOLCHAIN=clang \
    -DANDROID_ABI=$ABI \
    -DANDROID_PLATFORM=$ANDROID_PLATFORM \
    -DANDROID_NATIVE_API_LEVEL=$MINSDKVERSION \
    -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.8 \
    -DANDROID_STL=c++_static \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    "${OTHER_ANDROID_ARGS[@]}" "${OTHER_MARIAN_ARGS[@]}" \
    ..
