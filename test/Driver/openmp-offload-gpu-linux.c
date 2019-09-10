///
/// Perform driver tests for OpenMP offloading on Linux systems
///

// UNSUPPORTED: system-windows

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target

/// Check cubin file generation and partial linking with ld
// RUN:   %clang -### -target powerpc64le-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -no-canonical-prefixes -save-temps %s -c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PTXAS-CUBIN-BUNDLING %s

// CHK-PTXAS-CUBIN-BUNDLING: clang{{.*}}" "-o" "[[PTX:.*\.s]]"
// CHK-PTXAS-CUBIN-BUNDLING-NEXT: ptxas{{.*}}" "--output-file" "[[CUBIN:.*\.cubin]]" {{.*}}"[[PTX]]"
// CHK-PTXAS-CUBIN-BUNDLING: fatbinary{{.*}}" "--create=[[FATBIN:.*\.fatbin]]" "
// CHK-PTXAS-CUBIN-BUNDLING-SAME: --embedded-fatbin=[[FATBINC:.*\.fatbin.c]]" "
// CHK-PTXAS-CUBIN-BUNDLING-SAME: --cmdline=--compile-only" "--image=profile={{.*}}[[PTX]]" "
// CHK-PTXAS-CUBIN-BUNDLING-SAME: --image=profile={{.*}}file=[[CUBIN]]" "--cuda" "--device-c"
// CHK-PTXAS-CUBIN-BUNDLING: clang++{{.*}}" "-c" "-o" "[[HOSTDEV:.*\.o]]"{{.*}}" "[[FATBINC]]" "-D__NV_MODULE_ID=
// CHK-PTXAS-CUBIN-BUNDLING-NOT: clang-offload-bundler
// CHK-PTXAS-CUBIN-BUNDLING: ld" "-r" "[[HOSTDEV]]" "{{.*}}.o" "-o" "{{.*}}.o"

/// ###########################################################################

/// Check object file unbundling is not happening when skipping bundler
// RUN:   touch %t.o
// RUN:   %clang -### -target powerpc64le-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -no-canonical-prefixes -save-temps %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-CUBIN-UNBUNDLING-NVLINK %s

/// Use DAG to ensure that object file has not been unbundled.
// CHK-CUBIN-UNBUNDLING-NVLINK-DAG: nvlink{{.*}}" {{.*}}"[[OBJ:.*\.o]]"
// CHK-CUBIN-UNBUNDLING-NVLINK-DAG: ld{{.*}}" {{.*}}"[[OBJ]]"

/// ###########################################################################

/// Check object file generation is not happening when skipping bundler
// RUN:   touch %t1.o
// RUN:   touch %t2.o
// RUN:   %clang -### -no-canonical-prefixes -target powerpc64le-unknown-linux-gnu -fopenmp=libomp \
// RUN:          -fopenmp-targets=nvptx64-nvidia-cuda %t1.o %t2.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TWOCUBIN %s
/// Check cubin file generation and usage by nvlink when toolchain has BindArchAction
// RUN:   %clang -### -no-canonical-prefixes -target x86_64-apple-darwin17.0.0 -fopenmp=libomp \
// RUN:          -fopenmp-targets=nvptx64-nvidia-cuda %t1.o %t2.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TWOCUBIN %s

// CHK-TWOCUBIN: nvlink{{.*}}openmp-offload-{{.*}}.o" "{{.*}}openmp-offload-{{.*}}.o"
