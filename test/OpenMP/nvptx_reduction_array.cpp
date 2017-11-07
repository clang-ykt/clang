// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

void spmd() {
   double red[2];

#pragma omp target map(tofrom: red[0:2])
#pragma omp teams distribute parallel for simd reduction(+: red[0:2])
  for (int i = 0; i < 2; i++) {
    red[i] += 1;
  }
}

// CHECK: alloca [1 x i8*]
// CHECK: alloca [2 x double]


void teams_distribute() {
  double red[2];

#pragma omp target map(tofrom: red[0:2])
#pragma omp teams reduction(+: red[0:2])
  {
    int a = 2;
#pragma omp distribute
    for (int i = 0; i < 2; i++) {
      red[i] += a;
    }
  }
}

// CHECK: alloca [1 x i8*]
// CHECK: alloca [2 x double]


void teams_distribute_parallel_for() {
  double red[2];

#pragma omp target map(tofrom: red[0:2])
#pragma omp teams reduction(+: red[0:2])
  {
    int a = 2;
#pragma omp distribute parallel for reduction(+: red[0:2])
    for (int i = 0; i < 2; i++) {
      red[i] += a;
    }
  }
}

// CHECK: alloca [1 x i8*]
// CHECK: alloca [2 x double]
