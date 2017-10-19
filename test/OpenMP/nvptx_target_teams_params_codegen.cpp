// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited | FileCheck %s
// expected-no-diagnostics

int a;

// CHECK: @[[TGTFN:__omp_offloading_.+]]_property = weak constant %struct.__openmp_nvptx_target_property_ty { i8 1, i32 1, i32 64 }

int main() {
  double a;
#pragma omp target teams map(a) reduction(+: a)
  {
    a+=1;
  }
  return 0;
}

// CHECK: define internal void @[[TGTFN]]_impl{{[^(]+}}(double* noalias {{%[a-zA-Z0-9]+}})
// CHECK: call i32 @__omp_kernel_initialization()

// CHECK: define weak void @[[TGTFN]](double* dereferenceable({{[0-9]+}}) {{[^,]+}}, i8* {{[^)]+}})
// CHECK: [[SCRATCH_ADDR:%.+]] = alloca i8*
// CHECK: store i8* {{[^,]+}}, i8** [[SCRATCH_ADDR]]
// CHECK: [[SCRATCH:%.+]] = load i8*, i8** [[SCRATCH_ADDR]]
// CHECK: call void @__kmpc_kernel_init_params(i8* [[SCRATCH]])
// CHECK: call void @[[TGTFN]]_impl{{[^(]+}}(double* {{[^)]+}})

