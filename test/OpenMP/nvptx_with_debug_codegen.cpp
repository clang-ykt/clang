// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited | FileCheck %s
// expected-no-diagnostics

// CHECK: , i8 addrspace(1)* noalias %{{.+}}, i1 {{.+}}
// CHECK: , i8* noalias %
// CHECK: , i8* dereferenceable(1) %
// CHECK: , i8 addrspace(1)* noalias %
// CHECK: , i8 addrspace(1)* noalias %
// CHECK: , i8* dereferenceable(1) %
// CHECK: , i8* dereferenceable(1) %
// CHECK: , i8 addrspace(1)* noalias %
// CHECK: , i8 addrspace(1)* noalias %
// CHECK: , i8* dereferenceable(1) %
// CHECK: , i8* dereferenceable(1) %
// CHECK: distinct !DICompileUnit(
int a;
bool bb;

int main() {
  int b[10][10];
  int c[10][10][10];
#pragma omp target teams firstprivate(a, b) map(tofrom                \
                                                : c [0:5]) map(tofrom \
                                                               : bb) if (a)
#pragma omp distribute parallel for
  for (unsigned j = 0; j < 10; ++j) {
    int &f = c[1][1][1];
    int d = 15;
    a = 5;
    b[0][a] = 10;
    c[0][0][a] = 11;
    b[0][a] = c[0][0][a];
    bb &= b[0][a];
  }
#pragma omp target parallel firstprivate(a) map(tofrom         \
                                                : c, b) map(to \
                                                            : bb)
  {
    int d = 15;
    a = 5;
    b[0][a] = 10;
    c[0][0][a] = 11;
    b[0][a] = c[0][0][a];
    d = bb;
  }
#pragma omp target teams distribute parallel for map(tofrom              \
                                                     : a, c, b) map(from \
                                                                    : bb)
  for (unsigned j = 0; j < 10; ++j) {
    int &f = a;
    int d = 15;
    a = 5;
    b[0][a] = 10;
    c[0][0][a] = 11;
    b[0][a] = c[0][0][a];
    bb = b[0][a];
  }
  return 0;
}
