// RUN: %clang_cc1 -verify -fopenmp -fopenmp-implicit-declare-target -x c++ -std=c++11 -triple powerpc64le-ibm-linux-gnu -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-implicit-declare-target -x c++ -std=c++11 -triple powerpc64le-ibm-linux-gnu -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t -o - | FileCheck %s
// expected-no-diagnostics

struct S {
  int s;
  S() {}
  S(const S &s) {}
  ~S() {}
};

// CHECK: define {{.*}} @__omp_offloading
int main(int argc, char** argv) {
  S s;
  auto lambda = [=]() { return argc + s.s; };
// CHECK: call {{.*}} [[LAMBDA_CCONSTR:@.+main[^(]+]](
#pragma omp target firstprivate(lambda)
// CHECK: = call {{.*}} i32 {{.*}}@{{.+}}main{{[^(]+}}
  lambda();
// CHECK: call {{.*}} [[LAMBDA_DESTR:@.+main.+]](
// CHECK: ret
  return 0;
}

// CHECK: define internal {{.+}} [[LAMBDA_CCONSTR]](
// CHECK: define internal {{.+}} [[LAMBDA_DESTR]](
