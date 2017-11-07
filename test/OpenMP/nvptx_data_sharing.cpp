// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

void teams_distribute(int *arr) {
#pragma omp target teams map(arr[0:10])
  {
    int a = 1;
#pragma omp distribute
    for (int i = 0; i < 10; i++) {
      arr[i] += a;
    }
  }
}

// CHECK: define {{.*}}void {{@__omp_offloading_.+teams_distribute.+}}(
// CHECK: br i1{{.*}}, label {{%?}}[[MASTER:.+]],
// CHECK: [[MASTER]]:


void teams_if_distribute(int *arr, int b) {
#pragma omp target teams map(arr[0:10])
  {
    int a = 1;
    if (b == 1) {
#pragma omp distribute
      for (int i = 0; i < 10; i++) {
        arr[i] += a;
      }
    } else {
#pragma omp distribute
      for (int i = 0; i < 10; i++) {
        arr[i] += b * a;
      }
    }
  }
}

// CHECK: define {{.*}}void {{@__omp_offloading_.+teams_if_distribute.+}}(
// CHECK: br i1{{.*}}, label {{%?}}[[MASTER:.+]],
// CHECK: [[MASTER]]:


void teams_for_distribute(int *arr, int b) {
#pragma omp target teams map(arr[0:10])
  {
    int a = 1;
    for (int o = 0; o < b; o++) {
#pragma omp distribute
      for (int i = 0; i < 10; i++) {
        arr[i] += a;
      }
    }
  }
}

// CHECK: define {{.*}}void {{@__omp_offloading_.+teams_for_distribute.+}}(
// CHECK: br i1{{.*}}, label {{%?}}[[MASTER:.+]],
// CHECK: [[MASTER]]:
