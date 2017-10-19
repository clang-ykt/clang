// Test host codegen only.
// RUN: %clang_cc1 -verify -fopenmp  -fopenmp-implicit-declare-target -fno-openmp-implicit-map-lambdas -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp  -fopenmp-implicit-declare-target -fno-openmp-implicit-map-lambdas -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp  -fopenmp-implicit-declare-target -fno-openmp-implicit-map-lambdas -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-implicit-declare-target -fno-openmp-implicit-map-lambdas  -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-implicit-declare-target -fno-openmp-implicit-map-lambdas  -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-implicit-declare-target -fno-openmp-implicit-map-lambdas  -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[ANON_T:%.+]] = type { i32*, i32* }
// CHECK: [[SIZES:@.+]] = private {{.+}} constant [1 x i[[PTRSZ:32|64]]] [i{{32|64}} {{8|16}}]
// CHECK: [[TYPES:@.+]] = private {{.+}} constant [1 x i64] [i64 673]

// CHECK: define {{.*}}[[MAIN:@.+]](
int main()
{
  int* p = new int[100];
  int* q = new int[100];
  auto body = [=](int i){
    p[i] = q[i];
  };

#pragma omp target teams distribute parallel for
  for (int i = 0; i < 100; ++i) {
    body(i);
  }
}

// CHECK: [[BASE_PTRS:%.+]] = alloca [1 x i8*]{{.+}}
// CHECK: [[PTRS:%.+]] = alloca [1 x i8*]{{.+}}

// storage of pointers in baseptrs and ptrs arrays
// CHECK: [[LOC_LAMBDA:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[CAST_LAMBDA:%.+]] = bitcast i8** [[LOC_LAMBDA]] to [[ANON_T]]**
// CHECK: store [[ANON_T]]* %{{.+}}, [[ANON_T]]** [[CAST_LAMBDA]]{{.+}}
// CHECK: [[LOC_LAMBDA:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PTRS]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[CAST_LAMBDA:%.+]] = bitcast i8** [[LOC_LAMBDA]] to [[ANON_T]]**
// CHECK: store [[ANON_T]]* %{{.+}}, [[ANON_T]]** [[CAST_LAMBDA]]{{.+}}

// actual target invocation
// CHECK: [[BASES_GEP:%.+]] = getelementptr {{.+}} [1 x {{.+}}*], [1 x {{.+}}*]* [[BASE_PTRS]], {{.+}} 0, {{.+}} 0
// CHECK: [[PTRS_GEP:%.+]] = getelementptr {{.+}} [1 x {{.+}}*], [1 x {{.+}}*]* [[PTRS]], {{.+}} 0, {{.+}} 0
// CHECK: {{%.+}} = call{{.+}} @__tgt_target_teams({{.+}}, {{.+}}, {{.+}}, i8** [[BASES_GEP]], i8** [[PTRS_GEP]], i[[PTRSZ]]* getelementptr inbounds ([1 x i{{.+}}], [1 x i{{.+}}]* [[SIZES]], i{{.+}} 0, i{{.+}} 0), i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[TYPES]], i{{.+}} 0, i{{.+}} 0), {{.+}}, {{.+}})

#endif
