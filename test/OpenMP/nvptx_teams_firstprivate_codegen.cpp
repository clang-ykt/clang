// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -debug-info-kind=limited -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -debug-info-kind=limited -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER



template<typename tx, typename ty>
struct TT{
  tx X;
  ty Y;
};

// TCHECK: [[TT:%.+]] = type { i64, i8 }
// TCHECK: [[S1:%.+]] = type { double }

int foo(int n, double* ptr) {
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;
  
  #pragma omp target private(a)
  #pragma omp teams firstprivate(a)
  {
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
  // TCHECK:  [[ATA:%.+]] = alloca i32,
  // TCHECK:  [[ATE:%.+]] = alloca i32,
  // TCHECK-DAG:  store i{{[0-9]+}} [[ATA_VAL:%.+]], i{{[0-9]+}}* [[ATE]],
  // TCHECK-DAG:  [[ATA_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[ATA]],
  // TCHECK:  ret void
  

  // a is implictly firstprivate - copy the value from parameter to private teams variable
  #pragma omp target
  #pragma omp teams firstprivate(a)
  {
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[A_IN:%.+]])
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  [[A_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  ret void  

  #pragma omp target firstprivate(a)
  #pragma omp teams firstprivate(a)
  {
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{.+[0-9]}}(i{{[0-9]+}} [[A_IN:%.+]])
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK-64:  [[CONV:%.+]] = bitcast i{{[0-9]+}}* [[A_ADDR]] to i{{[0-9]+}}*
  // TCHECK-64:  [[A_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CONV]],
  // TCHECK-32:  [[A_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  ret void  

  #pragma omp target private(a)
  #pragma omp teams firstprivate(a)
  {
    a = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
  // TCHECK:  [[ATA:%.+]] = alloca i32,
  // TCHECK:  alloca i{{32|64}},
  // TCHECK:  [[ATE:%.+]] = alloca i32,
  // TCHECK-DAG:  store i{{[0-9]+}} [[ATA_VAL:%.+]], i{{[0-9]+}}* [[ATE]],
  // TCHECK-DAG:  [[ATA_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[ATA]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
  // TCHECK:  ret void

  #pragma omp target
  #pragma omp teams firstprivate(a)
  {
    a = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[A_IN:%.+]])
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  [[A_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
  // TCHECK:  ret void

  #pragma omp target firstprivate(a)
  #pragma omp teams firstprivate(a)
  {
    a = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[A_IN:%[^,^)]+]])
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  [[A_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],  
  // TCHECK:  store i{{[0-9]+}} [[A_IN_VAL]], i{{[0-9]+}}* [[ATE]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
  // TCHECK:  ret void  


  #pragma omp target private(a, aa)
  #pragma omp teams firstprivate(a,aa)
  {
    a = 1;
    aa = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
  // TCHECK:  [[ATA:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2TA:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2TE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK-DAG:  store i{{[0-9]+}} [[ATA_VAL:%.+]], i{{[0-9]+}}* [[ATE]],
  // TCHECK-DAG:  [[ATA_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[ATA]],
  // TCHECK-DAG:  store i{{[0-9]+}} [[A2TA_VAL:%.+]], i{{[0-9]+}}* [[A2TE]],
  // TCHECK-DAG:  [[A2TA_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2TA]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2TE]],
  // TCHECK:  ret void
  
  #pragma omp target
  #pragma omp teams firstprivate(a,aa)
  {
    a = 1;
    aa = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[A_IN:%.+]], i{{[0-9]+}} {{[a-z]*}} [[A2_IN:%[^,^)]+]])
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2TE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[A2_IN]], i{{[0-9]+}}* [[A2_ADDR]],
  // TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2_ADDR]],
  // TCHECK:  [[A_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[A_IN_VAL]], i{{[0-9]+}}* [[ATE]],
  // TCHECK:  [[A2_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[A2_IN_VAL]], i{{[0-9]+}}* [[A2TE]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2TE]],
  // TCHECK:  ret void
  
  #pragma omp target firstprivate(a, aa)
  #pragma omp teams firstprivate(a,aa)
  {
    a = 1;
    aa = 1;

    aa = a+1;
  }

  // check that we are not using the firstprivate parameter
  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[A_IN:%.+]], i{{[0-9]+}} {{[a-z]*}} [[A2_IN:%[^,^)]+]])
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2TE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[A2_IN]], i{{[0-9]+}}* [[A2_ADDR]],
  // TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2_ADDR]],
  // TCHECK:  [[A_IN_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[A_IN_VAL]], i{{[0-9]+}}* [[ATE]],
  // TCHECK:  [[A2_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2_ADDR]],
  // TCHECK: store i{{[0-9]+}} [[A2_IN_VAL]], i{{[0-9]+}}* [[A2TE]],
  // a = 1, aa = 1
  // TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
  // TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2TE]],

  // aa = a+1
  // TCHECK:  [[ATE_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[ATE]],
  // TCHECK:  [[ATE_INC:%.+]] = add{{.+}} i{{[0-9]+}} [[ATE_VAL]], 1
  // TCHECK:  [[ATE_CONV:%.+]] = trunc {{.+}} to
  // TCHECK:  store i{{[0-9]+}} [[ATE_CONV]], i{{[0-9]+}}* [[A2TE]],
  // TCHECK:  ret void

  #pragma omp target private(a, b, c, d)
  #pragma omp teams firstprivate(a, b, c, d)
  {
    a = 1;
    b[2] = 1.0;
    c[1][2] = 1.0;
    d.X = 1;
    d.Y = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
  // TCHECK:  [[ATA:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[BTA:%.+]] = alloca [10 x float],
  // TCHECK:  [[CTA:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[DTA:%.+]] = alloca [[TT]],
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[BTE:%.+]] = alloca [10 x float],
  // TCHECK:  [[CTE:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[DTE:%.+]] = alloca [[TT]],
  // TCHECK-DAG:  store i{{[0-9]+}} [[ATA_VAL:%.+]], i{{[0-9]+}}* [[ATE]],
  // TCHECK-DAG:  [[ATA_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[ATA]],
  // TCHECK:  [[BTE_CPY:%.+]] = bitcast [10 x float]* [[BTE]] to i8*
  // TCHECK:  [[BTA_CPY:%.+]] = bitcast [10 x float]* [[BTA]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[BTE_CPY]], i8* [[BTA_CPY]],{{.+}})
  // TCHECK:  [[CTE_CPY:%.+]] = bitcast [5 x [10 x double]]* [[CTE]] to i8*
  // TCHECK:  [[CTA_CPY:%.+]] = bitcast [5 x [10 x double]]* [[CTA]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[CTE_CPY]], i8* [[CTA_CPY]],{{.+}})
  // TCHECK:  [[DTE_CPY:%.+]] = bitcast [[TT]]* [[DTE]] to i8*
  // TCHECK:  [[DTA_CPY:%.+]] = bitcast [[TT]]* [[DTA]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[DTE_CPY]], i8* [[DTA_CPY]],{{.+}})

  // TCHECK:  store {{.+}} [[ATE]],
  // TCHECK:  [[B_IDX:%.+]] = getelementptr{{.+}} [[BTE]], {{.+}},
  // TCHECK:  store {{.+}} [[B_IDX]],
  // TCHECK:  [[C_IDX1:%.+]] = getelementptr {{.+}} [[CTE]], {{.+}}
  // TCHECK:  [[C_IDX2:%.+]] = getelementptr {{.+}} [[C_IDX1]], {{.+}}
  // TCHECK:  store {{.+}} [[C_IDX2]],
  // TCHECK:  [[D_X:%.+]] = getelementptr {{.+}} [[DTE]], {{.+}}
  // TCHECK:  store {{.+}} [[D_X]],
  // TCHECK:  [[D_Y:%.+]] = getelementptr {{.+}} [[DTE]], {{.+}}
  // TCHECK:  store {{.+}} [[D_Y]],
  
  #pragma omp target
  #pragma omp teams firstprivate(a, b, c, d)
  {
    a = 1;
    b[2] = 1.0;
    c[1][2] = 1.0;
    d.X = 1;
    d.Y = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[A_IN:%.+]], [10 x float]*{{.+}} [[B_IN:%.+]], [5 x [10 x double]]*{{.+}} [[C_IN:%.+]], [[TT]]*{{.+}} [[D_IN:%[^,^)]+]])
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[B_ADDR:%.+]] = alloca [10 x float]*,
  // TCHECK:  [[C_ADDR:%.+]] = alloca [5 x [10 x double]]*,
  // TCHECK:  [[D_ADDR:%.+]] = alloca [[TT]]*,
  // TCHECK:  [[B_ADDR2:%.+]] = alloca [10 x float]*,
  // TCHECK:  [[C_ADDR2:%.+]] = alloca [5 x [10 x double]]*,
  // TCHECK:  [[D_ADDR2:%.+]] = alloca [[TT]]*,
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[B:%.+]] = alloca [10 x float],
  // TCHECK:  [[C:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[D:%.+]] = alloca [[TT]],
  // TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  store [10 x float]* [[B_IN]], [10 x float]** [[B_ADDR]],
  // TCHECK:  store [5 x [10 x double]]* [[C_IN]], [5 x [10 x double]]** [[C_ADDR]],
  // TCHECK:  store [[TT]]* [[D_IN]], [[TT]]** [[D_ADDR]],
  // TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x float]*, [10 x float]** [[B_ADDR2]],
  // TCHECK:  [[C_ADDR_REF:%.+]] = load [5 x [10 x double]]*, [5 x [10 x double]]** [[C_ADDR2]],
  // TCHECK:  [[D_ADDR_REF:%.+]] = load %struct.TT*, %struct.TT** [[D_ADDR2]],
  
  // TCHECK:  [[A_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  [[B_CPY:%.+]] = bitcast [10 x float]* [[B]] to i8*
  // TCHECK:  [[B_IN_CPY:%.+]] = bitcast [10 x float]* [[B_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[B_CPY]], i8* [[B_IN_CPY]],{{.+}})
  // TCHECK:  [[C_CPY:%.+]] = bitcast [5 x [10 x double]]* [[C]] to i8*
  // TCHECK:  [[C_IN_CPY:%.+]] = bitcast [5 x [10 x double]]* [[C_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[C_CPY]], i8* [[C_IN_CPY]],{{.+}})
  // TCHECK:  [[D_CPY:%.+]] = bitcast [[TT]]* [[D]] to i8*
  // TCHECK:  [[D_IN_CPY:%.+]] = bitcast [[TT]]* [[D_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[D_CPY]], i8* [[D_IN_CPY]],{{.+}})
  
  // TCHECK:  store {{.+}} [[A]],
  // TCHECK:  [[B_IDX:%.+]] = getelementptr{{.+}} [[B]], {{.+}},
  // TCHECK:  store {{.+}} [[B_IDX]],
  // TCHECK:  [[C_IDX1:%.+]] = getelementptr {{.+}} [[C]], {{.+}}
  // TCHECK:  [[C_IDX2:%.+]] = getelementptr {{.+}} [[C_IDX1]], {{.+}}
  // TCHECK:  store {{.+}} [[C_IDX2]],
  // TCHECK:  [[D_X:%.+]] = getelementptr {{.+}} [[D]], {{.+}}
  // TCHECK:  store {{.+}} [[D_X]],
  // TCHECK:  [[D_Y:%.+]] = getelementptr {{.+}} [[D]], {{.+}}
  // TCHECK:  store {{.+}} [[D_Y]],

  int tmp;
  #pragma omp target firstprivate(a, b, c, d) map(tmp)
  #pragma omp teams firstprivate(a, b, c, d)
  {
    tmp = 1;
    a = 1;
    b[2] = 1.0;
    c[1][2] = 1.0;
    d.X = 1;
    d.Y = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[A_IN:%.+]], [10 x float]*{{.+}} [[B_IN:%.+]], [5 x [10 x double]]*{{.+}} [[C_IN:%.+]], [[TT]]*{{.+}} [[D_IN:%.+]], i{{[0-9]+}}{{.+}}* {{.+}} {{[^,^)]+}})
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[B_ADDR:%.+]] = alloca [10 x float]*,
  // TCHECK:  [[C_ADDR:%.+]] = alloca [5 x [10 x double]]*,
  // TCHECK:  [[D_ADDR:%.+]] = alloca [[TT]]*,
  // TCHECK:  [[B_ADDR2:%.+]] = alloca [10 x float]*,
  // TCHECK:  [[C_ADDR2:%.+]] = alloca [5 x [10 x double]]*,
  // TCHECK:  [[D_ADDR2:%.+]] = alloca [[TT]]*,
  // TCHECK:  [[BTA:%.+]] = alloca [10 x float],
  // TCHECK:  [[CTA:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[DTA:%.+]] = alloca [[TT]],
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[BTE:%.+]] = alloca [10 x float],
  // TCHECK:  [[CTE:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[DTE:%.+]] = alloca [[TT]],

  // TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  store [10 x float]* [[B_IN]], [10 x float]** [[B_ADDR]],
  // TCHECK:  store [5 x [10 x double]]* [[C_IN]], [5 x [10 x double]]** [[C_ADDR]],
  // TCHECK:  store [[TT]]* [[D_IN]], [[TT]]** [[D_ADDR]],
  // TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x float]*, [10 x float]** [[B_ADDR2]],
  // TCHECK:  [[C_ADDR_REF:%.+]] = load [5 x [10 x double]]*, [5 x [10 x double]]** [[C_ADDR2]],
  // TCHECK:  [[D_ADDR_REF:%.+]] = load %struct.TT*, %struct.TT** [[D_ADDR2]],
  // TCHECK:  [[B_CPY:%.+]] = bitcast [10 x float]* [[BTA]] to i8*
  // TCHECK:  [[B_IN_CPY:%.+]] = bitcast [10 x float]* [[B_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[B_CPY]], i8* [[B_IN_CPY]],{{.+}})
  // TCHECK:  [[C_CPY:%.+]] = bitcast [5 x [10 x double]]* [[CTA]] to i8*
  // TCHECK:  [[C_IN_CPY:%.+]] = bitcast [5 x [10 x double]]* [[C_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[C_CPY]], i8* [[C_IN_CPY]],{{.+}})
  // TCHECK:  [[D_CPY:%.+]] = bitcast [[TT]]* [[DTA]] to i8*
  // TCHECK:  [[D_IN_CPY:%.+]] = bitcast [[TT]]* [[D_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[D_CPY]], i8* [[D_IN_CPY]],{{.+}})
  // TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  [[ATA_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],  
  // TCHECK:  store i{{[0-9]+}} [[ATA_VAL]], i{{[0-9]+}}* [[A]],
  // TCHECK:  [[BTE_CPY:%.+]] = bitcast [10 x float]* [[BTE]] to i8*
  // TCHECK:  [[BTA_CPY:%.+]] = bitcast [10 x float]* [[BTA]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[BTE_CPY]], i8* [[BTA_CPY]],{{.+}})
  // TCHECK:  [[CTE_CPY:%.+]] = bitcast [5 x [10 x double]]* [[CTE]] to i8*
  // TCHECK:  [[CTA_CPY:%.+]] = bitcast [5 x [10 x double]]* [[CTA]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[CTE_CPY]], i8* [[CTA_CPY]],{{.+}})
  // TCHECK:  [[DTE_CPY:%.+]] = bitcast [[TT]]* [[DTE]] to i8*
  // TCHECK:  [[DTA_CPY:%.+]] = bitcast [[TT]]* [[DTA]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[DTE_CPY]], i8* [[DTA_CPY]],{{.+}})
  
  // TCHECK:  store {{.+}} [[A]],
  // TCHECK:  [[B_IDX:%.+]] = getelementptr{{.+}} [[BTE]], {{.+}},
  // TCHECK:  store {{.+}} [[B_IDX]],
  // TCHECK:  [[C_IDX1:%.+]] = getelementptr {{.+}} [[CTE]], {{.+}}
  // TCHECK:  [[C_IDX2:%.+]] = getelementptr {{.+}} [[C_IDX1]], {{.+}}
  // TCHECK:  store {{.+}} [[C_IDX2]],
  // TCHECK:  [[D_X:%.+]] = getelementptr {{.+}} [[DTE]], {{.+}}
  // TCHECK:  store {{.+}} [[D_X]],
  // TCHECK:  [[D_Y:%.+]] = getelementptr {{.+}} [[DTE]], {{.+}}
  // TCHECK:  store {{.+}} [[D_Y]],

  #pragma omp target
  #pragma omp teams firstprivate(ptr)
  {
    ptr[0]++;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(double* [[PTR_IN:%.+]])
  // TCHECK:  [[PTR_ADDR:%.+]] = alloca double*,
  // TCHECK:  [[PTR:%.+]] = alloca double*,
  // TCHECK:  store double* [[PTR_IN]], double** [[PTR_ADDR]],
  // TCHECK:  {{.+}} = load double*, double** [[PTR_ADDR]],
  // TCHECK:  [[PTR_ADDR_REF:%.+]] = load double*, double** [[PTR_ADDR]],
  // TCHECK:  store double* [[PTR_ADDR_REF]], double** [[PTR]],
  // TCHECK:  [[PTR_VAL:%.+]] = load double*, double** [[PTR]],
  // TCHECK:  [[PTR_VAL_0:%.+]] = getelementptr inbounds double, double* [[PTR_VAL]], i{{.+}} 0
  // TCHECK:  [[PTR_VAL_0_VAL:%.+]] = load double, double* [[PTR_VAL_0]],
  // TCHECK:  add double [[PTR_VAL_0_VAL]], 1.{{.+}}
  // TCHECK:  store double {{.+}}, double* [[PTR_VAL_0]],

  #pragma omp target firstprivate(ptr) map(tmp)
  #pragma omp teams firstprivate(ptr)
  {
    tmp = 1;
    ptr[0]++;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(double* [[PTR_IN:%.+]], i{{[0-9]+}}{{.+}}* {{.+}} {{[^,^)]+}})
  // TCHECK:  [[PTR_ADDR:%.+]] = alloca double*,
  // TCHECK:  [[PTR:%.+]] = alloca double*,
  // TCHECK:  store double* [[PTR_IN]], double** [[PTR_ADDR]],
  // TCHECK:  {{.+}} = load double*, double** [[PTR_ADDR]],
  // TCHECK:  [[PTR_ADDR_REF:%.+]] = load double*, double** [[PTR_ADDR]],
  // TCHECK:  store double* [[PTR_ADDR_REF]], double** [[PTR]],
  // TCHECK:  [[PTR_VAL:%.+]] = load double*, double** [[PTR]],
  // TCHECK:  getelementptr inbounds double, double* [[PTR_VAL:%.+]], i{{.+}} 0
  // TCHECK-NOT: store double {{.}}, double* [[PTR_ADDR]],
  return a;
}


template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

#pragma omp target private(a,aa,b)
#pragma omp teams firstprivate(a,aa,b)
  {
    a = 1;
    aa = 1;
    b[2] = 1;
  }

#pragma omp target
#pragma omp teams firstprivate(a,aa,b)
  {
    a = 1;
    aa = 1;
    b[2] = 1;
  }

int tmp[1];
#pragma omp target firstprivate(a,aa,b)
#pragma omp teams firstprivate(a,aa,b)
  {
    tmp[0] = 1;
    a = 1;
    aa = 1;
    b[2] = 1;
  }

  return a;
}

static
int fstatic(int n) {
  int a = 0;
  short aa = 0;
  char aaa = 0;
  int b[10];

#pragma omp target private(a,aa,aaa,b)
#pragma omp teams firstprivate(a,aa,aaa,b)
  {
    a = 1;
    aa = 1;
    aaa = 1;
    b[2] = 1;
  }

// TCHECK: define {{.*}}void @__omp_offloading_{{[^(]+}}()
// TCHECK:  [[ATA:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2TA:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3TA:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[BTA:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  alloca i{{[0-9]+}},
// TCHECK:  alloca i{{[0-9]+}},
// TCHECK:  alloca i{{[0-9]+}},
// TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2TE:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3TE:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[BTE:%.+]] = alloca [10 x i{{[0-9]+}}],

// TCHECK-DAG:  store i{{[0-9]+}} [[ATA_VAL:%.+]], i{{[0-9]+}}* [[ATE]],
// TCHECK-DAG:  [[ATA_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[ATA]],
// TCHECK-DAG:  store i{{[0-9]+}} [[A2TA_VAL:%.+]], i{{[0-9]+}}* [[A2TE]],
// TCHECK-DAG:  [[A2TA_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2TA]],
// TCHECK-DAG:  store i{{[0-9]+}} [[A3TA_VAL:%.+]], i{{[0-9]+}}* [[A3TE]],
// TCHECK-DAG:  [[A3TA_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A3TA]],

// TCHECK:  [[BTE_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[BTE]] to i8*
// TCHECK:  [[BTA_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[BTA]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[BTE_CPY]], i8* [[BTA_CPY]],{{.+}})

// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2TE]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A3TE]],
// TCHECK:  [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[BTE]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK:  ret void
 
#pragma omp target
#pragma omp teams firstprivate(a,aa,aaa,b)
  {
    a = 1;
    aa = 1;
    aaa = 1;
    b[2] = 1;
  }

// TCHECK: define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}}{{.+}} [[A_IN:%.+]], i{{[0-9]+}}{{.+}} [[A2_IN:%.+]], i{{[0-9]+}}{{.+}} [[A3_IN:%.+]], [10 x i{{[0-9]+}}]*{{.+}} [[B_IN:%.+]])
// TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B_ADDR:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK:  [[BTMP:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK:  [[A_ADDR2:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2_ADDR2:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3_ADDR2:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A2_IN]], i{{[0-9]+}}* [[A2_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A3_IN]], i{{[0-9]+}}* [[A3_ADDR]],
// TCHECK:  store [10 x i{{[0-9]+}}]* [[B_IN]], [10 x i{{[0-9]+}}]** [[B_ADDR]],
// TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[A3_ADDR2]],
// TCHECK:  [[A_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A_IN_VAL]], i{{[0-9]+}}* [[A]],
// TCHECK:  [[A2_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A2_IN_VAL]], i{{[0-9]+}}* [[A2]],
// TCHECK:  [[A3_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A3_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A3_IN_VAL]], i{{[0-9]+}}* [[A3]],
// TCHECK:  [[B_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B]] to i8*
// TCHECK:  [[B_IN_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_ADDR_REF]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[B_CPY]], i8* [[B_IN_CPY]],{{.+}})

// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A3]],
// TCHECK:  [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[B]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK:  ret void

int tmp[2];
#pragma omp target firstprivate(a,aa,aaa,b)
#pragma omp teams firstprivate(a,aa,aaa,b)
  {
    tmp[0] = 1;
    a = 1;
    aa = 1;
    aaa = 1;
    b[2] = 1;
  }

// TCHECK: define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[A_IN:%.+]], i{{[0-9]+}}{{.+}} [[A2_IN:%.+]], i{{[0-9]+}}{{.+}} [[A3_IN:%.+]], [10 x i{{[0-9]+}}]*{{.+}} [[B_IN:%.+]], [2 x i32]*{{.+}} {{[^,^)]+}})
// TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B_ADDR:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK:  [[B_ADDR2:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK:  [[BTA:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  alloca i{{[0-9]+}},
// TCHECK:  alloca i{{[0-9]+}},
// TCHECK:  [[A3_CAST:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3:%.+]] = alloca i{{[0-9]+}},  
// TCHECK:  [[BTE:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A2_IN]], i{{[0-9]+}}* [[A2_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A3_IN]], i{{[0-9]+}}* [[A3_ADDR]],
// TCHECK:  store [10 x i{{[0-9]+}}]* [[B_IN]], [10 x i{{[0-9]+}}]** [[B_ADDR]],

// TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x i{{[0-9]+}}]*, [10 x i{{[0-9]+}}]** [[B_ADDR2]],
// TCHECK:  [[B_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[BTA]] to i8*
// TCHECK:  [[B_IN_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_ADDR_REF]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[B_CPY]], i8* [[B_IN_CPY]],{{.+}})

// TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[A3_CAST]],
// TCHECK:  [[ATA_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[ATA_VAL]], i{{[0-9]+}}* [[A]],
// TCHECK:  [[A2TA_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A2TA_VAL]], i{{[0-9]+}}* [[A2]],
// TCHECK:  [[A3TA_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A3_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A3TA_VAL]], i{{[0-9]+}}* [[A3]],
// TCHECK:  [[BTE_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[BTE]] to i8*
// TCHECK:  [[BTA_IN_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[BTA]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[BTE_CPY]], i8* [[BTA_IN_CPY]],{{.+}})

// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A3]],
// TCHECK:  [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[BTE]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK:  ret void

  return a;
}

struct S1 {
  double a;

  int r1(int n){
    int b = n+1;
    short int c[2][5];

#pragma omp target private(b,c)
#pragma omp teams firstprivate(b,c)
    {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}([[S1]]* [[TH:%.+]])
  // TCHECK:  [[TH_ADDR:%.+]] = alloca [[S1]]*,
  // TCHECK:  [[BTA:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[CTA:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]],
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  [[BTE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[CTE:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]],
  // TCHECK:  store [[S1]]* [[TH]], [[S1]]** [[TH_ADDR]],
  // TCHECK:  [[TH_ADDR_REF:%.+]] = load [[S1]]*, [[S1]]** [[TH_ADDR]],

  // TCHECK-DAG:  store i{{[0-9]+}} [[BTA_VAL:%.+]], i{{[0-9]+}}* [[BTE]],
  // TCHECK-DAG:  [[BTA_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[BTA]],
  // TCHECK:  [[CTE_CPY:%.+]] = bitcast [2 x [5 x i{{[0-9]+}}]]* [[CTE]] to i8*
  // TCHECK:  [[CTA_CPY:%.+]] = bitcast [2 x [5 x i{{[0-9]+}}]]* [[CTA]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[CTE_CPY]], i8* [[CTA_CPY]],{{.+}})

  // this->a = (double)b + 1.5;
  // TCHECK: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[BTE]],
  // TCHECK: [[B_CONV:%.+]] = sitofp i{{[0-9]+}} [[B_VAL]] to double
  // TCHECK: [[NEW_A_VAL:%.+]] = fadd double [[B_CONV]], 1.5{{.+}}+00
  // TCHECK: [[A_FIELD:%.+]] = getelementptr inbounds [[S1]], [[S1]]* [[TH_ADDR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // TCHECK: store double [[NEW_A_VAL]], double* [[A_FIELD]],

  // c[1][1] = ++a;
  // TCHECK: [[A_FIELD4:%.+]] = getelementptr inbounds [[S1]], [[S1]]* [[TH_ADDR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // TCHECK: [[A_FIELD4_VAL:%.+]] = load double, double* [[A_FIELD4]],
  // TCHECK: [[A_FIELD_INC:%.+]] = fadd double [[A_FIELD4_VAL]], 1.0{{.+}}+00
  // TCHECK: store double [[A_FIELD_INC]], double* [[A_FIELD4]],  
  // TCHECK: [[A_FIELD_INC_CONV:%.+]] = fptosi double [[A_FIELD_INC]] to i{{[0-9]+}}
  // TCHECK: [[C_1_REF:%.+]] = getelementptr inbounds [2 x [5 x i{{[0-9]+}}]], [2 x [5 x i{{[0-9]+}}]]* [[CTE]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // TCHECK: [[C_1_1_REF:%.+]] = getelementptr inbounds [5 x i{{[0-9]+}}], [5 x i{{[0-9]+}}]* [[C_1_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // TCHECK: store i{{[0-9]+}} [[A_FIELD_INC_CONV]], i{{[0-9]+}}* [[C_1_1_REF]],
  // TCHECK: ret void
    
#pragma omp target
#pragma omp teams firstprivate(b,c)
    {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[B_IN:%.+]], [2 x [5 x i{{[0-9]+}}]]*{{.+}} [[C_IN:%.+]], [[S1]]* [[TH:%.+]])
  // TCHECK:  [[B_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[C_ADDR:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]]*,
  // TCHECK:  [[TH_ADDR:%.+]] = alloca [[S1]]*,
  // TCHECK:  [[C_ADDR2:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]]*,
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK:  [[B:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[C:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]],

  // TCHECK:  store i{{[0-9]+}} [[B_IN]], i{{[0-9]+}}* [[B_ADDR]],
  // TCHECK:  store [2 x [5 x i{{[0-9]+}}]]* [[C_IN]], [2 x [5 x i{{[0-9]+}}]]** [[C_ADDR]],
  // TCHECK:  store [[S1]]* [[TH]], [[S1]]** [[TH_ADDR]],

  // TCHECK:  [[C_ADDR_REF:%.+]] = load [2 x [5 x i{{[0-9]+}}]]*, [2 x [5 x i{{[0-9]+}}]]** [[C_ADDR2]],
  // TCHECK:  [[TH_ADDR_REF:%.+]] = load [[S1]]*, [[S1]]** [[TH_ADDR]],
  // TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_ADDR]],
  // TCHECK:  [[B_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[B_IN_VAL]], i{{[0-9]+}}* [[B]],
  // TCHECK:  [[C_CPY:%.+]] = bitcast [2 x [5 x i{{[0-9]+}}]]* [[C]] to i8*
  // TCHECK:  [[C_IN_CPY:%.+]] = bitcast [2 x [5 x i{{[0-9]+}}]]* [[C_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[C_CPY]], i8* [[C_IN_CPY]],{{.+}})

  // this->a = (double)b + 1.5;
  // TCHECK: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B]],
  // TCHECK: [[B_CONV:%.+]] = sitofp i{{[0-9]+}} [[B_VAL]] to double
  // TCHECK: [[NEW_A_VAL:%.+]] = fadd double [[B_CONV]], 1.5{{.+}}+00
  // TCHECK: [[A_FIELD:%.+]] = getelementptr inbounds [[S1]], [[S1]]* [[TH_ADDR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // TCHECK: store double [[NEW_A_VAL]], double* [[A_FIELD]],

  // c[1][1] = ++a;
  // TCHECK: [[A_FIELD4:%.+]] = getelementptr inbounds [[S1]], [[S1]]* [[TH_ADDR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // TCHECK: [[A_FIELD4_VAL:%.+]] = load double, double* [[A_FIELD4]],
  // TCHECK: [[A_FIELD_INC:%.+]] = fadd double [[A_FIELD4_VAL]], 1.0{{.+}}+00
  // TCHECK: store double [[A_FIELD_INC]], double* [[A_FIELD4]],  
  // TCHECK: [[A_FIELD_INC_CONV:%.+]] = fptosi double [[A_FIELD_INC]] to i{{[0-9]+}}
  // TCHECK: [[C_1_REF:%.+]] = getelementptr inbounds [2 x [5 x i{{[0-9]+}}]], [2 x [5 x i{{[0-9]+}}]]* [[C]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // TCHECK: [[C_1_1_REF:%.+]] = getelementptr inbounds [5 x i{{[0-9]+}}], [5 x i{{[0-9]+}}]* [[C_1_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // TCHECK: store i{{[0-9]+}} [[A_FIELD_INC_CONV]], i{{[0-9]+}}* [[C_1_1_REF]],
  // TCHECK: ret void

int tmp[2];
#pragma omp target firstprivate(b,c)
#pragma omp teams firstprivate(b,c)
    {
      tmp[0] = 1;
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

    return c[1][1] + (int)b;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[B_IN:%.+]], [2 x [5 x i{{[0-9]+}}]]*{{.+}} [[C_IN:%.+]], [2 x i32]*{{.+}} {{.+}}, [[S1]]* [[TH:%.+]])
  // TCHECK: [[B_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK: [[C_ADDR:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]]*,
  // TCHECK: [[TH_ADDR:%.+]] = alloca [[S1]]*,
  // TCHECK: [[C_ADDR2:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]]*,
  // TCHECK:  [[CTA:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]],
  // TCHECK:  alloca i{{[0-9]+}},
  // TCHECK: [[B:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[CTE:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]],

  // TCHECK:  store i{{[0-9]+}} [[B_IN]], i{{[0-9]+}}* [[B_ADDR]],
  // TCHECK:  store [2 x [5 x i{{[0-9]+}}]]* [[C_IN]], [2 x [5 x i{{[0-9]+}}]]** [[C_ADDR]],
  // TCHECK:  store [[S1]]* [[TH]], [[S1]]** [[TH_ADDR]],
  // TCHECK:  [[C_ADDR_REF:%.+]] = load [2 x [5 x i{{[0-9]+}}]]*, [2 x [5 x i{{[0-9]+}}]]** [[C_ADDR2]],
  // TCHECK:  [[TH_ADDR_REF:%.+]] = load [[S1]]*, [[S1]]** [[TH_ADDR]],
  // TCHECK:  [[C_CPY:%.+]] = bitcast [2 x [5 x i{{[0-9]+}}]]* [[CTA]] to i8*
  // TCHECK:  [[C_IN_CPY:%.+]] = bitcast [2 x [5 x i{{[0-9]+}}]]* [[C_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[C_CPY]], i8* [[C_IN_CPY]],{{.+}})

  // TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_ADDR]],
  // TCHECK:  [[B_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[B_IN_VAL]], i{{[0-9]+}}* [[B]],
  // TCHECK:  [[C_CPY:%.+]] = bitcast [2 x [5 x i{{[0-9]+}}]]* [[CTE]] to i8*
  // TCHECK:  [[C_IN_CPY:%.+]] = bitcast [2 x [5 x i{{[0-9]+}}]]* [[CTA]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[C_CPY]], i8* [[C_IN_CPY]],{{.+}})

  // this->a = (double)b + 1.5;
  // TCHECK: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B]],
  // TCHECK: [[B_CONV:%.+]] = sitofp i{{[0-9]+}} [[B_VAL]] to double
  // TCHECK: [[NEW_A_VAL:%.+]] = fadd double [[B_CONV]], 1.5{{.+}}+00
  // TCHECK: [[A_FIELD:%.+]] = getelementptr inbounds [[S1]], [[S1]]* [[TH_ADDR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // TCHECK: store double [[NEW_A_VAL]], double* [[A_FIELD]],

  // c[1][1] = ++a;
  // TCHECK: [[A_FIELD4:%.+]] = getelementptr inbounds [[S1]], [[S1]]* [[TH_ADDR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // TCHECK: [[A_FIELD4_VAL:%.+]] = load double, double* [[A_FIELD4]],
  // TCHECK: [[A_FIELD_INC:%.+]] = fadd double [[A_FIELD4_VAL]], 1.0{{.+}}+00
  // TCHECK: store double [[A_FIELD_INC]], double* [[A_FIELD4]],  
  // TCHECK: [[A_FIELD_INC_CONV:%.+]] = fptosi double [[A_FIELD_INC]] to i{{[0-9]+}}
  // TCHECK: [[C_1_REF:%.+]] = getelementptr inbounds [2 x [5 x i{{[0-9]+}}]], [2 x [5 x i{{[0-9]+}}]]* [[CTE]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // TCHECK: [[C_1_1_REF:%.+]] = getelementptr inbounds [5 x i{{[0-9]+}}], [5 x i{{[0-9]+}}]* [[C_1_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // TCHECK: store i{{[0-9]+}} [[A_FIELD_INC_CONV]], i{{[0-9]+}}* [[C_1_1_REF]],
  // TCHECK: ret void

};


int bar(int n, double* ptr){
  int a = 0;
  a += foo(n, ptr);
  S1 S;
  a += S.r1(n);
  a += fstatic(n);
  a += ftemplate<int>(n);

  return a;
}

// template
// TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
// TCHECK:  [[ATA:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2TA:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[BTA:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  alloca i{{[0-9]+}},
// TCHECK:  alloca i{{[0-9]+}},
// TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2TE:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[BTE:%.+]] = alloca [10 x i{{[0-9]+}}],

// TCHECK-DAG:  store i{{[0-9]+}} [[ATA_VAL:%.+]], i{{[0-9]+}}* [[ATE]],
// TCHECK-DAG:  [[ATA_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[ATA]],
// TCHECK-DAG:  store i{{[0-9]+}} [[A2TA_VAL:%.+]], i{{[0-9]+}}* [[A2TE]],
// TCHECK-DAG:  [[A2TA_VAL]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2TA]],

// TCHECK:  [[BTE_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[BTE]] to i8*
// TCHECK:  [[BTA_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[BTA]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[BTE_CPY]], i8* [[BTA_CPY]],{{.+}})

// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2TE]],
// TCHECK: [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[BTE]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK: ret void


// TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[A_IN:%.+]], i{{[0-9]+}}{{.+}} [[A2_IN:%.+]], [10 x i{{[0-9]+}}]*{{.+}} [[B_IN:%.+]])
// TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B_ADDR:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK:  [[B_ADDR2:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK:  alloca i{{[0-9]+}},
// TCHECK:  alloca i{{[0-9]+}},
// TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B:%.+]] = alloca [10 x i{{[0-9]+}}],

// TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A2_IN]], i{{[0-9]+}}* [[A2_ADDR]],
// TCHECK:  store [10 x i{{[0-9]+}}]* [[B_IN]], [10 x i{{[0-9]+}}]** [[B_ADDR]],
// TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x i{{[0-9]+}}]*, [10 x i{{[0-9]+}}]** [[B_ADDR2]],
// TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  [[A_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A_IN_VAL]], i{{[0-9]+}}* [[A]],
// TCHECK:  [[A2_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A2_IN_VAL]], i{{[0-9]+}}* [[A2]],
// TCHECK:  [[B_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B]] to i8*
// TCHECK:  [[B_IN_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_ADDR_REF]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[B_CPY]], i8* [[B_IN_CPY]],{{.+}})

// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2]],
// TCHECK: [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[B]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK: ret void

// TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}} [[A_IN:%.+]], i{{[0-9]+}}{{.+}} [[A2_IN:%.+]], [10 x i{{[0-9]+}}]*{{.+}} [[B_IN:%.+]], [1 x i32]* {{.+}} {{.+}})
// TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B_ADDR:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK:  [[B_ADDR2:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK:  [[BTA:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  alloca i{{[0-9]+}},
// TCHECK:  alloca i{{[0-9]+}},
// TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[BTE:%.+]] = alloca [10 x i{{[0-9]+}}],

// TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A2_IN]], i{{[0-9]+}}* [[A2_ADDR]],
// TCHECK:  store [10 x i{{[0-9]+}}]* [[B_IN]], [10 x i{{[0-9]+}}]** [[B_ADDR]],
// TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x i{{[0-9]+}}]*, [10 x i{{[0-9]+}}]** [[B_ADDR2]],
// TCHECK:  [[B_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[BTA]] to i8*
// TCHECK:  [[B_IN_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_ADDR_REF]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[B_CPY]], i8* [[B_IN_CPY]],{{.+}})

// TCHECK:  {{.+}} = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
// TCHECK: [[A_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A_IN_VAL]], i{{[0-9]+}}* [[A]],
// TCHECK:  [[A2_IN_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A2_IN_VAL]], i{{[0-9]+}}* [[A2]],
// TCHECK:  [[B_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[BTE]] to i8*
// TCHECK:  [[B_IN_CPY:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[BTA]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* [[B_CPY]], i8* [[B_IN_CPY]],{{.+}})

// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2]],
// TCHECK: [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[BTE]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK: ret void

#endif

