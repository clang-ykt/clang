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

int foo(int n) {
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;
  
  #pragma omp target private(a)
  #pragma omp teams private(a)
  {
  }

  // generate both private variables
  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2:%.+]] = alloca i{{[0-9]+}},
  // TCHECK-NOT: store {{.+}}, {{.+}} [[A]],
  // TCHECK:  ret void
  

  // a is implictly firstprivate
  #pragma omp target
  #pragma omp teams private(a)
  {
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // TCHECK-NOT: store {{.+}}, {{.+}} [[A]],
  // TCHECK:  ret void  

  #pragma omp target firstprivate(a)
  #pragma omp teams private(a)
  {
  }

  // because of firstprivate, a.addr and a parameter are
  // created. This is fine, as long as we do not use a.addr
  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}({{.+}})
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK-NOT: store {{.+}}, i{{[0-9]+}}* [[ATE]],
  // TCHECK:  ret void  

  #pragma omp target private(a)
  #pragma omp teams private(a)
  {
    a = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
  // TCHECK:  [[ATA:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
  // TCHECK-NOT:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATA]],
  // TCHECK:  ret void

  #pragma omp target
  #pragma omp teams private(a)
  {
    a = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
  // TCHECK:  ret void

  #pragma omp target firstprivate(a)
  #pragma omp teams private(a)
  {
    a = 1;
  }

  // check that we store in a without looking at the parameter
  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}({{.+}})
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
  // TCHECK:  ret void

  #pragma omp target private(a, aa)
  #pragma omp teams private(a,aa)
  {
    a = 1;
    aa = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
  // TCHECK:  [[ATA:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2TA:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2TE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2TE]],
  // TCHECK:  ret void
  
  #pragma omp target
  #pragma omp teams private(a,aa)
  {
    a = 1;
    aa = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2]],
  // TCHECK:  ret void
  
  #pragma omp target firstprivate(a, aa)
  #pragma omp teams private(a,aa)
  {
    a = 1;
    aa = 1;

    aa = a+1;
  }

  // check that we are not using the firstprivate parameter
  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}({{.+}})
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2TE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2TE]],  
  // TCHECK:  [[A_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[ATE]]
  // TCHECK:  [[A_INC:%.+]] = add{{.+}} i{{[0-9]+}} [[A_VAL]], 1
  // TCHECK:  [[CONV:%.+]] = trunc i{{[0-9]+}} [[A_INC]] to i{{[0-9]+}}
  // TCHECK:  store i{{[0-9]+}} [[CONV]], i{{[0-9]+}}* [[A2TE]]
  // TCHECK:  ret void

  #pragma omp target private(a, b,c, d)
  #pragma omp teams private(a, b, c, d)
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
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[BTE:%.+]] = alloca [10 x float],
  // TCHECK:  [[CTE:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[DTE:%.+]] = alloca [[TT]],
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
  #pragma omp teams private(a, b, c, d)
  {
    a = 1;
    b[2] = 1.0;
    c[1][2] = 1.0;
    d.X = 1;
    d.Y = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}()
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[B:%.+]] = alloca [10 x float],
  // TCHECK:  [[C:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[D:%.+]] = alloca [[TT]],
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

  #pragma omp target firstprivate(a, b, c, d)
  #pragma omp teams private(a, b, c, d)
  {
    a = 1;
    b[2] = 1.0;
    c[1][2] = 1.0;
    d.X = 1;
    d.Y = 1;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{[^(]+}}({{.+}})
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[B_ADDR:%.+]] = alloca [10 x float]*,
  // TCHECK:  [[C_ADDR:%.+]] = alloca [5 x [10 x double]]*,
  // TCHECK:  [[D_ADDR:%.+]] = alloca [[TT]]*,
  // TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[BTE:%.+]] = alloca [10 x float],
  // TCHECK:  [[CTE:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[DTE:%.+]] = alloca [[TT]],
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

  return a;
}


template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

#pragma omp target private(a,aa,b)
#pragma omp teams private(a,aa,b)
  {
    a = 1;
    aa = 1;
    b[2] = 1;
  }

#pragma omp target
#pragma omp teams private(a,aa,b)
  {
    a = 1;
    aa = 1;
    b[2] = 1;
  }

#pragma omp target firstprivate(a,aa,b)
#pragma omp teams private(a,aa,b)
  {
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
#pragma omp teams private(a,aa,aaa,b)
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
// TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2TE:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3TE:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[BTE:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2TE]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A3TE]],
// TCHECK:  [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[BTE]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK:  ret void
  
#pragma omp target
#pragma omp teams private(a,aa,aaa,b)
  {
    a = 1;
    aa = 1;
    aaa = 1;
    b[2] = 1;
  }

// TCHECK: define {{.*}}void @__omp_offloading_{{[^(]+}}()
// TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A3]],
// TCHECK:  [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[B]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK:  ret void

#pragma omp target firstprivate(a,aa,aaa,b)
#pragma omp teams private(a,aa,aaa,b)
  {
    a = 1;
    aa = 1;
    aaa = 1;
    b[2] = 1;
  }

// TCHECK: define {{.*}}void @__omp_offloading_{{[^(]+}}({{.+}})
// TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B_ADDR:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK:  [[ATE:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2TE:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3TE:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[BTE:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2TE]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A3TE]],
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
#pragma omp teams private(b,c)
    {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

  // TCHECK: define {{.*}}void @__omp_offloading_{{[^(]+}}([[S1]]* [[TH:%.+]])
  // TCHECK: [[TH_ADDR:%.+]] = alloca [[S1]]*,
  // TCHECK: [[BTA:%.+]] = alloca i{{[0-9]+}},
  // TCHECK: [[CTA:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]],
  // TCHECK: [[BTE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK: [[CTE:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]],
  // TCHECK: store [[S1]]* [[TH]], [[S1]]** [[TH_ADDR]],
  // TCHECK: [[TH_ADDR_REF:%.+]] = load [[S1]]*, [[S1]]** [[TH_ADDR]],

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
    
    int tmp;
#pragma omp target map(tmp)
#pragma omp teams private(b,c)
    {
      tmp = 1;
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

  // TCHECK: define {{.*}}void @__omp_offloading_{{[^(]+}}(i{{[0-9]+}}{{.+}}* {{.+}} {{[^,^)]+}}, [[S1]]* [[TH:%.+]])
  // TCHECK: [[TH_ADDR:%.+]] = alloca [[S1]]*,
  // TCHECK: [[B:%.+]] = alloca i{{[0-9]+}},
  // TCHECK: [[C:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]],
  // TCHECK: store [[S1]]* [[TH]], [[S1]]** [[TH_ADDR]],
  // TCHECK: [[TH_ADDR_REF:%.+]] = load [[S1]]*, [[S1]]** [[TH_ADDR]],

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

#pragma omp target firstprivate(b,c)
#pragma omp teams private(b,c)
    {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

    return c[1][1] + (int)b;
  }

  // TCHECK: define {{.*}}void @__omp_offloading_{{[^(]+}}([[S1]]* [[TH:%.+]], i{{[0-9]+}}{{.+}})
  // TCHECK: [[TH_ADDR:%.+]] = alloca [[S1]]*,
  // TCHECK: [[B_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK: [[C_ADDR:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]]*,
  // TCHECK: [[BTE:%.+]] = alloca i{{[0-9]+}},
  // TCHECK: [[CTE:%.+]] = alloca [2 x [5 x i{{[0-9]+}}]],
  // TCHECK: store [[S1]]* [[TH]], [[S1]]** [[TH_ADDR]],
  // TCHECK: [[TH_ADDR_REF:%.+]] = load [[S1]]*, [[S1]]** [[TH_ADDR]],

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

};


int bar(int n){
  int a = 0;
  a += foo(n);
  S1 S;
  a += S.r1(n);
  a += fstatic(n);
  a += ftemplate<int>(n);

  return a;
}

// template
// TCHECK: define {{.*}}void @__omp_offloading_{{[^(]+}}()
// TCHECK: [[ATA:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[A2TA:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[BTA:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK: [[ATE:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[A2TE:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[BTE:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2TE]],
// TCHECK: [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[BTE]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK: ret void

// TCHECK: define {{.*}}void @__omp_offloading_{{[^(]+}}()
// TCHECK: [[A:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[A2:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[B:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2]],
// TCHECK: [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[B]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK: ret void

// TCHECK: define {{.*}}void @__omp_offloading_{{[^(]+}}({{.+}})
// TCHECK: [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[B_ADDR:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK: [[ATE:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[A2TE:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[BTE:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[ATE]],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2TE]],
// TCHECK: [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[BTE]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK: ret void

#endif

