// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[X:@.+]] = global double
namespace A {
double x;
}
namespace B {
using A::x;
}

int foo() {
  double t_var = 0;
  int p_var = 0;
#pragma omp simd lastprivate(conditional: t_var) lastprivate(p_var)
  for (int i = 0; i < 2; ++i) {
    if (i % 9)
      t_var = i;
    p_var = i;
  }
  return 0;
}

int main() {
#pragma omp simd lastprivate(conditional: A::x, B::x)
  for (int i = 0; i < 2; ++i) {
    if (i % 29)
      A::x = i;
    if (i % 39)
      B::x = i;
  }
  return foo();
}

// CHECK: define i{{[0-9]+}} @{{.*}}foo{{.*}}()
// CHECK: [[T_VAR:%.+]] = alloca double,
// CHECK: [[P_VAR:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[CLP_IDX:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[CLP_VAR:%.+]] = alloca double,
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[OMP_IV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_CLP_IV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[T_VAR_PRIV:%.+]] = alloca double,
// CHECK: [[P_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},

// Initialize Conditional Lastprivate Indices.
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[CLP_IDX]]

// Check for update to conditional lastprivate index in loop body.
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IV]],
// CHECK: [[IV:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IV]],
// CHECK: [[CONV:%.+]] = sext i{{[0-9]+}} [[IV]] to i{{[0-9]+}}
// CHECK: store i{{[0-9]+}} [[CONV]], i{{[0-9]+}}* [[OMP_CLP_IV]],

// CHECK: store double %{{.+}}, double* [[T_VAR_PRIV]],
// CHECK: [[OMP_CLP_IV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_CLP_IV]]
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: [[CMP:%.+]] = icmp ugt i{{[0-9]+}} [[OMP_CLP_IV_VAL]], [[CLP_IDX_VAL]]
// CHECK: br i1 [[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]

// CHECK: [[THEN]]
// CHECK: [[T_VAR_PRIV_VAL:%.+]] = load double, double* [[T_VAR_PRIV]],
// CHECK: br label %[[END:.+]]

// CHECK: [[ELSE]]
// CHECK: [[CLP_VAL:%.+]] = load double, double* [[CLP_VAR]],
// CHECK: br label %[[END]]

// CHECK: [[END]]
// CHECK: [[CLP_PHI:%.+]] = phi double [ [[T_VAR_PRIV_VAL]], %[[THEN]] ], [ [[CLP_VAL]], %[[ELSE]] ]
// CHECK: store double [[CLP_PHI]], double* [[CLP_VAR]],

// CHECK: [[OMP_CLP_IV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_CLP_IV]]
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: [[CMP:%.+]] = icmp ugt i{{[0-9]+}} [[OMP_CLP_IV_VAL]], [[CLP_IDX_VAL]]
// CHECK: br i1 [[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]

// CHECK: [[THEN]]
// CHECK: [[OMP_CLP_IV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_CLP_IV]],
// CHECK: br label %[[END:.+]]

// CHECK: [[ELSE]]
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: br label %[[END]]

// CHECK: [[END]]
// CHECK: [[CLP_PHI:%.+]] = phi i{{[0-9]+}} [ [[OMP_CLP_IV_VAL]], %[[THEN]] ], [ [[CLP_IDX_VAL]], %[[ELSE]] ]
// CHECK: store i{{[0-9]+}} [[CLP_PHI]], i{{[0-9]+}}* [[CLP_IDX]],


// CHECK: [[CLP_MAX_VAL:%.+]] = load double, double* [[CLP_VAR]],
// CHECK: store double [[CLP_MAX_VAL]], double* [[T_VAR]],

// original p_var=private_p_var;
// CHECK: [[P_VAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[P_VAR_PRIV]],
// CHECK: store i{{[0-9]+}} [[P_VAR_VAL]], i{{[0-9]+}}* [[P_VAR]],
// CHECK: ret i32





// CHECK: define i{{[0-9]+}} @main()
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[CLP_IDX:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[CLP_VAR:%.+]] = alloca double,
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[OMP_IV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_CLP_IV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[X_VAR_PRIV:%.+]] = alloca double,

// Initialize Conditional Lastprivate Indices.
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[CLP_IDX]]

// Check for update to conditional lastprivate index in loop body.
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IV]],
// CHECK: [[IV:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IV]],
// CHECK: [[CONV:%.+]] = sext i{{[0-9]+}} [[IV]] to i{{[0-9]+}}
// CHECK: store i{{[0-9]+}} [[CONV]], i{{[0-9]+}}* [[OMP_CLP_IV]],

// CHECK: store double %{{.+}}, double* [[X_VAR_PRIV]],
// CHECK: [[OMP_CLP_IV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_CLP_IV]]
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: [[CMP:%.+]] = icmp ugt i{{[0-9]+}} [[OMP_CLP_IV_VAL]], [[CLP_IDX_VAL]]
// CHECK: br i1 [[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]

// CHECK: [[THEN]]
// CHECK: [[T_VAR_PRIV_VAL:%.+]] = load double, double* [[X_VAR_PRIV]],
// CHECK: br label %[[END:.+]]

// CHECK: [[ELSE]]
// CHECK: [[CLP_VAL:%.+]] = load double, double* [[CLP_VAR]],
// CHECK: br label %[[END]]

// CHECK: [[END]]
// CHECK: [[CLP_PHI:%.+]] = phi double [ [[T_VAR_PRIV_VAL]], %[[THEN]] ], [ [[CLP_VAL]], %[[ELSE]] ]
// CHECK: store double [[CLP_PHI]], double* [[CLP_VAR]],

// CHECK: [[OMP_CLP_IV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_CLP_IV]]
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: [[CMP:%.+]] = icmp ugt i{{[0-9]+}} [[OMP_CLP_IV_VAL]], [[CLP_IDX_VAL]]
// CHECK: br i1 [[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]

// CHECK: [[THEN]]
// CHECK: [[OMP_CLP_IV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_CLP_IV]],
// CHECK: br label %[[END:.+]]

// CHECK: [[ELSE]]
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: br label %[[END]]

// CHECK: [[END]]
// CHECK: [[CLP_PHI:%.+]] = phi i{{[0-9]+}} [ [[OMP_CLP_IV_VAL]], %[[THEN]] ], [ [[CLP_IDX_VAL]], %[[ELSE]] ]
// CHECK: store i{{[0-9]+}} [[CLP_PHI]], i{{[0-9]+}}* [[CLP_IDX]],


// CHECK: store double %{{.+}}, double* [[X_VAR_PRIV]],
// CHECK: [[OMP_CLP_IV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_CLP_IV]]
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: [[CMP:%.+]] = icmp ugt i{{[0-9]+}} [[OMP_CLP_IV_VAL]], [[CLP_IDX_VAL]]
// CHECK: br i1 [[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]

// CHECK: [[THEN]]
// CHECK: [[T_VAR_PRIV_VAL:%.+]] = load double, double* [[X_VAR_PRIV]],
// CHECK: br label %[[END:.+]]

// CHECK: [[ELSE]]
// CHECK: [[CLP_VAL:%.+]] = load double, double* [[CLP_VAR]],
// CHECK: br label %[[END]]

// CHECK: [[END]]
// CHECK: [[CLP_PHI:%.+]] = phi double [ [[T_VAR_PRIV_VAL]], %[[THEN]] ], [ [[CLP_VAL]], %[[ELSE]] ]
// CHECK: store double [[CLP_PHI]], double* [[CLP_VAR]],

// CHECK: [[OMP_CLP_IV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_CLP_IV]]
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: [[CMP:%.+]] = icmp ugt i{{[0-9]+}} [[OMP_CLP_IV_VAL]], [[CLP_IDX_VAL]]
// CHECK: br i1 [[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]

// CHECK: [[THEN]]
// CHECK: [[OMP_CLP_IV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_CLP_IV]],
// CHECK: br label %[[END:.+]]

// CHECK: [[ELSE]]
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: br label %[[END]]

// CHECK: [[END]]
// CHECK: [[CLP_PHI:%.+]] = phi i{{[0-9]+}} [ [[OMP_CLP_IV_VAL]], %[[THEN]] ], [ [[CLP_IDX_VAL]], %[[ELSE]] ]
// CHECK: store i{{[0-9]+}} [[CLP_PHI]], i{{[0-9]+}}* [[CLP_IDX]],


// Check for final copying of conditional private values back to original vars.
// CHECK: [[CLP_MAX_VAL:%.+]] = load double, double* [[CLP_VAR]],
// CHECK: store double [[CLP_MAX_VAL]], double* [[X]],

// CHECK: ret i32

#endif

