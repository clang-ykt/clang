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

int main() {
  static double s_var = 0;
  float t_var = 0;
#pragma omp parallel
#pragma omp sections lastprivate(t_var) lastprivate(conditional: s_var)
  {
#pragma omp section
    {
      t_var = 2;
      s_var = 1;
    }
#pragma omp section
    {
      t_var = 3;
    }
  }
#pragma omp parallel
#pragma omp sections lastprivate(conditional: A::x, B::x)
  {
#pragma omp section
    {
      A::x = 10;
    }
#pragma omp section
    {
      B::x = 20;
    }
#pragma omp section
    {
    }
  }
  return 0;
}

// CHECK: define i{{[0-9]+}} @main()
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 2, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, float*, double*)* [[MAIN_MICROTASK:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[MAIN_MICROTASK1:@.+]] to void
// CHECK: ret

// CHECK: define internal void [[MAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}},
// CHECK: [[CLP_IDX:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[CLP_VAR:%.+]] = alloca double,
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[OMP_IV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_CLP_IV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[T_VAR_PRIV:%.+]] = alloca float,
// CHECK: [[S_VAR_PRIV:%.+]] = alloca double,
// CHECK: [[CLP_REDUCE_ARRAY:%.+]] = alloca [1 x i{{[0-9]+}}]
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// CHECK: [[T_VAR_REF:%.+]] = load float*, float** %
// CHECK: [[S_VAR_REF:%.+]] = load double*, double** %
// Initialize Conditional Lastprivate Indices.
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[CLP_IDX]]
// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]

// CHECK: call void @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 %{{.+}}, i32 34, i32* [[IS_LAST_ADDR:%[^,]+]],

// Check for update to conditional lastprivate index in loop body.
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IV]],
// CHECK: [[IV:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IV]],
// CHECK-NEXT: [[CONV:%.+]] = sext i{{[0-9]+}} [[IV]] to i{{[0-9]+}}
// CHECK-NEXT: store i{{[0-9]+}} [[CONV]], i{{[0-9]+}}* [[OMP_CLP_IV]],
// CHECK-NEXT: [[IV:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IV]],
// CHECK-NEXT: switch i32 [[IV]]
// CHECK-NEXT: i32 0
// CHECK-NEXT: i32 1
// CHECK-NEXT: ]

// CHECK: store float 2.{{.+}}, float* [[T_VAR_PRIV]],
// CHECK: store double 1.{{.+}}, double* [[S_VAR_PRIV]],
// CHECK: [[OMP_CLP_IV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_CLP_IV]]
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: [[CMP:%.+]] = icmp ugt i{{[0-9]+}} [[OMP_CLP_IV_VAL]], [[CLP_IDX_VAL]]
// CHECK: br i1 [[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]

// CHECK: [[THEN]]
// CHECK: [[S_VAR_PRIV_VAL:%.+]] = load double, double* [[S_VAR_PRIV]],
// CHECK: br label %[[END:.+]]

// CHECK: [[ELSE]]
// CHECK: [[CLP_VAL:%.+]] = load double, double* [[CLP_VAR]],
// CHECK: br label %[[END]]

// CHECK: [[END]]
// CHECK: [[CLP_PHI:%.+]] = phi double [ [[S_VAR_PRIV_VAL]], %[[THEN]] ], [ [[CLP_VAL]], %[[ELSE]] ]
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

// CHECK: call void @__kmpc_for_static_fini(

// Check for final copying of conditional private values back to original vars.
// CHECK: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
// CHECK: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0

// CHECK: [[GEP:%.+]] = getelementptr inbounds [1 x i{{[0-9]+}}], [1 x i{{[0-9]+}}]* [[CLP_REDUCE_ARRAY]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: store i{{[0-9]+}} [[CLP_IDX_VAL]], i{{[0-9]+}}* [[GEP]],
// CHECK: [[VOID_PTR_ARRAY:%.+]] = bitcast [1 x i{{[0-9]+}}]* [[CLP_REDUCE_ARRAY]] to i8*
// CHECK: call void @__kmpc_reduce_conditional_lastprivate(%{{.+}}* @{{.+}}, i32 %{{.+}}, i32 1, i8* [[VOID_PTR_ARRAY]])
// CHECK: [[GEP:%.+]] = getelementptr inbounds [1 x i{{[0-9]+}}], [1 x i{{[0-9]+}}]* [[CLP_REDUCE_ARRAY]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[CLP_MAX_IDX:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GEP]],
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: [[CMP:%.+]] = icmp eq i64 [[CLP_IDX_VAL]], [[CLP_MAX_IDX]]
// CHECK: br i1 [[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]

// CHECK: [[THEN]]
// CHECK: [[CLP_MAX_VAL:%.+]] = load double, double* [[CLP_VAR]],
// CHECK: store double [[CLP_MAX_VAL]], double* [[S_VAR_REF]],


// Check for final copying of private values back to original vars.
// CHECK: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
// CHECK: [[LAST_THEN]]
// Actual copying.

// original t_var=private_t_var;
// CHECK: [[T_VAR_VAL:%.+]] = load float, float* [[T_VAR_PRIV]],
// CHECK: store float [[T_VAR_VAL]], float* [[T_VAR_REF]],

// CHECK: call void @__kmpc_barrier(
// CHECK: ret void



//
// CHECK: define internal void [[MAIN_MICROTASK1]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: [[CLP_IDX:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[CLP_VAR:%.+]] = alloca double,
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[OMP_IV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_CLP_IV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[X_VAR_PRIV:%.+]] = alloca double,
// CHECK-NOT: alloca double
// CHECK: [[CLP_REDUCE_ARRAY:%.+]] = alloca [1 x i{{[0-9]+}}]
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// Check for default initialization.
// CHECK-NOT: [[X_PRIV]]

// Initialize Conditional Lastprivate Indices.
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[CLP_IDX]]
// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call void @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 %{{.+}}, i32 34, i32* [[IS_LAST_ADDR:%[^,]+]],

// Check for update to conditional lastprivate index in loop body.
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IV]],
// CHECK: [[IV:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IV]],
// CHECK-NEXT: [[CONV:%.+]] = sext i{{[0-9]+}} [[IV]] to i{{[0-9]+}}
// CHECK-NEXT: store i{{[0-9]+}} [[CONV]], i{{[0-9]+}}* [[OMP_CLP_IV]],
// CHECK-NEXT: [[IV:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IV]],
// CHECK-NEXT: switch i32 [[IV]]
// CHECK-NEXT: i32 0
// CHECK-NEXT: i32 1
// CHECK-NEXT: i32 2
// CHECK-NEXT: ]

// CHECK: store double 1.{{.+}}, double* [[X_VAR_PRIV]],
// CHECK: [[OMP_CLP_IV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_CLP_IV]]
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: [[CMP:%.+]] = icmp ugt i{{[0-9]+}} [[OMP_CLP_IV_VAL]], [[CLP_IDX_VAL]]
// CHECK: br i1 [[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]

// CHECK: [[THEN]]
// CHECK: [[X_VAR_PRIV_VAL:%.+]] = load double, double* [[X_VAR_PRIV]],
// CHECK: br label %[[END:.+]]

// CHECK: [[ELSE]]
// CHECK: [[CLP_VAL:%.+]] = load double, double* [[CLP_VAR]],
// CHECK: br label %[[END]]

// CHECK: [[END]]
// CHECK: [[CLP_PHI:%.+]] = phi double [ [[X_VAR_PRIV_VAL]], %[[THEN]] ], [ [[CLP_VAL]], %[[ELSE]] ]
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

// CHECK: call void @__kmpc_for_static_fini(


// CHECK: [[GEP:%.+]] = getelementptr inbounds [1 x i{{[0-9]+}}], [1 x i{{[0-9]+}}]* [[CLP_REDUCE_ARRAY]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: store i{{[0-9]+}} [[CLP_IDX_VAL]], i{{[0-9]+}}* [[GEP]],
// CHECK: [[VOID_PTR_ARRAY:%.+]] = bitcast [1 x i{{[0-9]+}}]* [[CLP_REDUCE_ARRAY]] to i8*
// CHECK: call void @__kmpc_reduce_conditional_lastprivate(%{{.+}}* @{{.+}}, i32 %{{.+}}, i32 1, i8* [[VOID_PTR_ARRAY]])
// CHECK: [[GEP:%.+]] = getelementptr inbounds [1 x i{{[0-9]+}}], [1 x i{{[0-9]+}}]* [[CLP_REDUCE_ARRAY]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[CLP_MAX_IDX:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GEP]],
// CHECK: [[CLP_IDX_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CLP_IDX]],
// CHECK: [[CMP:%.+]] = icmp eq i64 [[CLP_IDX_VAL]], [[CLP_MAX_IDX]]
// CHECK: br i1 [[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]

// CHECK: [[THEN]]
// CHECK: [[CLP_MAX_VAL:%.+]] = load double, double* [[CLP_VAR]],
// CHECK: store double [[CLP_MAX_VAL]], double* [[X]],

// CHECK: call void @__kmpc_barrier(
// CHECK: ret void

#endif

