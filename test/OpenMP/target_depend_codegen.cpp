// Test host codegen only.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[ANON_T:%.+]] = type { i[[SZ:32|64]]* }
// CHECK: [[TASK_T_PRIV:%.+]] = type { [[TASK_T:%.+]], [[TASK_PRIVS_T:%.+]] }
// CHECK: [[TASK_T]] = type { i8*, {{.+}} }
// CHECK: [[TASK_PRIVS_T]] = type { [1 x i8*], [1 x i8*], [1 x i[[SZ:32|64]]] }
// CHECK: [[SIZES:@.+]] = private {{.+}} constant [1 x i[[SZ]]] [i[[SZ]] {{.+}}]
// CHECK: [[TYPES:@.+]] = private {{.+}} constant [1 x i64] [i64 {{.+}}]

// CHECK: define {{.*}}[[MAIN:@.+]](
int main() {
  int a, b;
  #pragma omp target depend(inout:a) map(tofrom:b)
  {
    b++;
  }
}
// CHECK: [[A_LOC:%.+]] = alloca i32,
// CHECK: [[B_LOC:%.+]] = alloca i32,
// CHECK: [[BASEPTRS:%.+]] = alloca [1 x i8*],
// CHECK: [[PTRS:%.+]] = alloca [1 x i8*],
// CHECK: [[AGG_CAPT:%.+]] = alloca [[ANON_T]],
// CHECK: [[BASEPTRS_GEP0:%.+]] = getelementptr {{.+}} [1 x i8*], [1 x i8*]* [[BASEPTRS]], {{.+}} 0, {{.+}} 0
// CHECK: [[BASEPTRS_GEP0_CAST:%.+]] = bitcast i8** [[BASEPTRS_GEP0]] to i32**
// CHECK: store {{.+}}* [[B_LOC]], {{.+}}** [[BASEPTRS_GEP0_CAST]],
// CHECK: [[PTRS_GEP0:%.+]] = getelementptr {{.+}} [1 x i8*], [1 x i8*]* [[PTRS]], {{.+}} 0, {{.+}} 0
// CHECK: [[PTRS_GEP0_CAST:%.+]] = bitcast i8** [[PTRS_GEP0]] to i32**
// CHECK: store {{.+}}* [[B_LOC]], {{.+}}** [[PTRS_GEP0_CAST]],
// CHECK: [[BASEPTRS_TOCPY:%.+]] = getelementptr {{.+}} [1 x i8*], [1 x i8*]* [[BASEPTRS]], {{.+}} 0, {{.+}} 0
// CHECK: [[PTRS_TOCPY:%.+]] = getelementptr {{.+}} [1 x i8*], [1 x i8*]* [[PTRS]], {{.+}} 0, {{.+}} 0
// CHECK: [[AGG_CAPT_GEP0:%.+]] = getelementptr {{.+}} [[ANON_T]], [[ANON_T]]* [[AGG_CAPT]], {{.+}} 0, {{.+}} 0
// CHECK: store {{.+}}* [[B_LOC]], {{.+}}** [[AGG_CAPT_GEP0]],
// CHECK: [[TASK_TV_I8:%.+]] = call {{.+}} @__kmpc_omp_target_task_alloc({{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}} [[OMP_T_OULINED:@.+]] to {{.+}}, {{.+}})
// CHECK: [[TASK_TVAR:%.+]] = bitcast i8* [[TASK_TV_I8]] to [[TASK_T_PRIV]]*

// COPY captured argument array in task shareds
// CHECK: [[TASK_TVAR_GEP0:%.+]] = getelementptr {{.+}} [[TASK_T_PRIV]], [[TASK_T_PRIV]]* [[TASK_TVAR]], {{.+}} 0, {{.+}} 0
// CHECK: [[TASK_T_GEP0:%.+]] = getelementptr {{.+}} [[TASK_T]], [[TASK_T]]* [[TASK_TVAR_GEP0]], {{.+}} 0, {{.+}} 0
// CHECK: [[TASK_T_VAL:%.+]] = load i8*, i8** [[TASK_T_GEP0]],
// CHECK: [[AGG_CAPT_CAST:%.+]] = bitcast [[ANON_T]]* [[AGG_CAPT]] to {{.+}}
// CHECK: call void @llvm.memcpy.{{.+}}({{.+}}* [[TASK_T_VAL]], {{.+}}* [[AGG_CAPT_CAST]], {{.+}})

// COPY BASEPTRS in PRIVATES
// CHECK: [[TASK_PRIV_VAR:%.+]] = getelementptr {{.+}} [[TASK_T_PRIV]], [[TASK_T_PRIV]]* [[TASK_TVAR]], {{.+}} 0, {{.+}} 1
// CHECK: [[TASK_PRIV_GEP0:%.+]] = getelementptr {{.+}} [[TASK_PRIVS_T]], [[TASK_PRIVS_T]]* [[TASK_PRIV_VAR]], {{.+}} 0, {{.+}} 0
// CHECK: [[TASK_PRIV_GEP0_CAST:%.+]] = bitcast [1 x i8*]* [[TASK_PRIV_GEP0]] to i8*
// CHECK: [[BASEPTRS_TOCPY_CAST:%.+]] = bitcast i8** [[BASEPTRS_TOCPY]] to i8*
// CHECK: call void @llvm.memcpy.{{.+}}(i8* [[TASK_PRIV_GEP0_CAST]], i8* [[BASEPTRS_TOCPY_CAST]], {{.+}})
  
// COPY PTRS in PRIVATES
// CHECK: [[TASK_PRIV_GEP1:%.+]] = getelementptr {{.+}} [[TASK_PRIVS_T]], [[TASK_PRIVS_T]]* [[TASK_PRIV_VAR]], {{.+}} 0, {{.+}} 1
// CHECK: [[TASK_PRIV_GEP1_CAST:%.+]] = bitcast [1 x i8*]* [[TASK_PRIV_GEP1]] to i8*
// CHECK: [[PTRS_TOCPY_CAST:%.+]] = bitcast i8** [[PTRS_TOCPY]] to i8*
// CHECK: call void @llvm.memcpy.{{.+}}(i8* [[TASK_PRIV_GEP1_CAST]], i8* [[PTRS_TOCPY_CAST]], {{.+}})

// COPY SIZES in PRIVATES
// CHECK: [[TASK_PRIV_GEP2:%.+]] = getelementptr {{.+}} [[TASK_PRIVS_T]], [[TASK_PRIVS_T]]* [[TASK_PRIV_VAR]], {{.+}} 0, {{.+}} 2
// CHECK: [[TASK_PRIV_GEP2_CAST:%.+]] = bitcast [1 x i[[SZ]]]* [[TASK_PRIV_GEP2]] to i8*
// CHECK: call void @llvm.memcpy.{{.+}}(i8* [[TASK_PRIV_GEP2_CAST]], i8* bitcast ([1 x i[[SZ]]]* [[SIZES]] to i8*), {{.+}})

// CHECK: {{.+}} = call {{.+}} @__kmpc_omp_task_with_deps(%ident_t* @0, i32 [[GTID:%[^,]+]], {{.+}})
// CHECK: call i32 @__kmpc_omp_taskwait(%ident_t* @0, i32 [[GTID]])
// CHECK: ret {{.+}}
// CHECK: }

// CHECK: define {{.+}} void [[MAP_FUN:@.+]]([[TASK_PRIVS_T]]* {{.+}}, [1 x {{.+}}*]** {{.+}}, [1 x {{.+}}*]** {{.+}}, [1 x i[[SZ]]]** {{.+}}) {{.+}}
// CHECK: [[ADDR:%.+]] = alloca [[TASK_PRIVS_T]]*,
// CHECK: [[BASES_L:%.+]] = alloca [1 x {{.+}}*]**,
// CHECK: [[PTRS_L:%.+]] = alloca [1 x {{.+}}*]**,
// CHECK: [[SIZES_L:%.+]] = alloca [1 x i[[SZ]]]**,
// CHECK: store [[TASK_PRIVS_T]]* {{.+}}, [[TASK_PRIVS_T]]** [[ADDR]],
// CHECK: store [1 x {{.+}}*]** {{.+}}, [1 x {{.+}}*]*** [[BASES_L]],
// CHECK: store [1 x {{.+}}*]** {{.+}}, [1 x {{.+}}*]*** [[PTRS_L]],
// CHECK: store [1 x i[[SZ]]]** {{.+}}, [1 x i[[SZ]]]*** [[SIZES_L]],
// CHECK: [[ADDR_V:%.+]] = load [[TASK_PRIVS_T]]*, [[TASK_PRIVS_T]]** [[ADDR]],
// CHECK: [[ADDR_V_GEP0:%.+]] = getelementptr {{.+}} [[TASK_PRIVS_T]], [[TASK_PRIVS_T]]* [[ADDR_V]], {{.+}} 0, {{.+}} 0
// CHECK: [[BASES_L_LD:%.+]] = load [1 x {{.+}}*]**, [1 x {{.+}}*]*** [[BASES_L]],
// CHECK: store [1 x {{.+}}*]* [[ADDR_V_GEP0]], [1 x {{.+}}*]** [[BASES_L_LD]],

// CHECK: [[ADDR_V_GEP1:%.+]] = getelementptr {{.+}} [[TASK_PRIVS_T]], [[TASK_PRIVS_T]]* [[ADDR_V]], {{.+}} 0, {{.+}} 1
// CHECK: [[PTRS_L_LD:%.+]] = load [1 x {{.+}}*]**, [1 x {{.+}}*]*** [[PTRS_L]],
// CHECK: store [1 x {{.+}}*]* [[ADDR_V_GEP1]], [1 x {{.+}}*]** [[PTRS_L_LD]],

// CHECK: [[ADDR_V_GEP2:%.+]] = getelementptr {{.+}} [[TASK_PRIVS_T]], [[TASK_PRIVS_T]]* [[ADDR_V]], {{.+}} 0, {{.+}} 2
// CHECK: [[SIZES_L_LD:%.+]] = load [1 x i[[SZ]]]**, [1 x i[[SZ]]]*** [[SIZES_L]],
// CHECK: store [1 x i[[SZ]]]* [[ADDR_V_GEP2]], [1 x i[[SZ]]]** [[SIZES_L_LD]],
// CHECK: ret void


// CHECK: define {{.*}}[[OMP_T_OULINED]](
// CHECK: [[PRIVS_ADDR_L:%.+]] = alloca i8*,
// CHECK: [[COPY_FN_L:%.+]] = alloca void ({{.+}})*,
// CHECK: [[BASES_L:%.+]] = alloca [1 x {{.+}}*]*,
// CHECK: [[PTRS_L:%.+]] = alloca [1 x {{.+}}*]*,
// CHECK: [[SIZES_L:%.+]] = alloca [1 x i[[SZ]]]*,
// CHECK: [[TASK_T_L:%.+]] = alloca [[TASK_T_PRIV]]*,
// CHECK: store [[TASK_T_PRIV]]* {{.+}}, [[TASK_T_PRIV]]** [[TASK_T_L]],

// call mapping function
// CHECK: [[TASK_PARM:%.+]] = load [[TASK_T_PRIV]]*, [[TASK_T_PRIV]]** [[TASK_T_L]],
// CHECK: [[PRIVS_PTR:%.+]] = getelementptr {{.+}} [[TASK_T_PRIV]], [[TASK_T_PRIV]]* [[TASK_PARM]], {{.+}} 0, {{.+}} 1
// CHECK: [[PRIVS_BCAST:%.+]] = bitcast [[TASK_PRIVS_T]]* [[PRIVS_PTR]] to i8*
// CHECK: store i8* [[PRIVS_BCAST]], i8** [[PRIVS_ADDR_L]],
// CHECK: store {{.+}} @{{.+}} to {{.+}} [[COPY_FN_L]],
// CHECK: [[COPY_FN_REF:%.+]] = load void ({{.+}})*, void ({{.+}})** [[COPY_FN_L]],
// CHECK: [[PRIVS_REF:%.+]] = load i8*, i8** [[PRIVS_ADDR_L]],
// CHECK: call void ({{.+}}) [[COPY_FN_REF]](i8* [[PRIVS_REF]], [1 x {{.+}}*]** [[BASES_L]], [1 x {{.+}}*]** [[PTRS_L]], [1 x i[[SZ]]]** [[SIZES_L]])

// actual target invocation
// CHECK: [[BASES_PTR:%.+]] = load [1 x {{.+}}*]*, [1 x {{.+}}*]** [[BASES_L]],
// CHECK: [[PTRS:%.+]] = load [1 x {{.+}}*]*, [1 x {{.+}}*]** [[PTRS_L]],
// CHECK: [[SIZES_PTR:%.+]] = load [1 x i[[SZ]]]*, [1 x i[[SZ]]]** [[SIZES_L]],
// CHECK: [[BASES_GEP:%.+]] = getelementptr {{.+}} [1 x {{.+}}*], [1 x {{.+}}*]* [[BASES_PTR]], {{.+}} 0, {{.+}} 0
// CHECK: [[PTRS_GEP:%.+]] = getelementptr {{.+}} [1 x {{.+}}*], [1 x {{.+}}*]* [[PTRS]], {{.+}} 0, {{.+}} 0
// CHECK: [[SIZES_GEP:%.+]] = getelementptr {{.+}} [1 x i[[SZ]]], [1 x i[[SZ]]]* [[SIZES_PTR]], {{.+}} 0, {{.+}} 0

// CHECK: [[TGT_RET:%.+]] = call{{.+}} @__tgt_target({{.+}}, {{.+}}, {{.+}}, i8** [[BASES_GEP]], i8** [[PTRS_GEP]], i[[SZ]]* [[SIZES_GEP]], {{.+}})
// CHECK: ret{{.+}}

#endif
