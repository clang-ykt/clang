// RUN:   %clang_cc1 -I%S/Inputs -ftrap=divz %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns | FileCheck -check-prefix=CHECK -check-prefix=DIVZ %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=divz -ftrap-exact %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o -  -disable-llvm-optzns | FileCheck -check-prefix=CHECK -check-prefix=DIVZ-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=fp %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=FP %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=fp -ftrap-exact %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=FP-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=inexact %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=INEXACT %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=inexact -ftrap-exact %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=INEXACT-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=inv %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=INV %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=inv -ftrap-exact %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=INV-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=ovf %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=OVF %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=ovf -ftrap-exact %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=OVF-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=unf %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=UNF %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=unf -ftrap-exact %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=UNF-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=none %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=NONE %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=none -ftrap-exact %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=NONE %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=divz,inv,unf %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=MANY %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=divz,inv,unf -ftrap-exact %s -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - -disable-llvm-optzns  | FileCheck -check-prefix=CHECK -check-prefix=MANY-EXACT %s

// CHECK: @n = common global float 0.0
// CHECK: @m = common global float 0.0
// CHECK: @llvm.used = appending global [2 x i8*] [i8* bitcast (void ()* @__FTRAP___INSTR__fun_12345689start_____ to i8*), i8* bitcast (void ()* @__FTRAP___INSTR__fun_12345689end_____ to i8*)],
// NONE-NOT: @llvm.used

// NONE-NOT: @__FTRAP___INSTR__fun_12345689start_____()
// CHECK: define internal void @__FTRAP___INSTR__fun_12345689start_____()
// DIVZ: call {{.*}}i32 @feclearexcept(i32{{.*}} 1)
// DIVZ-EXACT: call {{.*}}i32 @feclearexcept(i32{{.*}} 1)
// FP: call {{.*}}i32 @feclearexcept(i32{{.*}} 13)
// FP-EXACT: call {{.*}}i32 @feclearexcept(i32{{.*}} 13)
// INEXACT: call {{.*}}i32 @feclearexcept(i32{{.*}} 2)
// INEXACT-EXACT: call {{.*}}i32 @feclearexcept(i32{{.*}} 2)
// INV: call {{.*}}i32 @feclearexcept(i32{{.*}} 4)
// INV-EXACT: call {{.*}}i32 @feclearexcept(i32{{.*}} 4)
// OVF: call {{.*}}i32 @feclearexcept(i32{{.*}} 8)
// OVF-EXACT: call {{.*}}i32 @feclearexcept(i32{{.*}} 8)
// UNF: call {{.*}}i32 @feclearexcept(i32{{.*}} 16)
// UNF-EXACT: call {{.*}}i32 @feclearexcept(i32{{.*}} 16)
// MANY: call {{.*}}i32 @feclearexcept(i32{{.*}} 21)
// MANY-EXACT: call {{.*}}i32 @feclearexcept(i32{{.*}} 21)
// CHECK: ret void

// NONE-NOT: @__FTRAP___INSTR__fun_12345689end_____()
// CHECK: define internal void @__FTRAP___INSTR__fun_12345689end_____()
// DIVZ: call {{.*}}i32 @fetestexcept(i32{{.*}} 1)
// DIVZ-EXACT: call {{.*}}i32 @fetestexcept(i32{{.*}} 1)
// FP: call {{.*}}i32 @fetestexcept(i32{{.*}} 13)
// FP-EXACT: call {{.*}}i32 @fetestexcept(i32{{.*}} 13)
// INEXACT: call {{.*}}i32 @fetestexcept(i32{{.*}} 2)
// INEXACT-EXACT: call {{.*}}i32 @fetestexcept(i32{{.*}} 2)
// INV: call {{.*}}i32 @fetestexcept(i32{{.*}} 4)
// INV-EXACT: call {{.*}}i32 @fetestexcept(i32{{.*}} 4)
// OVF: call {{.*}}i32 @fetestexcept(i32{{.*}} 8)
// OVF-EXACT: call {{.*}}i32 @fetestexcept(i32{{.*}} 8)
// UNF: call {{.*}}i32 @fetestexcept(i32{{.*}} 16)
// UNF-EXACT: call {{.*}}i32 @fetestexcept(i32{{.*}} 16)
// MANY: call {{.*}}i32 @fetestexcept(i32{{.*}} 21)
// MANY-EXACT: call {{.*}}i32 @fetestexcept(i32{{.*}} 21)
// CHECK: icmp ne i32 %{{.+}}, 0
// CHECK: br i1
// CHECK: call signext i32 @raise(i32 signext 8)
// CHECK: ret void

// CHECK-LABEL: @main
volatile float n, m;
int main(int argc, char ** argv) {
// CHECK: alloca i32,
// CHECK: alloca i32,
// CHECK: alloca i8**,
// NONE: alloca i32,
// NONE: alloca i32,
// NONE: alloca i8**,
  double a = n / m;
// CHECK: alloca double,
// NONE: alloca double,
  int res = a + argc;
// CHECK: alloca i32,
// NONE: alloca i32,
// CHECK: call void @__FTRAP___INSTR__fun_12345689start_____()
// NONE-NOT: @__FTRAP___INSTR__fun_12345689start_____
// CHECK: load volatile float, float* @n,
// DIVZ-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// FP-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INEXACT-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INV-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// OVF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// UNF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// MANY-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// CHECK: load volatile float, float* @m,
// DIVZ-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// FP-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INEXACT-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INV-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// OVF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// UNF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// MANY-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// CHECK: fdiv float %
// DIVZ-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// FP-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INEXACT-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INV-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// OVF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// UNF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// MANY-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// CHECK: fpext float %{{.+}} to double
// DIVZ-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// FP-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INEXACT-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INV-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// OVF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// UNF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// MANY-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// CHECK: store double %
// CHECK: load double, double* %
// DIVZ-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// FP-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INEXACT-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INV-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// OVF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// UNF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// MANY-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// CHECK: load i32, i32* %
// CHECK: sitofp i32 %{{.+}} to double
// DIVZ-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// FP-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INEXACT-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INV-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// OVF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// UNF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// MANY-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// CHECK: fadd double %
// DIVZ-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// FP-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INEXACT-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// INV-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// OVF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// UNF-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// MANY-EXACT: call void @__FTRAP___INSTR__fun_12345689end_____()
// CHECK: fptosi double %{{.+}} to i32
// CHECK: store i32 %
// CHECK: load i32, i32* %
// CHECK: store i32 %
// CHECK: call void @__FTRAP___INSTR__fun_12345689end_____()
// NONE: load volatile float, float* @n,
// NONE: load volatile float, float* @m,
// NONE: fdiv float %
// NONE: fpext float %{{.+}} to double
// NONE: store double %
// NONE: load double, double* %
// NONE: load i32, i32* %
// NONE: sitofp i32 %{{.+}} to double
// NONE: fadd double %
// NONE: fptosi double %{{.+}} to i32
// NONE: store i32 %
// NONE: load i32, i32* %
// NONE-NOT: @__FTRAP___INSTR__fun_12345689end_____
// CHECK: ret i32
// NONE: ret i32
  return res;
}
