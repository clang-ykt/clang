// RUN:   %clang_cc1 -I%S/Inputs -ftrap=divz %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=DIVZ %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=divz -ftrap-exact %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=DIVZ-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=fp %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=FP %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=fp -ftrap-exact %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=FP-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=inexact %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=INEXACT %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=inexact -ftrap-exact %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=INEXACT-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=inv %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=INV %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=inv -ftrap-exact %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=INV-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=ovf %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=OVF %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=ovf -ftrap-exact %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=OVF-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=unf %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=UNF %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=unf -ftrap-exact %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=UNF-EXACT %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=none %s -E -o - | FileCheck -check-prefix=NONE %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=none -ftrap-exact %s -E -o - | FileCheck -check-prefix=NONE %s

// RUN:   %clang_cc1 -I%S/Inputs -ftrap=divz,inv,unf %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=MANY %s
// RUN:   %clang_cc1 -I%S/Inputs -ftrap=divz,inv,unf -ftrap-exact %s -E -o - | FileCheck -check-prefix=CHECK -check-prefix=MANY-EXACT %s


// CHECK: #pragma clang diagnostic push
// CHECK: #pragma GCC diagnostic push
// CHECK: #pragma clang diagnostic ignored "-Wall"
// CHECK: #pragma GCC diagnostic ignored "-Wall"
// NONE-NOT: #pragma clang diagnostic push
// NONE-NOT: #pragma GCC diagnostic push
// NONE-NOT: #pragma clang diagnostic ignored "-Wall"
// NONE-NOT: #pragma GCC diagnostic ignored "-Wall"

// CHECK: int feclearexcept(int);
// CHECK: int fetestexcept(int);
// DIVZ: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689start_____(void){feclearexcept(0|1);}
// DIVZ-EXACT: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689end_____(void){if(fetestexcept(0|1)) raise(8);}
// FP: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689start_____(void){feclearexcept(0|1|4|8);}
// FP-EXACT: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689end_____(void){if(fetestexcept(0|1|4|8)) raise(8);}
// INEXACT: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689start_____(void){feclearexcept(0|2);}
// INEXACT-EXACT: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689end_____(void){if(fetestexcept(0|2)) raise(8);}
// INV: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689start_____(void){feclearexcept(0|4);}
// INV-EXACT: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689end_____(void){if(fetestexcept(0|4)) raise(8);}
// OVF: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689start_____(void){feclearexcept(0|8);}
// OVF-EXACT: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689end_____(void){if(fetestexcept(0|8)) raise(8);}
// UNF: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689start_____(void){feclearexcept(0|16);}
// UNF-EXACT: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689end_____(void){if(fetestexcept(0|16)) raise(8);}
// MANY: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689start_____(void){feclearexcept(0|1|4|16);}
// MANY-EXACT: __attribute((used)) __attribute((no_instrument_function)) __attribute((nodebug)) __attribute((always_inline)) static void __FTRAP___INSTR__fun_12345689end_____(void){if(fetestexcept(0|1|4|16)) raise(8);}

// CHECK: #pragma GCC diagnostic pop
// CHECK: #pragma clang diagnostic pop
// NONE-NOT: #pragma GCC diagnostic pop
// NONE-NOT: #pragma clang diagnostic pop
