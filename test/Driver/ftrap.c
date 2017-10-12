// REQUIRES: clang-driver

// RUN:   %clang -### -ftrap=divz %s 2>&1 | FileCheck -check-prefix=DIVZ %s
// DIVZ: "-ftrap=divz"
// RUN:   %clang -### -ftrap=divz -ftrap-exact %s 2>&1 | FileCheck -check-prefix=DIVZ-EXACT %s
// DIVZ-EXACT: "-ftrap=divz" "-ftrap-exact"

// RUN:   %clang -### -ftrap=fp %s 2>&1 | FileCheck -check-prefix=FP %s
// FP: "-ftrap=fp"
// RUN:   %clang -### -ftrap=fp -ftrap-exact %s 2>&1 | FileCheck -check-prefix=FP-EXACT %s
// FP-EXACT: "-ftrap=fp" "-ftrap-exact"

// RUN:   %clang -### -ftrap=inexact %s 2>&1 | FileCheck -check-prefix=INEXACT %s
// INEXACT: "-ftrap=inexact"
// RUN:   %clang -### -ftrap=inexact -ftrap-exact %s 2>&1 | FileCheck -check-prefix=INEXACT-EXACT %s
// INEXACT-EXACT: "-ftrap=inexact" "-ftrap-exact"

// RUN:   %clang -### -ftrap=inv %s 2>&1 | FileCheck -check-prefix=INV %s
// INV: "-ftrap=inv"
// RUN:   %clang -### -ftrap=inv -ftrap-exact %s 2>&1 | FileCheck -check-prefix=INV-EXACT %s
// INV-EXACT: "-ftrap=inv" "-ftrap-exact"

// RUN:   %clang -### -ftrap=ovf %s 2>&1 | FileCheck -check-prefix=OVF %s
// OVF: "-ftrap=ovf"
// RUN:   %clang -### -ftrap=ovf -ftrap-exact %s 2>&1 | FileCheck -check-prefix=OVF-EXACT %s
// OVF-EXACT: "-ftrap=ovf" "-ftrap-exact"

// RUN:   %clang -### -ftrap=unf %s 2>&1 | FileCheck -check-prefix=UNF %s
// UNF: "-ftrap=unf"
// RUN:   %clang -### -ftrap=unf -ftrap-exact %s 2>&1 | FileCheck -check-prefix=UNF-EXACT %s
// UNF-EXACT: "-ftrap=unf" "-ftrap-exact"

// RUN:   %clang -### -ftrap=none %s 2>&1 | FileCheck -check-prefix=NONE %s
// NONE: "-ftrap=none"
// RUN:   %clang -### -ftrap=none -ftrap-exact %s 2>&1 | FileCheck -check-prefix=NONE-EXACT %s
// NONE-EXACT: "-ftrap=none" "-ftrap-exact"

// RUN:   %clang -### -ftrap=divz,inv,unf %s 2>&1 | FileCheck -check-prefix=MANY %s
// MANY: "-ftrap=divz,inv,unf"
// RUN:   %clang -### -ftrap=divz,inv,unf -ftrap-exact %s 2>&1 | FileCheck -check-prefix=MANY-EXACT %s
// MANY-EXACT: "-ftrap=divz,inv,unf" "-ftrap-exact"

