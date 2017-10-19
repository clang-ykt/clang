// RUN: %clang_cc1 -verify -fopenmp -ast-dump %s | FileCheck %s
// expected-no-diagnostics

struct S {
  int a, b;
  S() {
#pragma omp for lastprivate(conditional: a)
    for (int i = 0; i < 0; ++i)
      if (i % 5) a = i;
  }
};

// CHECK:           `-OMPForDirective {{.+}} <line:7:9, col:44>
// CHECK-NEXT:        |-OMPLastprivateClause {{.+}} <col:17, col:44>
// CHECK-NEXT:        | `-DeclRefExpr {{.+}} <col:42> 'int' lvalue OMPCapturedExpr {{.+}} 'a' 'int &'
// CHECK-NEXT:        |-CapturedStmt {{.+}} <line:8:5, line:9:20>
// CHECK-NEXT:        | |-CapturedDecl {{.+}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT:        | | |-ForStmt {{.+}} <line:8:5, line:9:20>
// CHECK:             | | | `-IfStmt {{.+}} <line:9:7, col:20>
// CHECK-NEXT:        | | |   |-<<<NULL>>>
// CHECK-NEXT:        | | |   |-<<<NULL>>>
// CHECK-NEXT:        | | |   |-ImplicitCastExpr {{.+}} <col:11, col:15> '_Bool' <IntegralToBoolean>
// CHECK-NEXT:        | | |   | `-BinaryOperator {{.+}} <col:11, col:15> 'int' '%'
// CHECK-NEXT:        | | |   |   |-ImplicitCastExpr {{.+}} <col:11> 'int' <LValueToRValue>
// CHECK-NEXT:        | | |   |   | `-DeclRefExpr {{.+}} <col:11> 'int' lvalue Var {{.+}} 'i' 'int'
// CHECK-NEXT:        | | |   |   `-IntegerLiteral {{.+}} <col:15> 'int' 5
// CHECK-NEXT:        | | |   |-StmtExpr {{.+}} <col:20> 'void'
// CHECK-NEXT:        | | |   | `-CompoundStmt {{.+}} <col:20>
// CHECK-NEXT:        | | |   |   `-OMPLastprivateUpdateDirective {{.+}} <col:20>
// CHECK-NEXT:        | | |   |     |-OMPLastprivate_updateClause {{.+}} <col:20>
// CHECK-NEXT:        | | |   |     | `-BinaryOperator {{.+}} <line:7:17> 'unsigned long' lvalue ','
// CHECK-NEXT:        | | |   |     |   |-BinaryOperator {{.+}} <col:17> 'int' lvalue '='
// CHECK-NEXT:        | | |   |     |   | |-DeclRefExpr {{.+}} <col:17> 'int' lvalue Var {{.+}} [[CLP:'.*']] 'int'
// CHECK-NEXT:        | | |   |     |   | `-ImplicitCastExpr {{.+}} <col:9, col:17> 'int' <LValueToRValue>
// CHECK-NEXT:        | | |   |     |   |   `-ConditionalOperator {{.+}} <col:9, col:17> 'int' lvalue
// CHECK-NEXT:        | | |   |     |   |     |-BinaryOperator {{.+}} <col:9, col:17> '_Bool' '>'
// CHECK-NEXT:        | | |   |     |   |     | |-ImplicitCastExpr {{.+}} <col:9> 'unsigned long' <LValueToRValue>
// CHECK-NEXT:        | | |   |     |   |     | | `-DeclRefExpr {{.+}} <col:9> 'unsigned long' lvalue Var {{.+}} [[CLP_IV:'.*']] 'unsigned long'
// CHECK-NEXT:        | | |   |     |   |     | `-ImplicitCastExpr {{.+}} <col:17> 'unsigned long' <LValueToRValue>
// CHECK-NEXT:        | | |   |     |   |     |   `-DeclRefExpr {{.+}} <col:17> 'unsigned long' lvalue Var {{.+}} [[CLP_IDX:'.*']] 'unsigned long'
// CHECK-NEXT:        | | |   |     |   |     |-MemberExpr {{.+}} <col:42> 'int' lvalue ->a {{.+}}
// CHECK-NEXT:        | | |   |     |   |     | `-CXXThisExpr {{.+}} <col:42> 'struct S *' this
// CHECK-NEXT:        | | |   |     |   |     `-DeclRefExpr {{.+}} <col:17> 'int' lvalue Var {{.+}} [[CLP]] 'int'
// CHECK-NEXT:        | | |   |     |   `-BinaryOperator {{.+}} <col:17> 'unsigned long' lvalue '='
// CHECK-NEXT:        | | |   |     |     |-DeclRefExpr {{.+}} <col:17> 'unsigned long' lvalue Var {{.+}} [[CLP_IDX]] 'unsigned long'
// CHECK-NEXT:        | | |   |     |     `-ImplicitCastExpr {{.+}} <col:9, col:17> 'unsigned long' <LValueToRValue>
// CHECK-NEXT:        | | |   |     |       `-ConditionalOperator {{.+}} <col:9, col:17> 'unsigned long' lvalue
// CHECK-NEXT:        | | |   |     |         |-BinaryOperator {{.+}} <col:9, col:17> '_Bool' '>'
// CHECK-NEXT:        | | |   |     |         | |-ImplicitCastExpr {{.+}} <col:9> 'unsigned long' <LValueToRValue>
// CHECK-NEXT:        | | |   |     |         | | `-DeclRefExpr {{.+}} <col:9> 'unsigned long' lvalue Var {{.+}} [[CLP_IV]] 'unsigned long'
// CHECK-NEXT:        | | |   |     |         | `-ImplicitCastExpr {{.+}} <col:17> 'unsigned long' <LValueToRValue>
// CHECK-NEXT:        | | |   |     |         |   `-DeclRefExpr {{.+}} <col:17> 'unsigned long' lvalue Var {{.+}} [[CLP_IDX]] 'unsigned long'
// CHECK-NEXT:        | | |   |     |         |-DeclRefExpr {{.+}} <col:9> 'unsigned long' lvalue Var {{.+}} [[CLP_IV]] 'unsigned long'
// CHECK-NEXT:        | | |   |     |         `-DeclRefExpr {{.+}} <col:17> 'unsigned long' lvalue Var {{.+}} [[CLP_IDX]] 'unsigned long'
// CHECK-NEXT:        | | |   |     `-CapturedStmt {{.+}} <line:9:18, col:22>
// CHECK-NEXT:        | | |   |       `-CapturedDecl {{.+}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT:        | | |   |         |-BinaryOperator {{.+}} <col:18, col:22> 'int' lvalue '='
// CHECK-NEXT:        | | |   |         | |-DeclRefExpr {{.+}} <col:18> 'int' lvalue OMPCapturedExpr {{.+}} 'a' 'int &'
// CHECK-NEXT:        | | |   |         | `-ImplicitCastExpr {{.+}} <col:22> 'int' <LValueToRValue>
// CHECK-NEXT:        | | |   |         |   `-DeclRefExpr {{.+}} <col:22> 'int' lvalue Var {{.+}} 'i' 'int'
