// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

struct S {
  S(): a(0) {}
  S(int v) : a(v) {}
  int a;
  typedef int type;
};

template <typename T>
class S7 : public T {
protected:
  T a;
  T &b;
  typename T::type c:12;
  typename T::type &d;
  S7() : a(0), b(a), c(0), d(a.a) {}

public:
  S7(typename T::type v) : a(v), b(a), c(v), d(a.a) {
#pragma omp for lastprivate(conditional: T::a, c)
    for (int k = 0; k < a.a; ++k) {
      if (k % 4 == 0)
        T::a = k;
      if (k % 6)
        c = k;
    }
  }
};

// CHECK: #pragma omp for lastprivate(conditional: T::a,this->c)

class S8 : public S7<S> {
  S8() {}

public:
  S8(int v) : S7<S>(v){
  }
};

// CHECK: #pragma omp for lastprivate(conditional: this->S::a,this->c)
// CHECK-NEXT: for (int k = 0; k < this->a.a; ++k) {
// CHECK-NEXT: if (k % 4 == 0)
// CHECK-NEXT: ({
// CHECK:      this->S::a = k;
// CHECK-NEXT: })
// CHECK-NEXT: if (k % 6)
// CHECK-NEXT: ({
// CHECK:      this->c = k;
// CHECK-NEXT: })
// CHECK-NEXT: }


int main(int argc, char **argv) {
// CHECK: int main(int argc, char **argv) {
  int d, f;
#pragma omp parallel
#pragma omp for lastprivate(conditional: d, f) collapse(2) schedule(auto) ordered nowait
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j) {
      foo();
      if (j % 2)
        d = j;
      if (j % 5)
        f = j;
    }
  // CHECK: #pragma omp parallel
  // CHECK-NEXT: #pragma omp for lastprivate(conditional: d,f) collapse(2) schedule(auto) ordered nowait
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j)
  // CHECK-NEXT: foo();
  // CHECK-NEXT: if (j % 2)
  // CHECK-NEXT: ({
  // CHECK:       d = j;
  // CHECK-NEXT: })
  // CHECK-NEXT: if (j % 5)
  // CHECK-NEXT: ({
  // CHECK:       f = j;
  // CHECK-NEXT: })
  return 0;
}

#endif
