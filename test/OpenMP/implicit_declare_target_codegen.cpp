// expected-no-diagnostics
#ifndef HEADER
#define HEADER
// Test implicit declare target extension
///==========================================================================///
// RUN: %clang_cc1 -fopenmp-implicit-declare-target -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp-implicit-declare-target -DCK1  -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -debug-info-kind=limited -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
#ifdef CK1

__thread int id;
template<typename t>
t bar(t i);
static int foo(int i) { return bar<int>(i); }

extern int baz(int i);

int faz(int i){ return baz(i); }

int fooz()
{
  int i = 0;

  #pragma omp target
  {
    foo(i);
    baz(id);
  }
  // CK1-NOT:	define linkonce_odr i32 @_{{.+}}faz
  // CK1-NOT:	define linkonce_odr i32 @_{{.+}}baz
  // CK1:  define internal i32 @_{{.+}}foo
  // CK1:  call i32 @_{{.+}}bar
  // CK1:  define linkonce_odr i32 @_{{.+}}bar
  // CK1:  define {{.*}}void @__omp_offloading_[[FILEID:[0-9a-f]+_[0-9a-f]+]]__{{.+}}fooz
  // CK1:  call i32 @_{{.+}}foo
  // CK1:  call i32 @_{{.+}}baz
  // CK1:  declare i32 @_{{.+}}baz

  return i;
}


template<typename t>
t bar(t i) { return i;}

#endif


// RUN: %clang_cc1 -fopenmp-implicit-declare-target -DCK2 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp-implicit-declare-target -DCK2  -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -debug-info-kind=limited -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
#ifdef CK2
template<typename t>
class waldo
{
   public:
      void setW(t i)
      {
         W = i;
      }
      void setH(t i)
      {
         H = i;
      }

   protected:
      t W;
      t H;
};

class fred
{
   public:
      int getVar(int var)
      {
         return var * 70;
      }
};

template<typename tW>
class zoo: public fred, public waldo<tW>
{
    public:
        virtual tW getAll_V()
        {
            return waldo<tW>::W;
        }

        tW getAll_H()
        {
            return waldo<tW>::H;
        }
};

template<typename tW, typename tH>
class plugh: public zoo<tW>
{
   public:
      plugh(tH i)
       {
          A = i;
       }

      tW getAll_V()
      {
         return waldo<tW>::W * waldo<tW>::H;
      }

      tW getAll_H()
      {
         return waldo<tW>::W * waldo<tW>::H;
      }
   protected:
      tH A;
};
// CK2:  %class.zoo = type { i32 (...)**, %class.waldo }
// CK2:  %class.waldo = type { i32, i32 }
int fooz()
{
      int i = 0;
      plugh<int,int> myClass(100);
      myClass.setH(1);
      myClass.setW(2);
      zoo<int>& spinner = myClass;

      #pragma omp target
      {
          // It is intentionally commented as we don't support dynamic virtual func in device.
          //spinner.getAll_V();
          spinner.getAll_H();
      }
      // CK2:  define {{.*}}void @__omp_offloading_[[FILEID:[0-9a-f]+_[0-9a-f]+]]__{{.+}}fooz
      // CK2:  call i32 @_{{.+}}zoo{{.+}}getAll_H{{.+}}(%class.zoo
      // CK2-NOT:  call i32 @_{{.+}}plugh{{.+}}getAll_H{{.+}}(%class.plugh
      // CK2:  define linkonce_odr i32 @_{{.+}}zoo{{.+}}getAll_H{{.+}}(%class.zoo
      // CK2-NOT:  define linkonce_odr i32 @_{{.+}}plugh{{.+}}getAll_H{{.+}}(%class.plugh

      return i;
}

#endif


// RUN: %clang_cc1 -fopenmp-implicit-declare-target -x c++ -std=c++11 -DCK3 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp-implicit-declare-target -x c++ -std=c++11 -DCK3  -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -debug-info-kind=limited -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
#ifdef CK3

template<typename T, typename F>
void funky(T *a, T *b, int n, F f)
{
  for (int i = 0; i < n; ++i)
    f(a, b, i);
}

void fooz()
{
  auto plugh = [](int *A, int *B, int i) { A[i] += B[i]; };
  int n =64;
  int a[n],b[n];

  #pragma omp target map(tofrom:a[:n],b[:n])
  {
    auto bar = [](int *A, int *B, int i ) { A[i] += B[i]; };

    for (int i = 0;  i < n; ++ i)
      plugh(a, b, i);
    for (int i = 0;  i < n; ++ i)
      bar(b, a, i);

    funky(a, b, n, plugh);

    funky(a, b, n, bar);
}
  // CK3:    %class.anon* noalias %plugh
  // CK3:    %plugh.addr = alloca %class.anon*, align 8
  // CK3:    %bar = alloca %class.anon.0, align 1
  // CK3:    store %class.anon* %plugh, %class.anon** %plugh.addr, align 8
  // CK3:    call void @"_{{.+}}funky{{.+}}fooz{{.+}}
  // CK3:    call void @"_{{.+}}funky
  // CK3:    define {{.*}}void @"_{{.+}}funky{{.+}}fooz
  // CK3:    define {{.*}}void @"_{{.+}}funky

}

#endif


// C++14 lambda test
// RUN: %clang_cc1 -fopenmp-implicit-declare-target -x c++ -std=c++14 -DCK4 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp-implicit-declare-target -x c++ -std=c++14 -DCK4  -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-64
#ifdef CK4

template<typename Body, typename... Ts>
auto finalizer(Body &&b, Ts&&... args)
{
    return 20;
}

auto make_funky = [](auto &&f) {
    return [=](auto&& ...args) {
        return finalizer(f, args...);
    };
};

template <typename Body, typename...Args>
decltype(auto) do_funky_back(Body&&b, Args&&... args) {
    return b((args)...);
}

auto foo = make_funky([](auto&&...args){
        return do_funky_back(args...);
        });

void fooz (int argc, char* argv[])
{
  int a = 0;
  #pragma omp target map(a)
  {
    a = foo(0, 1, 2, [](auto i1, auto i2, auto i3){
      return i1 + i2 + i3;
    });
  }

  // CK4:   define internal i32 [[LAMBDANAME:@.+]](%class.anon* dereferenceable(1) %b, i32* dereferenceable(4) %args, i32* dereferenceable(4) %args1, i32* dereferenceable(4) %args3, %class.anon.0* dereferenceable(1) %args5)
  // CK4:   define internal i32 [[LAMBDANAME2:@.+]](%class.anon.2* %this, i32* dereferenceable(4) %args, i32* dereferenceable(4) %args1, i32* dereferenceable(4) %args3, %class.anon.0* dereferenceable(1) %args5)
  // CK4:   call i32 [[LAMBDANAME]]
  // CK4:   define {{.*}}void @__omp_offloading_[[FILEID:[0-9a-f]+_[0-9a-f]+]]__{{.+}}fooz
  // CK4:   call i32 [[LAMBDANAME2]]

}
#endif

// Combined constructs test
// RUN: %clang_cc1 -fopenmp-implicit-declare-target -DCK5 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp-implicit-declare-target -DCK5  -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK5 --check-prefix CK5-64

#ifdef CK5

// CK5: define internal i32 @{{.*}}foo{{.*}}(i32 %i)
static int foo(int i) { return i; }

int fooz()
{
  int i = 0;

  // CK5: define {{.*}}void @__omp_offloading_{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target teams
  {
    foo(i);
  }

  // CK5: define {{.*}}void @__omp_offloading_{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target teams distribute
  for(i=0; i<10; ++i)
  {
    foo(i);
  }

  // CK5: define {{.*}}void @__omp_offloading_{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target teams distribute parallel for
  for(i=0; i<10; ++i)
  {
    foo(i);
  }

  // CK5: define {{.*}}void @__omp_offloading_{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target teams distribute parallel for simd
  for(i=0; i<10; ++i)
  {
    foo(i);
  }

  // CK5: define {{.*}}void @__omp_offloading_{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target teams distribute simd
  for(i=0; i<10; ++i)
  {
    foo(i);
  }

  // CK5: define {{.*}}void @__omp_offloading_{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target parallel for
  for(i=0; i<10; ++i)
  {
    foo(i);
  }

  // CK5: define {{.*}}void @__omp_offloading_{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target parallel for simd
  for(i=0; i<10; ++i)
  {
    foo(i);
  }

  // CK5: define {{.*}}void @__omp_offloading_{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target simd
  for(i=0; i<10; ++i)
  {
    foo(i);
  }

  // CK5: define {{.*}}void @__omp_offloading_{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target parallel
  {
    foo(i);
  }

  return i;
}


#endif

#endif
