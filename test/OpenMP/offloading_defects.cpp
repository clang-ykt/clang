// RUN:   %clangxx -DCK1 -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=nvptx64-nvidia-cuda -O3 -S -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix CK1 %s
#ifdef CK1


// CK1: define {{.*}}void @__omp_offloading_{{.+}}bar{{.+}}

int foo (float myvar)
{
  int a;
  // CK1: pmovmskb
  __asm ("pmovmskb %1, %0" : "=r" (a) : "x" (myvar));
  return a & 0x8;
}

int bar (int a, float b){
  int c = a + foo(b);

  // CK1: tgt_target(i64 -1
  #pragma omp target
    b = a+1;

  return b;
}
#endif

// #############################################################################
// RUN:   %clangxx -DCK2 -target powerpc64le-ibm-linux-gnu -mcpu=pwr8 -maltivec -O0 -S -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix CK2 %s
#ifdef CK2

static __inline__ vector float __attribute__((__overloadable__, __always_inline__)) vec_ld(int __a,
                                                   const vector float *__b) {
  return (vector float)__builtin_altivec_lvx(__a, __b);
}

static __inline__ vector float __attribute__((__overloadable__, __always_inline__)) vec_ld(int __a, const float *__b) {
  return (vector float)__builtin_altivec_lvx(__a, __b);
}

static __inline__ vector double __attribute__((__overloadable__, __always_inline__)) __builtin_vec_splats(double __a) {
  return (vector double)(__a);
}

static __inline__ vector double __attribute__((__overloadable__, __always_inline__)) vec_abs(vector double __a) {
  return __builtin_vsx_xvabsdp(__a);
}

static __inline__ vector bool long long __attribute__((__overloadable__, __always_inline__))
vec_cmpgt(vector double __a, vector double __b) {
  return (vector bool long long)__builtin_vsx_xvcmpgtdp(__a, __b);
}

static __inline__ vector double __attribute__((__overloadable__, __always_inline__))
vec_sel(vector double __a, vector double __b, vector bool long long __c) {
  vector long long __res = ((vector long long)__a & ~(vector long long)__c) |
                           ((vector long long)__b & (vector long long)__c);
  return (vector double)__res;
}

static __inline__ vector double __attribute__((__overloadable__, __always_inline__))
vec_sel(vector double __a, vector double __b, vector unsigned long long __c) {
  vector long long __res = ((vector long long)__a & ~(vector long long)__c) |
                           ((vector long long)__b & (vector long long)__c);
  return (vector double)__res;
}

static double x[8];

static inline __vector double vec_ld_dbl(int offset, double const *x) {

return (__vector double)vec_ld(offset,
          reinterpret_cast<float*>(const_cast<double*>(x)));
}

// CK2: declare <4 x i32> @llvm.ppc.altivec.lvx(i8*)
// CK2: declare <2 x i64> @llvm.ppc.vsx.xvcmpgtdp(<2 x double>, <2 x double>)
// CK2: declare <2 x double> @llvm.fabs.v2f64(<2 x double>)

int main(int argc, char **argv)
{
  double A[1] = {-1.0};

  // FIXME: Test with OpenMP offloading support too.
  #pragma omp target
  {
    A[0] += 1.0;
  }

  __vector double vsum = __builtin_vec_splats(A[0]);
  __vector double verror = __builtin_vec_splats((double) 0.0);
  __vector double v = vec_ld_dbl(0, x);
  __vector bool long long sel = vec_cmpgt( vec_abs(v), vec_abs(vsum) );
  __vector double a = vec_sel(vsum, v, sel);


  return 0;
}
#endif
