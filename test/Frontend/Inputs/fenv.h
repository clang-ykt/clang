int  feclearexcept(int);
int  fetestexcept(int);

#define FE_DIVBYZERO 1
#define FE_INEXACT 2
#define FE_INVALID 4
#define FE_OVERFLOW 8
#define FE_UNDERFLOW 16
