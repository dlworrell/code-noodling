AC_PREREQ([2.69])
AC_INIT([dice-cpu], [1.0.0],
        [https://github.com/dlworrell/code-noodling/issues], [dice-cpu])
AC_CONFIG_SRCDIR([src/dice_cpu.cpp])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign subdir-objects -Wall -Werror])

AC_PROG_CXX
AC_LANG([C++])
AX_CXX_COMPILE_STDCXX([17], [noext], [mandatory])

# The unbiased Lemire sampler multiplies two 64-bit values into a 128-bit
# product, so the compiler must provide __uint128_t.
AC_CHECK_TYPES([__uint128_t], [],
  [AC_MSG_ERROR([dice_cpu requires the __uint128_t type for its unbiased sampler; use GCC or Clang on a 64-bit target.])])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
