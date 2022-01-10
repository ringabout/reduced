# reduced

`nim c test.nim `
gives
```
D:\QQPCmgr\Desktop\Nim\weave\benchmarks\matmul_gemm_blas\gemm_pure_nim\reduced\gemm_ukernel_dispatch.nim(173, 5) Hint: 'nr' is declared but 
not used [XDeclaredButNotUsed]
D:\QQPCmgr\Desktop\Nim\weave\benchmarks\matmul_gemm_blas\gemm_pure_nim\reduced\gemm_ukernel_dispatch.nim(174, 5) Hint: 'mr' is declared but 
not used [XDeclaredButNotUsed]
D:\QQPCmgr\Desktop\Nim\weave\benchmarks\matmul_gemm_blas\gemm_pure_nim\reduced\gemm_ukernel_dispatch.nim(425, 11) Hint: 'is_c_unit_stride`gensym13' is declared but not used [XDeclaredButNotUsed]
D:\QQPCmgr\Desktop\Nim\weave\benchmarks\matmul_gemm_blas\gemm_pure_nim\reduced\test.nim(6, 5) Hint: 'tiles' is declared but not used [XDeclaredButNotUsed]
CC: test.nim
In file included from D:/mingw64/lib/gcc/x86_64-w64-mingw32/9.2.0/include/immintrin.h:55,
                 from D:/mingw64/lib/gcc/x86_64-w64-mingw32/9.2.0/include/x86intrin.h:32,
                 from C:\Users\blue\nimcache\test_d\@mtest.nim.c:8:
C:\Users\blue\nimcache\test_d\@mtest.nim.c: In function 'gebb_ukernel_float64_x86_AVX512__test_118':
D:/mingw64/lib/gcc/x86_64-w64-mingw32/9.2.0/include/avx512fintrin.h:325:1: error: inlining failed in call to always_inline '_mm512_setzero_pd': target specific option mismatch
  325 | _mm512_setzero_pd (void)
      | ^~~~~~~~~~~~~~~~~
C:\Users\blue\nimcache\test_d\@mtest.nim.c:654:11: note: called from here
  654 |  AB13_1 = _mm512_setzero_pd();
      |           ^~~~~~~~~~~~~~~~~~~
In file included from D:/mingw64/lib/gcc/x86_64-w64-mingw32/9.2.0/include/immintrin.h:55,
                 from D:/mingw64/lib/gcc/x86_64-w64-mingw32/9.2.0/include/x86intrin.h:32,
                 from C:\Users\blue\nimcache\test_d\@mtest.nim.c:8:
D:/mingw64/lib/gcc/x86_64-w64-mingw32/9.2.0/include/avx512fintrin.h:325:1: error: inlining failed in call to always_inline '_mm512_setzero_pd': target specific option mismatch
  325 | _mm512_setzero_pd (void)
      | ^~~~~~~~~~~~~~~~~
C:\Users\blue\nimcache\test_d\@mtest.nim.c:651:11: note: called from here
  651 |  AB13_0 = _mm512_setzero_pd();
      |           ^~~~~~~~~~~~~~~~~~~
In file included from D:/mingw64/lib/gcc/x86_64-w64-mingw32/9.2.0/include/immintrin.h:55,
                 from D:/mingw64/lib/gcc/x86_64-w64-mingw32/9.2.0/include/x86intrin.h:32,
                 from C:\Users\blue\nimcache\test_d\@mtest.nim.c:8:
D:/mingw64/lib/gcc/x86_64-w64-mingw32/9.2.0/include/avx512fintrin.h:325:1: error: inlining failed in call to always_inline '_mm512_setzero_pd': target specific option mismatch
  325 | _mm512_setzero_pd (void)
      | ^~~~~~~~~~~~~~~~~
C:\Users\blue\nimcache\test_d\@mtest.nim.c:648:11: note: called from here
  648 |  AB12_1 = _mm512_setzero_pd();
      |           ^~~~~~~~~~~~~~~~~~~
compilation terminated due to -fmax-errors=3.

```
