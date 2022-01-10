# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./common/[gemm_tiling],
  ./common/gemm_ukernel_dispatch

const ukernel = x86_AVX512.x86_ukernel(float64, true)
let tiles = ukernel.newTiles(float64, 1, 2, 3)
gebb_ukernel[float64, ukernel](                    # GEBB microkernel + epilogue
            1,                                  #   C[ic+ir:ic+ir+mr, jc+jr:jc+jr+nr] =
      2, nil, nil,                 #    αA[ic+ir:ic+ir+mr, pc:pc+kc] *
      3, default(MatrixView[float64])                                #     B[pc:pc+kc, jc+jr:jc+jr+nr] +
    )                                            #    βC[ic:ic+mc, jc:jc+nc]


# withCompilerOptimHints()

# # ############################################################
# #
# #      Optimized GEMM (Generalized Matrix-Multiplication)
# #
# # ############################################################

# # Features
# #  - Arbitrary stride support
# #  - Efficient implementation (within 90% of the speed of OpenBLAS, more tuning to expect)
# #  - Parallel and scale linearly with number of cores
# #
# # Future
# #  - Implementation extended to integers
# #  - ARM Neon optimisation
# #  - Small matrix multiply optimisation
# #  - Pre-packing to when computing using the same matrix
# #  - batched matrix multiplication

# # Terminology
# #   - M, Matrix: Both dimension are large or unknown
# #   - P, Panel: one of the dimension is small
# #   - B, Block: both dimension are small
# #
# #   - GEMM: GEneralized Matrix-Matrix multiplication
# #   - GEPP: GEneralized Panel-Panel multiplication
# #   - GEBP: Generalized Block-Panel multiplication (macrokernel)
# #   - GEBB: GEneralized Block-Block multiplication (microkernel)
# #   ...

# # ############################################################
# #
# #                     GEBP Macrokernel
# #
# # ############################################################

# proc gebp_mkernel[T; ukernel: static MicroKernel](
#       mc, nc, kc: int,
#       alpha: T, packA, packB: ptr UncheckedArray[T],
#       beta: T,
#       mcncC: MatrixView[T]
#     ) =
#   ## Macro kernel, multiply:
#   ##  - a block A[mc, kc] * panel B[kc, N]

#   # Since nr is small this the the good place to parallelize
#   # See: Anatomy of High-Performance Many-Threaded Matrix Multiplication
#   #      Smith et al
#   #      - http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf

#   # ⚠ We need to ensure that loop variables and pointers
#   # are private to each thread

#   # Nim doesn't support arbitrary increment with OpenMP
#   # So we store indexing/edge case data in tiles
#   const
#     MR = ukernel.extract_mr
#     NR = ukernel.extract_nr

#   gebb_ukernel[T, ukernel](                    # GEBB microkernel + epilogue
#               kc,                                  #   C[ic+ir:ic+ir+mr, jc+jr:jc+jr+nr] =
#         alpha, nil, nil,                 #    αA[ic+ir:ic+ir+mr, pc:pc+kc] *
#         beta, default(MatrixView[system.float64])                                #     B[pc:pc+kc, jc+jr:jc+jr+nr] +
#       )                                            #    βC[ic:ic+mc, jc:jc+nc]

#   # # #####################################
#   # # 4. for jr = 0,...,nc−1 in steps of nr
#   # parallelForStrided jr in 0 ..< nc, stride = NR:
#   #   captures: {mc, nc, kc, alpha, packA, packB, beta, mcncC}

#   #   let nr = min(nc - jr, NR)                        # C[ic:ic+mc, jc+jr:jc+jr+nr]

#   #   # ###################################
#   #   # 5. for ir = 0,...,m−1 in steps of mr
#   #   for ir in countup(0, mc-1, MR):
#   #     let mr = min(mc - ir, MR)
#   #     let c_aux = mcncC.stride(ir, jr)               # C[ic+ir:ic+ir+mr, jc+jr:jc+jr+nr]

#   #     let upanel_b = packB + jr*kc
#   #     prefetch(upanel_b, Read, ModerateTemporalLocality)
#   #     let upanel_a = packA + ir*kc
#   #     prefetch(upanel_a, Read, ModerateTemporalLocality)

#   #     if nr == NR and mr == MR:
#   #       # General case
#   #       gebb_ukernel[T, ukernel](                    # GEBB microkernel + epilogue
#   #               kc,                                  #   C[ic+ir:ic+ir+mr, jc+jr:jc+jr+nr] =
#   #         alpha, upanel_a, upanel_b,                 #    αA[ic+ir:ic+ir+mr, pc:pc+kc] *
#   #         beta, c_aux                                #     B[pc:pc+kc, jc+jr:jc+jr+nr] +
#   #       )                                            #    βC[ic:ic+mc, jc:jc+nc]
#   #     else:
#   #       # Matrix edges
#   #       gebb_ukernel_edge[T, ukernel](               # GEBB microkernel + epilogue
#   #         mr, nr, kc,                                #   C[ic+ir:ic+ir+mr, jc+jr:jc+jr+nr] =
#   #         alpha, upanel_a, upanel_b,                 #    αA[ic+ir:ic+ir+mr, pc:pc+kc] *
#   #         beta, c_aux                                #     B[pc:pc+kc, jc+jr:jc+jr+nr] +
#   #       )                                            #    βC[ic:ic+mc, jc:jc+nc]
#   #     loadBalance(Weave)

# # ###########################################################################################
# #
# #              GEMM Internal Implementation
# #
# # ###########################################################################################

# proc gemm_impl[T; ukernel: static MicroKernel](
#       M, N, K: int,
#       alpha: T, vA: MatrixView[T], vB: MatrixView[T],
#       beta: T, vC: MatrixView[T],
#       tiles: Tiles[T]
#     ) =

#   gebp_mkernel[T, ukernel](                     # GEBP macrokernel:
#               1, 1, 1,                                 #   C[ic:ic+mc, jc:jc+nc] =
#               alpha, nil, tiles.b,                      #    αA[ic:ic+mc, pc:pc+kc] * B[pc:pc+kc, jc:jc+nc] +
#               beta, default(MatrixView[system.float64])                      #    βC[ic:ic+mc, jc:jc+nc]
#             )

# proc gemm_strided_nestable[T: SomeNumber](
#       M, N, K: int,
#       alpha: T,
#       A: ptr T,
#       rowStrideA, colStrideA: int,
#       B: ptr T,
#       rowStrideB, colStrideB: int,
#       beta: T,
#       C: ptr T,
#       rowStrideC, colStrideC: int) =

#   # TODO: shortcut alpha = 0 or K = 0
#   # TODO: elementwise epilogue fusion like relu/tanh/sigmoid
#   # TODO: shortcut for small gemm

#   # Create a view to abstract deling with strides
#   # and passing those in each proc
#   let vA = A.toMatrixView(rowStrideA, colStrideA)
#   let vB = B.toMatrixView(rowStrideB, colStrideB)
#   let vC = C.toMatrixView(rowStrideC, colStrideC)


#   const ukernel = x86_AVX512.x86_ukernel(T, true)
#   let tiles = ukernel.newTiles(T, M, N, K)
#   gemm_impl[T, ukernel](
#       M, N, K,
#       alpha, vA, vB,
#       beta, vC,
#       tiles
#     )
#   deallocTiles(tiles)



# when isMainModule:
#   # # Tests
#   # init(Weave)
#   block:
#     let a = [[1.0, 2, 3],
#              [1.0, 1, 1],
#              [1.0, 1, 1]]

#     let b = [[1.0, 1],
#              [1.0, 1],
#              [1.0, 1]]

#     var res_ab: array[3, array[2, float]]
#     gemm_strided_nestable(
#       3, 2, 3,
#       1.0,  a[0][0].unsafeAddr, 3, 1,
#             b[0][0].unsafeAddr, 2, 1,
#       0.0,  res_ab[0][0].addr,  2, 1
#       )
#     # syncRoot(Weave)
#     # # echo "expected: ", ab
#     # # echo "result: ", res_ab
#     # doAssert res_ab == ab, $res_ab
#     # echo "SUCCESS\n"
