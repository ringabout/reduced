
import
  gemm_ukernel_dispatch

const ukernel = x86_AVX512.x86_ukernel(float64)
let tiles = ukernel.newTiles(float64, 1, 2, 3)
gebb_ukernel[float64, ukernel](                    # GEBB microkernel + epilogue
            1,                                  #   C[ic+ir:ic+ir+mr, jc+jr:jc+jr+nr] =
      2, nil, nil,                 #    αA[ic+ir:ic+ir+mr, pc:pc+kc] *
      3                                #     B[pc:pc+kc, jc+jr:jc+jr+nr] +
    )                                            #    βC[ic:ic+mc, jc:jc+nc]
