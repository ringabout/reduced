
import
  gemm_ukernel_dispatch

const ukernel = x86_AVX512.x86_ukernel(float64)
dispatch_general(ukernel)
