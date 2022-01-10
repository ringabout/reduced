type
  MatrixView*[T] = object
    buffer*: ptr UncheckedArray[T]
    rowStride*, colStride*: int

func toMatrixView*[T](data: ptr T, rowStride, colStride: int): MatrixView[T] {.inline.} =
  result.buffer = cast[ptr UncheckedArray[T]](data)
  result.rowStride = rowStride
  result.colStride = colStride

template `[]`*[T](view: MatrixView[T], row, col: Natural): T =
  ## Access like a 2D matrix
  view.buffer[row * view.rowStride + col * view.colStride]

template `[]=`*[T](view: MatrixView[T], row, col: Natural, value: T) =
  ## Access like a 2D matrix
  view.buffer[row * view.rowStride + col * view.colStride] = value

func stride*[T](view: MatrixView[T], row, col: Natural): MatrixView[T]{.inline.}=
  ## Returns a new view offset by the row and column stride
  result.buffer = cast[ptr UncheckedArray[T]](
    addr view.buffer[row*view.rowStride + col*view.colStride]
  )
  result.rowStride = view.rowStride
  result.colStride = view.colStride

type
  MicroKernel* = object
    mr*, nr*: int
    cpu_simd*: CPUFeatureX86
    nb_scalars*: int # Ideally MicroKernel should be generic over T
    nb_vecs_nr*: int
    c_unit_stride*: bool # We can use SIMD for the epilogue of C has a unit_stride
    pt*: int # Parallelization threshold

  CPUFeatureX86* = enum
    x86_Generic,
    x86_SSE,
    x86_SSE2,
    x86_SSE4_1,
    x86_AVX,
    x86_AVX_FMA,
    x86_AVX2,
    x86_AVX512
    #   Note that Skylake SP, Xeon Bronze Silver and Gold 5XXX
    #   only have a single AVX512 port and AVX2 can be faster
    #   due to AVX512 downclocking

  X86_FeatureMap = array[CPUFeatureX86, int]

const X86_vecwidth_float: X86_FeatureMap = [
  x86_Generic:         1,
  x86_SSE:     128 div 8,
  x86_SSE2:    128 div 8,
  x86_SSE4_1:  128 div 8,
  x86_AVX:     256 div 8,
  x86_AVX_FMA: 256 div 8,
  x86_AVX2:    256 div 8,
  x86_AVX512:  512 div 8
]

const X86_vecwidth_int: X86_FeatureMap = [
  x86_Generic:         1,
  x86_SSE:             1,
  x86_SSE2:    128 div 8,
  x86_SSE4_1:  128 div 8,
  x86_AVX:     128 div 8,  # Not even addition with integer AVX
  x86_AVX_FMA: 128 div 8,
  x86_AVX2:    256 div 8,
  x86_AVX512:  512 div 8
]

# MR configuration - rows of Ã in micro kernel
# 16 General purpose registers
const X86_regs: X86_FeatureMap = [
  x86_Generic: 2,
  x86_SSE:     6,
  x86_SSE2:    6,
  x86_SSE4_1:  6,
  x86_AVX:     6,
  x86_AVX_FMA: 6,
  x86_AVX2:    6,
  x86_AVX512:  14
]

# NR configuration - Nb of ~B SIMD vectors
# We will also keep as many rows of Ã in SIMD registers at the same time
const NbVecs: X86_FeatureMap = [
    x86_Generic: 1,
    x86_SSE:     2, # 16 XMM registers
    x86_SSE2:    2,
    x86_SSE4_1:  2,
    x86_AVX:     2, # 16 YMM registers
    x86_AVX_FMA: 2,
    x86_AVX2:    2,
    x86_AVX512:  2  # 32 ZMM registers
  ]

func x86_ukernel*(cpu: CPUFeatureX86, T: typedesc, c_unit_stride: bool): MicroKernel =
  result.cpu_simd = cpu
  result.c_unit_stride = c_unit_stride
  result.pt = 128
  when T is SomeFloat:
    result.nb_scalars = max(1, X86_vecwidth_float[cpu] div T.sizeof)
  elif T is SomeInteger: # Integers
    result.nb_scalars = max(1, X86_vecwidth_int[cpu] div T.sizeof)
  else:
    {.error: "Unsupported type: " & T.type.name.}

  result.mr = X86_regs[cpu]                 # 2~6 registers for the rows of Ã
  result.nb_vecs_nr = NbVecs[cpu]           # SIMD vectors of B
  result.nr = result.nb_vecs_nr * result.nb_scalars
