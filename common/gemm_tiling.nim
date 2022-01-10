import
  ./compiler_optim_hints,
  macros,
  typetraits

func align_raw_data*(T: typedesc, p: pointer): ptr UncheckedArray[T] =
  static: assert T.supportsCopyMem
  withCompilerOptimHints()

  let address = cast[ByteAddress](p)
  let aligned_ptr{.restrict.} = block: # We cannot directly apply restrict to the default "result"
    let remainder = address and (LASER_MEM_ALIGN - 1) # modulo LASER_MEM_ALIGN (power of 2)
    if remainder == 0:
      assume_aligned cast[ptr UncheckedArray[T]](address)
    else:
      let offset = LASER_MEM_ALIGN - remainder
      assume_aligned cast[ptr UncheckedArray[T]](address +% offset)
  return aligned_ptr


func `+`*(p: ptr, offset: int): type(p) {.inline.}=
  ## Pointer increment
  {.emit: "`result` = `p` + `offset`;".}

template to_ptr*(AB: typed, MR, NR: static int, T: typedesc): untyped =
  assume_aligned cast[ptr array[MR, array[NR, T]]](AB[0][0].unsafeaddr)

# ############################################################
#
#                    Matrix View
#
# ############################################################

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

when defined(amd64): # 64-bit
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
else: # 32-bit
  # MR configuration - registers for the rows of Ã
  # 8 General purpose registers
  const X86_regs: X86_FeatureMap = [
    x86_Generic: 2,
    x86_SSE:     2,
    x86_SSE2:    2,
    x86_SSE4_1:  2,
    x86_AVX:     2,
    x86_AVX_FMA: 2,
    x86_AVX2:    2,
    x86_AVX512:  2
  ]

  # NR configuration - Nb of ~B SIMD vectors
  const NbVecs: X86_FeatureMap = [
      x86_Generic: 1,
      x86_SSE:     2, # 8 XMM registers
      x86_SSE2:    2,
      x86_SSE4_1:  2,
      x86_AVX:     2, # 8 YMM registers
      x86_AVX_FMA: 2,
      x86_AVX2:    2,
      x86_AVX512:  2  # 8 ZMM registers
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

#############################################
# Workaround "undeclared identifier mr or nr"
# for some reason the compiler cannot access fields in
# the static MicroKernel.

macro extract_mr*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.mr
macro extract_nr*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.nr
macro extract_cpu_simd*(ukernel: static MicroKernel): untyped =
  let simd = ukernel.cpu_simd
  result = quote do: CPUFeatureX86(`simd`)
macro extract_nb_scalars*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.nb_scalars
macro extract_nb_vecs_nr*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.nb_vecs_nr
macro extract_c_unit_stride*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.c_unit_stride
macro extract_pt*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.pt


type
  Tiles*[T] = ptr TilesObj[T]
  TilesObj[T] = object
    a*: ptr UncheckedArray[T]
    b*: ptr UncheckedArray[T]
    mc*, nc*, kc*: int

    # Multithreaded panels
    ic_num_tasks*: int   # For private L1-L2 and shared L3
    upanelA_size*: int   # Each thread uses a different upanel of A

    # Allocation data
    a_alloc_mem: pointer
    b_alloc_mem: pointer
    # The Tiles data structure takes 64-byte = 1 cache-line


proc deallocTiles*[T](tiles: Tiles[T]) =
  if tiles.isNil:
    return

  if not tiles.a_alloc_mem.isNil:
    deallocShared tiles.a_alloc_mem
  if not tiles.b_alloc_mem.isNil:
    deallocShared tiles.b_alloc_mem

  freeShared(tiles)

proc newTiles*(
        ukernel: static MicroKernel,
        T: typedesc,
        M, N, K: Natural,
        ): Tiles[T] =
  # BLIS paper [2] section II Figure 2:
  #   - kc * nr in L1 cache µkernel
  #   - mc * kc in L2 cache Ã
  #   - kc * nc in L3 cache ~B (no L3 in Xeon Phi ¯\_(ツ)_/¯)
  result = createSharedU(TilesObj[T])

  const
    nr = ukernel.nr
    mr = ukernel.mr

  # Parallel config
  # Ic loop parallel means that each thread will share a panel B and pack a different A
  # result.ic_num_tasks = get_num_tiles(M, result.mc)

  # Packing
  # During packing the max size is unroll_stop*kc+kc*LR, LR = MR or NR
  result.upanelA_size = result.kc
  let bufA_size = T.sizeof * result.upanelA_size * result.ic_num_tasks
  let bufB_size = T.sizeof * result.kc

  result.a_alloc_mem = allocShared(bufA_size + 63)
  result.b_alloc_mem = allocShared(bufB_size + 63)
  result.a = assume_aligned align_raw_data(T, result.a_alloc_mem)
  result.b = assume_aligned align_raw_data(T, result.b_alloc_mem)
