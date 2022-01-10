import
  macros, typetraits

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


func x86_ukernel*(cpu: CPUFeatureX86, T: typedesc, c_unit_stride: bool): MicroKernel =
  result.cpu_simd = cpu
  result.c_unit_stride = c_unit_stride
  result.pt = 128
  result.nb_scalars = max(1, 512 div 8 div T.sizeof)

  result.mr = 14                 # 2~6 registers for the rows of Ã
  result.nb_vecs_nr = 2
  result.nr = result.nb_vecs_nr * result.nb_scalars

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


const LASER_MEM_ALIGN*{.intdefine.} = 64
static:
  assert LASER_MEM_ALIGN != 0, "Alignment " & $LASER_MEM_ALIGN & "must be a power of 2"
  assert (LASER_MEM_ALIGN and (LASER_MEM_ALIGN - 1)) == 0, "Alignment " & $LASER_MEM_ALIGN & "must be a power of 2"

template withCompilerOptimHints*() =
  {.pragma: align_variable, codegenDecl: "$# $# __attribute__((aligned(" & $LASER_MEM_ALIGN & ")))".}
  {.pragma: restrict, codegenDecl: "$# __restrict__ $#".}

const withBuiltins = defined(gcc) or defined(clang) or defined(icc)

type
  PrefetchRW* {.size: cint.sizeof.} = enum
    Read = 0
    Write = 1
  PrefetchLocality* {.size: cint.sizeof.} = enum
    NoTemporalLocality = 0 # Data can be discarded from CPU cache after access
    LowTemporalLocality = 1
    ModerateTemporalLocality = 2
    HighTemporalLocality = 3 # Data should be left in all levels of cache possible
    # Translation
    # 0 - use no cache eviction level
    # 1 - L1 cache eviction level
    # 2 - L2 cache eviction level
    # 3 - L1 and L2 cache eviction level

when withBuiltins:
  proc builtin_assume_aligned(data: pointer, alignment: csize_t): pointer {.importc: "__builtin_assume_aligned", noDecl.}
  proc builtin_prefetch(data: pointer, rw: PrefetchRW, locality: PrefetchLocality) {.importc: "__builtin_prefetch", noDecl.}


template assume_aligned*[T](data: ptr T, alignment: static int = LASER_MEM_ALIGN): ptr T =
  if withBuiltins:
    cast[ptr T](builtin_assume_aligned(data, alignment))
  else:
    data

template prefetch*[T](
            data: ptr (T or UncheckedArray[T]),
            rw: static PrefetchRW = Read,
            locality: static PrefetchLocality = HighTemporalLocality) =
  when withBuiltins:
    builtin_prefetch(data, rw, locality)
  else:
    discard

template pragma_ivdep() {.used.}=
  when defined(gcc):
    {.emit: "#pragma GCC ivdep".}
  else: # Supported on ICC and Cray
    {.emit: "pragma ivdep".}

template withCompilerFunctionHints() {.used.}=
  {.pragma: aligned_ptr_result, codegenDecl: "__attribute__((assume_aligned(" & $LASER_MEM_ALIGN & ")) $# $#$#".}
  {.pragma: malloc, codegenDecl: "__attribute__((malloc)) $# $#$#".}
  {.pragma: simd, codegenDecl: "__attribute__((simd)) $# $#$#".}
  {.pragma: hot, codegenDecl: "__attribute__((hot)) $# $#$#".}
  {.pragma: cold, codegenDecl: "__attribute__((cold)) $# $#$#".}
  {.pragma: gcc_pure, codegenDecl: "__attribute__((pure)) $# $#$#".}
  {.pragma: gcc_const, codegenDecl: "__attribute__((const)) $# $#$#".}


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
  result = createSharedU(TilesObj[T])

  const
    nr = ukernel.nr
    mr = ukernel.mr

  result.upanelA_size = result.kc
  let bufA_size = T.sizeof * result.upanelA_size * result.ic_num_tasks
  let bufB_size = T.sizeof * result.kc

  result.a_alloc_mem = allocShared(bufA_size + 63)
  result.b_alloc_mem = allocShared(bufB_size + 63)
  result.a = assume_aligned align_raw_data(T, result.a_alloc_mem)
  result.b = assume_aligned align_raw_data(T, result.b_alloc_mem)


when defined(i386) or defined(amd64):
  {.pragma: x86_type, byCopy, header:"<x86intrin.h>".}
  {.pragma: x86, noDecl, header:"<x86intrin.h>".}
  type
    m128* {.importc: "__m128", x86_type.} = object
      raw: array[4, float32]
    m128d* {.importc: "__m128d", x86_type.} = object
      raw: array[2, float64]
    m128i* {.importc: "__m128i", x86_type.} = object
      raw: array[16, byte]
    m256* {.importc: "__m256", x86_type.} = object
      raw: array[8, float32]
    m256d* {.importc: "__m256d", x86_type.} = object
      raw: array[4, float64]
    m256i* {.importc: "__m256i", x86_type.} = object
      raw: array[32, byte]
    m512* {.importc: "__m512", x86_type.} = object
      raw: array[16, float32]
    m512d* {.importc: "__m512d", x86_type.} = object
      raw: array[8, float64]
    m512i* {.importc: "__m512i", x86_type.} = object
      raw: array[64, byte]
    mmask16* {.importc: "__mmask16", x86_type.} = distinct uint16
    mmask64* {.importc: "__mmask64", x86_type.} = distinct uint64

  # ############################################################
  #
  #                    SSE2 - float64 - packed
  #
  # ############################################################

  func mm_setzero_pd*(): m128d {.importc: "_mm_setzero_pd", x86.}
  func mm_set1_pd*(a: float64): m128d {.importc: "_mm_set1_pd", x86.}
  func mm_load_pd*(aligned_mem_addr: ptr float64): m128d {.importc: "_mm_load_pd", x86.}
  func mm_loadu_pd*(mem_addr: ptr float64): m128d {.importc: "_mm_loadu_pd", x86.}
  func mm_store_pd*(mem_addr: ptr float64, a: m128d) {.importc: "_mm_store_pd", x86.}
  func mm_storeu_pd*(mem_addr: ptr float64, a: m128d) {.importc: "_mm_storeu_pd", x86.}
  func mm_add_pd*(a, b: m128d): m128d {.importc: "_mm_add_pd", x86.}
  func mm_sub_pd*(a, b: m128d): m128d {.importc: "_mm_sub_pd", x86.}
  func mm_mul_pd*(a, b: m128d): m128d {.importc: "_mm_mul_pd", x86.}


  # ############################################################
  #
  #                   AVX - float64 - packed
  #
  # ############################################################

  func mm256_setzero_pd*(): m256d {.importc: "_mm256_setzero_pd", x86.}
  func mm256_set1_pd*(a: float64): m256d {.importc: "_mm256_set1_pd", x86.}
  func mm256_load_pd*(aligned_mem_addr: ptr float64): m256d {.importc: "_mm256_load_pd", x86.}
  func mm256_loadu_pd*(mem_addr: ptr float64): m256d {.importc: "_mm256_loadu_pd", x86.}
  func mm256_store_pd*(mem_addr: ptr float64, a: m256d) {.importc: "_mm256_store_pd", x86.}
  func mm256_storeu_pd*(mem_addr: ptr float64, a: m256d) {.importc: "_mm256_storeu_pd", x86.}
  func mm256_add_pd*(a, b: m256d): m256d {.importc: "_mm256_add_pd", x86.}
  func mm256_mul_pd*(a, b: m256d): m256d {.importc: "_mm256_mul_pd", x86.}

  # ############################################################
  #
  #                 AVX + FMA - float32/64 - packed
  #
  # ############################################################

  func mm256_fmadd_ps*(a, b, c: m256): m256 {.importc: "_mm256_fmadd_ps", x86.}
  func mm256_fmadd_pd*(a, b, c: m256d): m256d {.importc: "_mm256_fmadd_pd", x86.}

  # ############################################################
  #
  #                    AVX512 - float64 - packed
  #
  # ############################################################

  func mm512_setzero_pd*(): m512d {.importc: "_mm512_setzero_pd", x86.}
  func mm512_set1_pd*(a: float64): m512d {.importc: "_mm512_set1_pd", x86.}
  func mm512_load_pd*(aligned_mem_addr: ptr float64): m512d {.importc: "_mm512_load_pd", x86.}
  func mm512_loadu_pd*(mem_addr: ptr float64): m512d {.importc: "_mm512_loadu_pd", x86.}
  func mm512_store_pd*(mem_addr: ptr float64, a: m512d) {.importc: "_mm512_store_pd", x86.}
  func mm512_storeu_pd*(mem_addr: ptr float64, a: m512d) {.importc: "_mm512_storeu_pd", x86.}
  func mm512_add_pd*(a, b: m512d): m512d {.importc: "_mm512_add_pd", x86.}
  func mm512_mul_pd*(a, b: m512d): m512d {.importc: "_mm512_mul_pd", x86.}
  func mm512_fmadd_pd*(a, b, c: m512d): m512d {.importc: "_mm512_fmadd_pd", x86.}


withCompilerOptimHints()

# ############################################################
#
#          Generic GEBB microkernel implementation
#
# ############################################################

template ukernel_generic_impl*(){.dirty.} =

  const MR = 0
  const NR = 0

  var AB{.align_variable.}: array[MR, array[NR, T]]
  var  A {.restrict.} = assume_aligned packedA # [kc, mc] by chunks of mr
  var  B {.restrict.} = assume_aligned packedB # [kc, nc] by chunks of nr

  for k in 0 ..< kc:
    prefetch(B[(k+1)*NR].addr, Read, LowTemporalLocality)
    for i in 0 ..< MR:
      for j in `||`(0, NR-1, "simd"):
        AB[i][j] += A[k*MR+i] * B[k*NR+j]

# ############################################################
#
#          Fallback Generic version
#
# ############################################################
#
# Cases
# 1. C *=   β, starting default
# 2. C  =  AB, if β = 0 and α = 1
# 3. C  = αAB, if β = 0 and α = 1
# 4. C +=  AB, if α = 1
# 5. C += αAB, if α = 1
#
# TODO: Fused operations like relu/sigmoid/tanh
#       should be done here as well

proc gebb_ukernel_epilogue_fallback*[MR, NR: static int, T](
      alpha: T, AB: ptr array[MR, array[NR, T]],
      beta: T,  vC: MatrixView[T]
    ){.inline.} =

  let pAB{.restrict.} = assume_aligned cast[ptr array[MR, array[NR, T]]](AB[0][0].unsafeAddr)

  if beta == 0.T:
    for i in 0 ..< MR:
      for j in 0 ..< NR:
        vC[i, j] = 0.T
  elif beta != 1.T:                  # C *= β
    for i in 0 ..< MR:
      for j in 0 ..< NR:
        vC[i, j] *= beta

  if alpha == 1.T:                   # C += AB
    for i in 0 ..< MR:
      for j in 0 ..< NR:
        vC[i, j] += pAB[i][j]
  else:                              # C += αAB
    for i in 0 ..< MR:
      for j in 0 ..< NR:
        vC[i, j] += alpha * pAB[i][j]

  # TODO: Fused operations like relu/sigmoid/tanh
  #       should be done here as well

proc gebb_ukernel_fallback*[T; ukernel: static MicroKernel](
      kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ) =
  ukernel_generic_impl()

  const is_c_unit_stride = ukernel.extract_c_unit_stride
  gebb_ukernel_epilogue_fallback(alpha, to_ptr(AB, MR, NR, T), beta, vC)

# ############################################################
#
#                      Matrix edges
#
# ############################################################

func gebb_ukernel_edge_epilogue*[MR, NR: static int, T](
      alpha: T, AB: ptr array[MR, array[NR, T]],
      beta: T,  vC: MatrixView[T],
      mr, nr: int # Tail to process
    ){.inline.} =

  let pAB{.restrict.} = assume_aligned cast[ptr array[MR, array[NR, T]]](AB[0][0].unsafeAddr)

  if beta == 0.T:
    if alpha == 1.T:                   # C = AB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] = pAB[i][j]
    else:                              # C = αAB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] = alpha * pAB[i][j]
  else:                                # C *= β
    for i in 0 ..< mr:
      for j in 0 ..< nr:
        vC[i, j] *= beta

    if alpha == 1.T:                   # C += AB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] += pAB[i][j]
    else:                              # C += αAB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] += alpha * pAB[i][j]

proc gebb_ukernel_edge_fallback*[T; ukernel: static MicroKernel](
      mr, nr, kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ) =
  ukernel_generic_impl()
  gebb_ukernel_edge_epilogue(alpha, to_ptr(AB, MR, NR, T), beta, vC, mr, nr)

template ukernel_simd_proc(ukernel_name, epilogue_name: NimNode, edge: bool) {.dirty.} =
  if edge:
    result.add quote do:
      proc `ukernel_name`*[ukernel: static MicroKernel](
            mr, nr, kc: int,
            alpha: `T`, packedA, packedB: ptr UncheckedArray[`T`],
            beta: `T`, vC: MatrixView[`T`]
          ) =

        let AB{.align_variable.} = ukernel_simd_impl(
          ukernel, `V`, packedA, packedB, kc,
          `simd_setZero`, `simd_load_aligned`, `simd_broadcast_value`, `simd_fma`
        )
        const
          is_c_unit_stride = 0
          MR = 0
          NR = 0

        gebb_ukernel_edge_epilogue(
                alpha, to_ptr(AB, MR, NR, `T`),
                beta, vC, mr, nr
              )
  else:
    result.add quote do:
      proc `ukernel_name`*[ukernel: static MicroKernel](
            kc: int,
            alpha: `T`, packedA, packedB: ptr UncheckedArray[`T`],
            beta: `T`, vC: MatrixView[`T`]
          ) =
        let AB{.align_variable.} = ukernel_simd_impl(
          ukernel, `V`, packedA, packedB, kc,
          `simd_setZero`, `simd_load_aligned`, `simd_broadcast_value`, `simd_fma`
        )
        const
          is_c_unit_stride = ukernel.c_unit_stride
          MR = ukernel.mr
          NR = ukernel.nr

        # when is_c_unit_stride:
        #   `epilogue_name`(alpha, AB, beta, vC)
        # else:
        gebb_ukernel_epilogue_fallback(
          alpha, to_ptr(AB, MR, NR, `T`),
          beta, vC)

macro ukernel_generator*(
      simd: static CPUFeatureX86,
      typ: untyped,
      vectype: untyped,
      nb_scalars: static int,
      simd_setZero: untyped,
      simd_broadcast_value: untyped,
      simd_load_aligned: untyped,
      simd_load_unaligned: untyped,
      simd_store_unaligned: untyped,
      simd_mul: untyped,
      simd_add: untyped,
      simd_fma: untyped,
    ): untyped =

  let T = newIdentNode($typ)
  let V = newIdentNode($vectype)
  let epilogue_name = newIdentNode("gebb_ukernel_epilogue_" & $T & "_" & $simd)
  result = newStmtList()

  # 2. Generate the microkernels for the general and edge cases
  block:
    let ukernel_name = newIdentNode("gebb_ukernel_" & $T & "_" & $simd)
    ukernel_simd_proc(ukernel_name, epilogue_name, edge = false)

macro ukernel_simd_impl*(
      ukernel: static MicroKernel, V: untyped, A, B: untyped, kc: int,
      simd_setZero, simd_load_aligned, simd_broadcast_value, simd_fma: untyped
    ): untyped =

  result = newStmtList()

  ## ukernel config
  let
    MR = ukernel.mr
    NR = ukernel.nr
    NbVecs = ukernel.nb_vecs_nr # == NR div NbScalars
    NbScalars = ukernel.nb_scalars

  ## Registers
  # We keep all C in registers MR*NR size occupying MR*NbVecs
  # We keep NbVecs slivers of A and B for C updates
  var
    rA: seq[NimNode]           # array[NbVecs, V]
    rB: seq[NimNode]           # array[NbVecs, V]
    rAB = nnkBracket.newTree() # array[MR, array[NbVecs, V]]
  for jj in 0 ..< NbVecs:
    rA.add genSym(nskVar, "A" & $jj)
    rB.add genSym(nskVar, "B" & $jj)
  for i in 0 ..< MR:
    var rABi = nnkBracket.newTree()
    for j in 0 ..< NbVecs:
      rABi.add genSym(nskVar, "AB" & $i & "__" & $j)
    rAB.add rABi

  ## Declare
  var declBody = newStmtList()
  for a in rA:
    declBody.add quote do:
      var `a`{.noinit.}: `V`
  for b in rB:
    declBody.add quote do:
      var `b`{.noinit.}: `V`
  for i in 0 ..< MR:
    for j in 0 ..< NbVecs:
      let ab = rAB[i][j]
      declBody.add quote do:
        var `ab` = `simd_setZero`()

  let k = genSym(nskForVar)

  ## Prefetch
  var prefetchBody = newStmtList()
  for jj in 0 ..< NbVecs:
    prefetchBody.add quote do:
      prefetch(`B`[(`k`+1)*`NR`+`jj`*`NbScalars`].addr, Read, LowTemporalLocality)

  ## Load
  var loadBody = newStmtList()
  for jj in 0 ..< NbVecs:
    let b = rB[jj]
    loadBody.add quote do:
      `b` = `simd_load_aligned`(`B`[`k`*`NR`+`jj`*`NbScalars`].addr)

  ## Interleaved broadcast and FMA
  var bcast_fma = newStmtList()
  block:
    let a0 = rA[0]
    bcast_fma.add quote do:
      `a0` = `simd_broadcast_value`(`A`[`k`*`MR`])

  for i in 0 ..< MR:
    # broadcast next iteration
    let next_register_idx = (i+1) mod NbVecs
    let a_next = rA[next_register_idx]
    bcast_fma.add quote do:
      # At the edge: `i`+1 = MR so equivalent to loading A[(k+1)*MR]
      `a_next` = `simd_broadcast_value`(`A`[`k`*`MR`+(`i`+1)])

    # load current
    let a = rA[i mod NbVecs]

    # Do FMA on the current one
    for jj in 0 ..< NbVecs:
      let b = rB[jj]
      let AB = rAB[i][jj]
      bcast_fma.add quote do:
        `AB` = `simd_fma`(`a`, `b`, `AB`)

  ## Assemble:
  result = quote do:
    `declBody`
    for `k` in 0 ..< `kc`:
      `loadBody`
      `prefetchBody`
      `bcast_fma`
    ## Write registers to a MR/NR array
    `rAB`


{.localpassC:"-mavx512f -mavx512dq".}

ukernel_generator(
    x86_AVX512,
    typ = float64,
    vectype = m512d,
    nb_scalars = 8,
    simd_setZero = mm512_setzero_pd,
    simd_broadcast_value = mm512_set1_pd,
    simd_load_aligned = mm512_load_pd,
    simd_load_unaligned = mm512_loadu_pd,
    simd_store_unaligned = mm512_storeu_pd,
    simd_mul = mm512_mul_pd,
    simd_add = mm512_add_pd,
    simd_fma = mm512_fmadd_pd
  )

{.experimental: "dynamicBindSym".}

macro dispatch_general(
    ukernel: static MicroKernel,
    kc: int,
    alpha: typed, packedA, packedB: ptr UncheckedArray[typed],
    beta: typed, vC: MatrixView[typed]
  ): untyped =
  let simd = ukernel.cpu_simd

  result = newStmtList()

  # 2. Dispatch according to type and SIMD support
  let symT = getTypeInst(alpha)
  let simdTag = $simd
  let ukernel_name = bindSym("gebb_ukernel_" & $symT & "_" & simdTag)
  result.add quote do:
    `ukernel_name`[ukernel]( # Hack: ukernel is generic from the calling proc
      `kc`,
      `alpha`, `packedA`, `packedB`,
      `beta`, `vC`
    )

proc gebb_ukernel*[T; ukernel: static MicroKernel](
      kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ){.inline.} =
  ukernel.dispatch_general(kc, alpha, packedA, packedB, beta, vC)

