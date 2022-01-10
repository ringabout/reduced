import
  macros, typetraits,
  ./gemm_tiling



const LASER_MEM_ALIGN*{.intdefine.} = 64
static:
  assert LASER_MEM_ALIGN != 0, "Alignment " & $LASER_MEM_ALIGN & "must be a power of 2"
  assert (LASER_MEM_ALIGN and (LASER_MEM_ALIGN - 1)) == 0, "Alignment " & $LASER_MEM_ALIGN & "must be a power of 2"

template withCompilerOptimHints*() =
  # See https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html
  # and https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#Common-Variable-Attributes

  # Variable is created aligned by LASER_MEM_ALIGN.
  # This is useful to ensure an object can be loaded
  # in a minimum amount of cache lines load
  # For example, the stack part of tensors are 128 bytes and can be loaded in 2 cache lines
  # but would require 3 loads if they are misaligned.
  when defined(vcc):
    {.pragma: align_variable, codegenDecl: "__declspec(align(" & $LASER_MEM_ALIGN & ")) $# $#".}
  else:
    {.pragma: align_variable, codegenDecl: "$# $# __attribute__((aligned(" & $LASER_MEM_ALIGN & ")))".}

  # Variable. Pointer does not alias any existing valid pointers.
  when not defined(vcc):
    {.pragma: restrict, codegenDecl: "$# __restrict__ $#".}
  else:
    {.pragma: restrict, codegenDecl: "$# __restrict $#".}

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

when defined(cpp):
  proc static_cast[T: ptr](input: pointer): T
    {.importcpp: "static_cast<'0>(@)".}

template assume_aligned*[T](data: ptr T, alignment: static int = LASER_MEM_ALIGN): ptr T =
  when defined(cpp) and withBuiltins: # builtin_assume_aligned returns void pointers, this does not compile in C++, they must all be typed
    static_cast[ptr T](builtin_assume_aligned(data, alignment))
  elif withBuiltins:
    cast[ptr T](builtin_assume_aligned(data, alignment))
  else:
    data

template prefetch*[T](
            data: ptr (T or UncheckedArray[T]),
            rw: static PrefetchRW = Read,
            locality: static PrefetchLocality = HighTemporalLocality) =
  ## Prefetch examples:
  ##   - https://scripts.mit.edu/~birge/blog/accelerating-code-using-gccs-prefetch-extension/
  ##   - https://stackoverflow.com/questions/7327994/prefetching-examples
  ##   - https://lemire.me/blog/2018/04/30/is-software-prefetching-__builtin_prefetch-useful-for-performance/
  ##   - https://www.naftaliharris.com/blog/2x-speedup-with-one-line-of-code/
  when withBuiltins:
    builtin_prefetch(data, rw, locality)
  else:
    discard

template pragma_ivdep() {.used.}=
  ## Tell the compiler to ignore unproven loop dependencies
  ## such as "a[i] = a[i + k] * c;" if k is unknown, as it introduces a loop
  ## dependency if it's negative
  ## https://software.intel.com/en-us/node/524501
  ##
  ## Placeholder
  # We don't expose that as it only works on C for loop. Nim only generates while loop
  # except when using OpenMP. But the OpenMP "simd" already achieves the same as ivdep.
  when defined(gcc):
    {.emit: "#pragma GCC ivdep".}
  else: # Supported on ICC and Cray
    {.emit: "pragma ivdep".}

template withCompilerFunctionHints() {.used.}=
  ## Not exposed, Nim codegen will declare them as normal C function.
  ## This messes up with N_NIMCALL, N_LIB_PRIVATE, N_INLINE and also
  ## creates duplicate symbols when one function called by a hot or pure function
  ## is public and inline (because hot and pure cascade to all cunfctions called)
  ## and they cannot be stacked easily: (hot, pure) will only apply the last

  # Function. Returned pointer is aligned to LASER_MEM_ALIGN
  {.pragma: aligned_ptr_result, codegenDecl: "__attribute__((assume_aligned(" & $LASER_MEM_ALIGN & ")) $# $#$#".}

  # Function. Returned pointer cannot alias any other valid pointer and no pointers to valid object occur in any
  # storage pointed to.
  {.pragma: malloc, codegenDecl: "__attribute__((malloc)) $# $#$#".}

  # Function. Creates one or more function versions that can process multiple arguments using SIMD.
  # Ignored when -fopenmp is used and within an OpenMP simd loop
  {.pragma: simd, codegenDecl: "__attribute__((simd)) $# $#$#".}

  # Function. Indicates hot and cold path. Ignored when using profile guided optimization.
  {.pragma: hot, codegenDecl: "__attribute__((hot)) $# $#$#".}
  {.pragma: cold, codegenDecl: "__attribute__((cold)) $# $#$#".}

  # ## pure and const
  # ## Affect Common Sub-expression Elimination, Dead Code Elimination and loop optimization.
  # See
  #   - https://lwn.net/Articles/285332/
  #   - http://benyossef.com/helping-the-compiler-help-you/
  #
  # Function. The function only accesses its input params and global variables state.
  # It does not modify any global, calling it multiple times with the same params
  # and global variables will produce the same result.
  {.pragma: gcc_pure, codegenDecl: "__attribute__((pure)) $# $#$#".}
  #
  # Function. The function only accesses its input params and calling it multiple times
  # with the same params will produce the same result.
  # Warning ⚠:
  #   Pointer inputs must not be dereferenced to read the memory pointed to.
  #   In Nim stack arrays are passed by pointers and big stack data structures
  #   are passed by reference as well. I.e. Result unknown.
  {.pragma: gcc_const, codegenDecl: "__attribute__((const)) $# $#$#".}

  # We don't define per-function fast-math, GCC attribute optimize is broken:
  # --> https://gcc.gnu.org/ml/gcc/2009-10/msg00402.html
  #
  # Workaround floating point latency for algorithms like sum
  # should be done manually.
  #
  # See : https://stackoverflow.com/questions/39095993/does-each-floating-point-operation-take-the-same-time
  # and https://www.agner.org/optimize/vectorclass.pdf "Using multiple accumulators"
  #
  # FP addition has a latency of 3~5 clock cycles, i.e. the result cannot be reused for that much time.
  # But the throughput is 1 FP add per clock cycle (and even 2 per clock cycle for Skylake)
  # So we need to use extra accumulators to fully utilize the FP throughput despite FP latency.
  # On Skylake, all FP latencies are 4: https://www.agner.org/optimize/blog/read.php?i=415
  #
  # Note that this is per CPU cores, each core needs its own "global CPU accumulator" to combat
  # false sharing when multithreading.
  #
  # This wouldn't be needed with fast-math because compiler would consider FP addition associative
  # and create intermediate variables as needed to exploit this through put.


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


when defined(i386) or defined(amd64):
  # SIMD throughput and latency:
  #   - https://software.intel.com/sites/landingpage/IntrinsicsGuide/
  #   - https://www.agner.org/optimize/instruction_tables.pdf

  # Reminder: x86 is little-endian, order is [low part, high part]
  # Documentation at https://software.intel.com/sites/landingpage/IntrinsicsGuide/

  when defined(vcc):
    {.pragma: x86_type, byCopy, header:"<intrin.h>".}
    {.pragma: x86, noDecl, header:"<intrin.h>".}
  else:
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
  #                   SSE - float32 - packed
  #
  # ############################################################

  func mm_setzero_ps*(): m128 {.importc: "_mm_setzero_ps", x86.}
  func mm_set1_ps*(a: float32): m128 {.importc: "_mm_set1_ps", x86.}
  func mm_load_ps*(aligned_mem_addr: ptr float32): m128 {.importc: "_mm_load_ps", x86.}
  func mm_loadu_ps*(data: ptr float32): m128 {.importc: "_mm_loadu_ps", x86.}
  func mm_store_ps*(mem_addr: ptr float32, a: m128) {.importc: "_mm_store_ps", x86.}
  func mm_storeu_ps*(mem_addr: ptr float32, a: m128) {.importc: "_mm_storeu_ps", x86.}
  func mm_add_ps*(a, b: m128): m128 {.importc: "_mm_add_ps", x86.}
  func mm_sub_ps*(a, b: m128): m128 {.importc: "_mm_sub_ps", x86.}
  func mm_mul_ps*(a, b: m128): m128 {.importc: "_mm_mul_ps", x86.}
  func mm_max_ps*(a, b: m128): m128 {.importc: "_mm_max_ps", x86.}
  func mm_min_ps*(a, b: m128): m128 {.importc: "_mm_min_ps", x86.}
  func mm_or_ps*(a, b: m128): m128 {.importc: "_mm_or_ps", x86.}

  # ############################################################
  #
  #                    SSE - float32 - scalar
  #
  # ############################################################

  func mm_load_ss*(aligned_mem_addr: ptr float32): m128 {.importc: "_mm_load_ss", x86.}
  func mm_add_ss*(a, b: m128): m128 {.importc: "_mm_add_ss", x86.}
  func mm_max_ss*(a, b: m128): m128 {.importc: "_mm_max_ss", x86.}
  func mm_min_ss*(a, b: m128): m128 {.importc: "_mm_min_ss", x86.}

  func mm_cvtss_f32*(a: m128): float32 {.importc: "_mm_cvtss_f32", x86.}
    ## Extract the low part of the input
    ## Input:
    ##   { A0, A1, A2, A3 }
    ## Result:
    ##   A0

  func mm_movehl_ps*(a, b: m128): m128 {.importc: "_mm_movehl_ps", x86.}
    ## Input:
    ##   { A0, A1, A2, A3 }, { B0, B1, B2, B3 }
    ## Result:
    ##   { B2, B3, A2, A3 }
  func mm_movelh_ps*(a, b: m128): m128 {.importc: "_mm_movelh_ps", x86.}
    ## Input:
    ##   { A0, A1, A2, A3 }, { B0, B1, B2, B3 }
    ## Result:
    ##   { A0, A1, B0, B1 }

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
  #                    SSE2 - integer - packed
  #
  # ############################################################

  func mm_setzero_si128*(): m128i {.importc: "_mm_setzero_si128", x86.}
  func mm_set1_epi8*(a: int8 or uint8): m128i {.importc: "_mm_set1_epi8", x86.}
  func mm_set1_epi16*(a: int16 or uint16): m128i {.importc: "_mm_set1_epi16", x86.}
  func mm_set1_epi32*(a: int32 or uint32): m128i {.importc: "_mm_set1_epi32", x86.}
  func mm_set1_epi64x*(a: int64 or uint64): m128i {.importc: "_mm_set1_epi64x", x86.}
  func mm_load_si128*(mem_addr: ptr m128i): m128i {.importc: "_mm_load_si128", x86.}
  func mm_loadu_si128*(mem_addr: ptr m128i): m128i {.importc: "_mm_loadu_si128", x86.}
  func mm_storeu_si128*(mem_addr: ptr m128i, a: m128i) {.importc: "_mm_storeu_si128", x86.}
  func mm_add_epi8*(a, b: m128i): m128i {.importc: "_mm_add_epi8", x86.}
  func mm_add_epi16*(a, b: m128i): m128i {.importc: "_mm_add_epi16", x86.}
  func mm_add_epi32*(a, b: m128i): m128i {.importc: "_mm_add_epi32", x86.}
  func mm_add_epi64*(a, b: m128i): m128i {.importc: "_mm_add_epi64", x86.}

  func mm_or_si128*(a, b: m128i): m128i {.importc: "_mm_or_si128", x86.}
  func mm_and_si128*(a, b: m128i): m128i {.importc: "_mm_and_si128", x86.}
  func mm_slli_epi64*(a: m128i, imm8: cint): m128i {.importc: "_mm_slli_epi64", x86.}
    ## Shift 2xint64 left
  func mm_srli_epi64*(a: m128i, imm8: cint): m128i {.importc: "_mm_srli_epi64", x86.}
    ## Shift 2xint64 right
  func mm_srli_epi32*(a: m128i, count: int32): m128i {.importc: "_mm_srli_epi32", x86.}
  func mm_slli_epi32*(a: m128i, count: int32): m128i {.importc: "_mm_slli_epi32", x86.}

  func mm_mullo_epi16*(a, b: m128i): m128i {.importc: "_mm_mullo_epi16", x86.}
    ## Multiply element-wise 2 vectors of 8 16-bit ints
    ## into intermediate 8 32-bit ints, and keep the low 16-bit parts

  func mm_shuffle_epi32*(a: m128i, imm8: cint): m128i {.importc: "_mm_shuffle_epi32", x86.}
    ## Shuffle 32-bit integers in a according to the control in imm8
    ## Formula is in big endian representation
    ## a = {a3, a2, a1, a0}
    ## dst = {d3, d2, d1, d0}
    ## imm8 = {bits76, bits54, bits32, bits10}
    ## d0 will refer a[bits10]
    ## d1            a[bits32]

  func mm_mul_epu32*(a: m128i, b: m128i): m128i {.importc: "_mm_mul_epu32", x86.}
    ## From a = {a1_hi, a1_lo, a0_hi, a0_lo} with a1 and a0 being 64-bit number
    ## and  b = {b1_hi, b1_lo, b0_hi, b0_lo}
    ##
    ## Result = {a1_lo * b1_lo, a0_lo * b0_lo}.
    ## This is an extended precision multiplication 32x32 -> 64

  func mm_set_epi32*(e3, e2, e1, e0: cint): m128i {.importc: "_mm_set_epi32", x86.}
    ## Initialize m128i with {e3, e2, e1, e0} (big endian order)
    ## Storing it will yield [e0, e1, e2, e3]

  func mm_castps_si128*(a: m128): m128i {.importc: "_mm_castps_si128", x86.}
    ## Cast a float32x4 vectors into a 128-bit int vector with the same bit pattern
  func mm_castsi128_ps*(a: m128i): m128 {.importc: "_mm_castsi128_ps", x86.}
    ## Cast a 128-bit int vector into a float32x8 vector with the same bit pattern
  func mm_cvtps_epi32*(a: m128): m128i {.importc: "_mm_cvtps_epi32", x86.}
    ## Convert a float32x4 to int32x4
  func mm_cvtepi32_ps*(a: m128i): m128 {.importc: "_mm_cvtepi32_ps", x86.}
    ## Convert a int32x4 to float32x4

  func mm_cmpgt_epi32*(a, b: m128i): m128i {.importc: "_mm_cmpgt_epi32", x86.}
    ## Compare a greater than b

  func mm_cvtsi128_si32*(a: m128i): cint {.importc: "_mm_cvtsi128_si32", x86.}
    ## Copy the low part of a to int32

  func mm_extract_epi16*(a: m128i, imm8: cint): cint {.importc: "_mm_extract_epi16", x86.}
    ## Extract an int16 from a, selected with imm8
    ## and store it in the lower part of destination (padded with zeroes)

  func mm_movemask_epi8*(a: m128i): int32 {.importc: "_mm_movemask_epi8", x86.}
    ## Returns the most significant bit
    ## of each 8-bit elements in `a`

  # ############################################################
  #
  #                    SSE3 - float32
  #
  # ############################################################

  func mm_movehdup_ps*(a: m128): m128 {.importc: "_mm_movehdup_ps", x86.}
    ## Duplicates high parts of the input
    ## Input:
    ##   { A0, A1, A2, A3 }
    ## Result:
    ##   { A1, A1, A3, A3 }
  func mm_moveldup_ps*(a: m128): m128 {.importc: "_mm_moveldup_ps", x86.}
    ## Duplicates low parts of the input
    ## Input:
    ##   { A0, A1, A2, A3 }
    ## Result:
    ##   { A0, A0, A2, A2 }

  # ############################################################
  #
  #                    SSE4.1 - integer - packed
  #
  # ############################################################

  func mm_mullo_epi32*(a, b: m128i): m128i {.importc: "_mm_mullo_epi32", x86.}
    ## Multiply element-wise 2 vectors of 4 32-bit ints
    ## into intermediate 4 64-bit ints, and keep the low 32-bit parts

  # ############################################################
  #
  #                    AVX - float32 - packed
  #
  # ############################################################

  func mm256_setzero_ps*(): m256 {.importc: "_mm256_setzero_ps", x86.}
  func mm256_set1_ps*(a: float32): m256 {.importc: "_mm256_set1_ps", x86.}
  func mm256_load_ps*(aligned_mem_addr: ptr float32): m256 {.importc: "_mm256_load_ps", x86.}
  func mm256_loadu_ps*(mem_addr: ptr float32): m256 {.importc: "_mm256_loadu_ps", x86.}
  func mm256_store_ps*(mem_addr: ptr float32, a: m256) {.importc: "_mm256_store_ps", x86.}
  func mm256_storeu_ps*(mem_addr: ptr float32, a: m256) {.importc: "_mm256_storeu_ps", x86.}
  func mm256_add_ps*(a, b: m256): m256 {.importc: "_mm256_add_ps", x86.}
  func mm256_mul_ps*(a, b: m256): m256 {.importc: "_mm256_mul_ps", x86.}
  func mm256_sub_ps*(a, b: m256): m256 {.importc: "_mm256_sub_ps", x86.}

  func mm256_and_ps*(a, b: m256): m256 {.importc: "_mm256_and_ps", x86.}
    ## Bitwise and
  func mm256_or_ps*(a, b: m256): m256 {.importc: "_mm256_or_ps", x86.}

  func mm256_min_ps*(a, b: m256): m256 {.importc: "_mm256_min_ps", x86.}
  func mm256_max_ps*(a, b: m256): m256 {.importc: "_mm256_max_ps", x86.}
  func mm256_castps256_ps128*(a: m256): m128 {.importc: "_mm256_castps256_ps128", x86.}
    ## Returns the lower part of a m256 in a m128
  func mm256_extractf128_ps*(v: m256, m: cint{lit}): m128 {.importc: "_mm256_extractf128_ps", x86.}
    ## Extracts the low part (m = 0) or high part (m = 1) of a m256 into a m128
    ## m must be a literal

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
  #                   AVX - integers - packed
  #
  # ############################################################

  func mm256_setzero_si256*(): m256i {.importc: "_mm256_setzero_si256", x86.}
  func mm256_set1_epi8*(a: int8 or uint8): m256i {.importc: "_mm256_set1_epi8", x86.}
  func mm256_set1_epi16*(a: int16 or uint16): m256i {.importc: "_mm256_set1_epi16", x86.}
  func mm256_set1_epi32*(a: int32 or uint32): m256i {.importc: "_mm256_set1_epi32", x86.}
  func mm256_set1_epi64x*(a: int64 or uint64): m256i {.importc: "_mm256_set1_epi64x", x86.}
  func mm256_load_si256*(mem_addr: ptr m256i): m256i {.importc: "_mm256_load_si256", x86.}
  func mm256_loadu_si256*(mem_addr: ptr m256i): m256i {.importc: "_mm256_loadu_si256", x86.}
  func mm256_storeu_si256*(mem_addr: ptr m256i, a: m256i) {.importc: "_mm256_storeu_si256", x86.}

  func mm256_castps_si256*(a: m256): m256i {.importc: "_mm256_castps_si256", x86.}
    ## Cast a float32x8 vectors into a 256-bit int vector with the same bit pattern
  func mm256_castsi256_ps*(a: m256i): m256 {.importc: "_mm256_castsi256_ps", x86.}
    ## Cast a 256-bit int vector into a float32x8 vector with the same bit pattern
  func mm256_cvtps_epi32*(a: m256): m256i {.importc: "_mm256_cvtps_epi32", x86.}
    ## Convert a float32x8 to int32x8
  func mm256_cvtepi32_ps*(a: m256i): m256 {.importc: "_mm256_cvtepi32_ps", x86.}
    ## Convert a int32x8 to float32x8

  # ############################################################
  #
  #                   AVX2 - integers - packed
  #
  # ############################################################

  func mm256_add_epi8*(a, b: m256i): m256i {.importc: "_mm256_add_epi8", x86.}
  func mm256_add_epi16*(a, b: m256i): m256i {.importc: "_mm256_add_epi16", x86.}
  func mm256_add_epi32*(a, b: m256i): m256i {.importc: "_mm256_add_epi32", x86.}
  func mm256_add_epi64*(a, b: m256i): m256i {.importc: "_mm256_add_epi64", x86.}

  func mm256_and_si256*(a, b: m256i): m256i {.importc: "_mm256_and_si256", x86.}
    ## Bitwise and
  func mm256_srli_epi64*(a: m256i, imm8: cint): m256i {.importc: "_mm256_srli_epi64", x86.}
    ## Logical right shift

  func mm256_mullo_epi16*(a, b: m256i): m256i {.importc: "_mm256_mullo_epi16", x86.}
    ## Multiply element-wise 2 vectors of 16 16-bit ints
    ## into intermediate 16 32-bit ints, and keep the low 16-bit parts

  func mm256_mullo_epi32*(a, b: m256i): m256i {.importc: "_mm256_mullo_epi32", x86.}
    ## Multiply element-wise 2 vectors of 8x 32-bit ints
    ## into intermediate 8x 64-bit ints, and keep the low 32-bit parts

  func mm256_shuffle_epi32*(a: m256i, imm8: cint): m256i {.importc: "_mm256_shuffle_epi32", x86.}
    ## Shuffle 32-bit integers in a according to the control in imm8
    ## Formula is in big endian representation
    ## a = {hi[a7, a6, a5, a4, lo[a3, a2, a1, a0]}
    ## dst = {d7, d6, d5, d4, d3, d2, d1, d0}
    ## imm8 = {bits76, bits54, bits32, bits10}
    ## d0 will refer a.lo[bits10]
    ## d1            a.lo[bits32]
    ## ...
    ## d4 will refer a.hi[bits10]
    ## d5            a.hi[bits32]

  func mm256_mul_epu32*(a: m256i, b: m256i): m256i {.importc: "_mm256_mul_epu32", x86.}
    ## From a = {a3_hi, a3_lo, a2_hi, a2_lo, a1_hi, a1_lo, a0_hi, a0_lo}
    ## with a3, a2, a1, a0 being 64-bit number
    ## and  b = {b3_hi, b3_lo, b2_hi, b2_lo, b1_hi, b1_lo, b0_hi, b0_lo}
    ##
    ## Result = {a3_lo * b3_lo, a2_lo * b2_lo, a1_lo * b1_lo, a0_lo * b0_lo}.
    ## This is an extended precision multiplication 32x32 -> 64

  func mm256_movemask_epi8*(a: m256i): int32 {.importc: "_mm256_movemask_epi8", x86.}
    ## Returns the most significant bit
    ## of each 8-bit elements in `a`

  func mm256_cmpgt_epi32*(a, b: m256i): m256i {.importc: "_mm256_cmpgt_epi32", x86.}
    ## Compare a greater than b

  func mm256_srli_epi32*(a: m256i, count: int32): m256i {.importc: "_mm256_srli_epi32", x86.}
  func mm256_slli_epi32*(a: m256i, count: int32): m256i {.importc: "_mm256_slli_epi32", x86.}

  func mm_i32gather_epi32*(m: ptr (uint32 or int32), i: m128i, s: int32): m128i {.importc: "_mm_i32gather_epi32", x86.}
  func mm256_i32gather_epi32*(m: ptr (uint32 or int32), i: m256i, s: int32): m256i {.importc: "_mm256_i32gather_epi32", x86.}

  # ############################################################
  #
  #                    AVX512 - float32 - packed
  #
  # ############################################################

  func mm512_setzero_ps*(): m512 {.importc: "_mm512_setzero_ps", x86.}
  func mm512_set1_ps*(a: float32): m512 {.importc: "_mm512_set1_ps", x86.}
  func mm512_load_ps*(aligned_mem_addr: ptr float32): m512 {.importc: "_mm512_load_ps", x86.}
  func mm512_loadu_ps*(mem_addr: ptr float32): m512 {.importc: "_mm512_loadu_ps", x86.}
  func mm512_store_ps*(mem_addr: ptr float32, a: m512) {.importc: "_mm512_store_ps", x86.}
  func mm512_storeu_ps*(mem_addr: ptr float32, a: m512) {.importc: "_mm512_storeu_ps", x86.}
  func mm512_add_ps*(a, b: m512): m512 {.importc: "_mm512_add_ps", x86.}
  func mm512_sub_ps*(a, b: m512): m512 {.importc: "_mm512_sub_ps", x86.}
  func mm512_mul_ps*(a, b: m512): m512 {.importc: "_mm512_mul_ps", x86.}
  func mm512_fmadd_ps*(a, b, c: m512): m512 {.importc: "_mm512_fmadd_ps", x86.}

  func mm512_min_ps*(a, b: m512): m512 {.importc: "_mm512_min_ps", x86.}
  func mm512_max_ps*(a, b: m512): m512 {.importc: "_mm512_max_ps", x86.}

  func mm512_or_ps*(a, b: m512): m512 {.importc: "_mm512_or_ps", x86.}

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

  # # ############################################################
  # #
  # #                   AVX512 - integers - packed
  # #
  # # ############################################################

  func mm512_setzero_si512*(): m512i {.importc: "_mm512_setzero_si512", x86.}
  func mm512_set1_epi8*(a: int8 or uint8): m512i {.importc: "_mm512_set1_epi8", x86.}
  func mm512_set1_epi16*(a: int16 or uint16): m512i {.importc: "_mm512_set1_epi16", x86.}
  func mm512_set1_epi32*(a: int32 or uint32): m512i {.importc: "_mm512_set1_epi32", x86.}
  func mm512_set1_epi64*(a: int64 or uint64): m512i {.importc: "_mm512_set1_epi64", x86.}
  func mm512_load_si512*(mem_addr: ptr SomeInteger): m512i {.importc: "_mm512_load_si512", x86.}
  func mm512_loadu_si512*(mem_addr: ptr SomeInteger): m512i {.importc: "_mm512_loadu_si512", x86.}
  func mm512_storeu_si512*(mem_addr: ptr SomeInteger, a: m512i) {.importc: "_mm512_storeu_si512", x86.}

  func mm512_add_epi8*(a, b: m512i): m512i {.importc: "_mm512_add_epi8", x86.}
  func mm512_add_epi16*(a, b: m512i): m512i {.importc: "_mm512_add_epi16", x86.}
  func mm512_add_epi32*(a, b: m512i): m512i {.importc: "_mm512_add_epi32", x86.}
  func mm512_add_epi64*(a, b: m512i): m512i {.importc: "_mm512_add_epi64", x86.}

  func mm512_mullo_epi32*(a, b: m512i): m512i {.importc: "_mm512_mullo_epi32", x86.}
    ## Multiply element-wise 2 vectors of 16 32-bit ints
    ## into intermediate 16 32-bit ints, and keep the low 32-bit parts

  func mm512_mullo_epi64*(a, b: m512i): m512i {.importc: "_mm512_mullo_epi64", x86.}
    ## Multiply element-wise 2 vectors of 8x 64-bit ints
    ## into intermediate 8x 64-bit ints, and keep the low 64-bit parts

  func mm512_and_si512*(a, b: m512i): m512i {.importc: "_mm512_and_si512", x86.}
    ## Bitwise and

  func mm512_cmpgt_epi32_mask*(a, b: m512i): mmask16 {.importc: "_mm512_cmpgt_epi32_mask", x86.}
    ## Compare a greater than b, returns a 16-bit mask

  func mm512_maskz_set1_epi32*(k: mmask16, a: cint): m512i {.importc: "_mm512_maskz_set1_epi32", x86.}
    ## Compare a greater than b
    ## Broadcast 32-bit integer a to all elements of dst using zeromask k
    ## (elements are zeroed out when the corresponding mask bit is not set).

  func mm512_movm_epi32*(a: mmask16): m512i {.importc: "_mm512_movm_epi32", x86.}

  func mm512_movepi8_mask*(a: m512i): mmask64 {.importc: "_mm512_movepi8_mask", x86.}
    ## Returns the most significant bit
    ## of each 8-bit elements in `a`

  func mm512_srli_epi32*(a: m512i, count: int32): m512i {.importc: "_mm512_srli_epi32", x86.}
  func mm512_slli_epi32*(a: m512i, count: int32): m512i {.importc: "_mm512_slli_epi32", x86.}

  func mm512_i32gather_epi32*(i: m512i, m: ptr (uint32 or int32), s: int32): m512i {.importc: "_mm512_i32gather_epi32", x86.}
    ## Warning ⚠: Argument are switched compared to mm256_i32gather_epi32

  func mm512_castps_si512*(a: m512): m512i {.importc: "_mm512_castps_si512", x86.}
    ## Cast a float32x16 vectors into a 512-bit int vector with the same bit pattern
  func mm512_castsi512_ps*(a: m512i): m512 {.importc: "_mm512_castsi512_ps", x86.}
    ## Cast a 512-bit int vector into a float32x16 vector with the same bit pattern
  func mm512_cvtps_epi32*(a: m512): m512i {.importc: "_mm512_cvtps_epi32", x86.}
    ## Convert a float32x16 to int32x8
  func mm512_cvtepi32_ps*(a: m512i): m512 {.importc: "_mm512_cvtepi32_ps", x86.}
    ## Convert a int32x8 to float32x16

  func cvtmask64_u64*(a: mmask64): uint64 {.importc: "_cvtmask64_u64", x86.}

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

  # TODO: Fused operations like relu/sigmoid/tanh
  #       should be done here as well

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

# #############################################################

template epilogue() {.dirty.} =
  result.add quote do:
    proc `epilogue_name`[MR, NbVecs: static int](
            alpha: `T`, AB: array[MR, array[NbVecs, `V`]],
            beta: `T`, vC: MatrixView[`T`]
          ) =
      template C(i,j: int): untyped {.dirty.} =
        vC.buffer[i*vC.rowStride + j*`nb_scalars`]

      if beta == 0.`T`:
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            `simd_store_unaligned`(C(i,j).addr, `simd_setZero`())
      elif beta != 1.`T`:
        let beta_vec = `simd_broadcast_value`(beta)
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            `simd_store_unaligned`(C(i,j).addr, `simd_mul`(beta_vec, C(i,j).addr.`simd_load_unaligned`))

      if alpha == 1.`T`:
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            `simd_store_unaligned`(C(i,j).addr, `simd_add`(AB[i][j], C(i,j).addr.`simd_load_unaligned`))
      else:
        let alpha_vec = `simd_broadcast_value`(alpha)
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            `simd_store_unaligned`(C(i,j).addr, `simd_fma`(alpha_vec, AB[i][j], C(i,j).addr.`simd_load_unaligned`))

# #############################################################

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

  # 1. Generate the epilogue function
  epilogue()

  # 2. Generate the microkernels for the general and edge cases
  block:
    let ukernel_name = newIdentNode("gebb_ukernel_" & $T & "_" & $simd)
    ukernel_simd_proc(ukernel_name, epilogue_name, edge = false)
  block:
    let ukernel_name = newIdentNode("gebb_ukernel_edge_" & $T & "_" & $simd)
    ukernel_simd_proc(ukernel_name, epilogue_name, edge = true)

# ############################################################
#
#             Actual SIMD implementation
#
# ############################################################

macro ukernel_simd_impl*(
      ukernel: static MicroKernel, V: untyped, A, B: untyped, kc: int,
      simd_setZero, simd_load_aligned, simd_broadcast_value, simd_fma: untyped
    ): untyped =


  let MR = ukernel.mr
  let NR = ukernel.nr

  if false: # Debug implementation
    result = quote do:
      var AB{.align_variable.}: array[`MR`, array[`NR`, float64]]
      var  A {.restrict.} = assume_aligned packedA # [kc, mc] by chunks of mr
      var  B {.restrict.} = assume_aligned packedB # [kc, nc] by chunks of nr

      for k in 0 ..< kc:
        prefetch(B[(k+1)*`NR`].addr, Read, LowTemporalLocality)
        for i in 0 ..< `MR`:
          for j in 0 ..< `NR`-1:
            AB[i][j] += A[k*`MR`+i] * B[k*`NR`+j]
      AB

  else: # Vectorized implementation
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
