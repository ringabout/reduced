import macros

type
  MicroKernel = object
    cpu_simd: CPUFeatureX86

  CPUFeatureX86* = enum
    x86_Generic,
    x86_SSE,
    x86_SSE2,
    x86_SSE4_1,
    x86_AVX,
    x86_AVX_FMA,
    x86_AVX2,
    x86_AVX512


func x86_ukernel*(cpu: CPUFeatureX86, T: typedesc): MicroKernel =
  result.cpu_simd = cpu

type
  Tiles[T] = ptr TilesObj[T]
  TilesObj[T] = object

proc newTiles*(
        ukernel: static MicroKernel,
        T: typedesc,
        M, N, K: Natural,
        ): Tiles[T] =
  result = createSharedU(TilesObj[T])

{.pragma: x86_type, byCopy, header:"<x86intrin.h>".}
{.pragma: x86, noDecl, header:"<x86intrin.h>".}
type
  m512d {.importc: "__m512d", x86_type.} = object
    raw: array[8, float64]

func mm512_setzero_pd(): m512d {.importc: "_mm512_setzero_pd", x86.}

template ukernel_simd_proc(ukernel_name, epilogue_name: NimNode) {.dirty.} =
  result.add quote do:
    proc `ukernel_name`[ukernel: static MicroKernel](
          kc: int,
          alpha: `T`, packedA, packedB: ptr UncheckedArray[`T`],
          beta: `T`
        ) =
      let AB = ukernel_simd_impl(
        ukernel, `V`, packedA, packedB, kc,
        `simd_setZero`
      )

macro ukernel_generator(
      simd: static CPUFeatureX86,
      typ: untyped,
      vectype: untyped,
      simd_setZero: untyped,
    ): untyped =

  let T = newIdentNode($typ)
  let V = newIdentNode($vectype)
  let epilogue_name = newIdentNode("gebb_ukernel_epilogue_" & $T & "_" & $simd)
  result = newStmtList()

  # 2. Generate the microkernels for the general and edge cases
  block:
    let ukernel_name = newIdentNode("gebb_ukernel_" & $T & "_" & $simd)
    ukernel_simd_proc(ukernel_name, epilogue_name)

macro ukernel_simd_impl(
      ukernel: static MicroKernel, V: untyped, A, B: untyped, kc: int,
      simd_setZero: untyped
    ): untyped =

  result = newStmtList()
  var
    rAB = nnkBracket.newTree() # array[MR, array[NbVecs, V]]

  var rABi = nnkBracket.newTree()
  rABi.add genSym(nskVar, "AB" & $0 & "__" & $0)
  rAB.add rABi

  ## Declare
  var declBody = newStmtList()
  let ab = rAB[0][0]
  declBody.add quote do:
    var `ab` = `simd_setZero`()

  ## Assemble:
  result = quote do:
    `declBody`
    `rAB`


{.localpassC:"-mavx512f -mavx512dq".}

ukernel_generator(
    x86_AVX512,
    typ = float64,
    vectype = m512d,
    simd_setZero = mm512_setzero_pd,
  )

{.experimental: "dynamicBindSym".}

macro dispatch_general(
    ukernel: static MicroKernel,
    kc: int,
    alpha: typed, packedA, packedB: ptr UncheckedArray[typed],
    beta: typed
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
      `beta`
    )

proc gebb_ukernel*[T; ukernel: static MicroKernel](
      kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T
    ){.inline.} =
  ukernel.dispatch_general(kc, alpha, packedA, packedB, beta)
