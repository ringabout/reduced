import macros

type
  MicroKernel = object
    cpu_simd: CPUFeatureX86

  CPUFeatureX86* = enum
    x86_AVX512

func x86_ukernel*(cpu: CPUFeatureX86, T: typedesc): MicroKernel =
  result.cpu_simd = cpu

{.pragma: x86_type, byCopy, header:"<x86intrin.h>".}
{.pragma: x86, noDecl, header:"<x86intrin.h>".}
type
  m512d {.importc: "__m512d", x86_type.} = object
    raw: array[8, float64]

func mm512_setzero_pd(): m512d {.importc: "_mm512_setzero_pd", x86.}

macro ukernel_generator(
      simd: static CPUFeatureX86,
      vectype: untyped
    ): untyped =

  result = newStmtList()

  # 2. Generate the microkernels for the general and edge cases
  block:
    let ukernel_name = newIdentNode("gebb_ukernel_" & $float64 & "_" & $simd)
    result.add quote do:
      proc `ukernel_name`[ukernel: static MicroKernel]() =
        let AB = ukernel_simd_impl(ukernel)

macro ukernel_simd_impl(
      ukernel: static MicroKernel
    ): untyped =

  result = newStmtList()
  var
    rAB = nnkBracket.newTree()

  var rABi = nnkBracket.newTree()
  rABi.add genSym(nskVar, "AB" & $0 & "__" & $0)
  rAB.add rABi

  ## Declare
  var declBody = newStmtList()
  let ab = rAB[0][0]
  declBody.add quote do:
    var `ab` = mm512_setzero_pd()

  ## Assemble:
  result = quote do:
    `declBody`
    `rAB`

{.localpassC:"-mavx512f -mavx512dq".}

ukernel_generator(
    x86_AVX512,
    vectype = m512d
  )

{.experimental: "dynamicBindSym".}

macro dispatch_general*(
    ukernel: static MicroKernel,
  ): untyped =
  let simd = ukernel.cpu_simd

  result = newStmtList()

  # 2. Dispatch according to type and SIMD support
  let simdTag = $simd
  let ukernel_name = bindSym("gebb_ukernel_" & $float64 & "_" & simdTag)
  result.add quote do:
    `ukernel_name`[ukernel]()
