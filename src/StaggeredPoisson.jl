module StaggeredPoisson

export
    AbstractPoissonSolver,

    # solver.jl
    PoissonSolver,
    solve_poisson,
    solve_poisson!,
    Periodic,
    Neumann,
    NeumannFFT,
    NeumannDCT,
    Dirichlet,

    # utils.jl
    makegrid

using
    FFTW,
    GPUifyLoops

const GPU = GPUifyLoops.GPU

const HAVE_CUDA = try
    using CUDAdrv, CUDAnative, CuArrays
    true
catch
    false
end

const dim = 3 # this is a 3d code.
const solver_float_type = Float64
const solver_eltype = Complex{solver_float_type} # Enforce Float64 type for solver

macro hascuda(ex)
    return HAVE_CUDA ? :($(esc(ex))) : :(nothing)
end

abstract type AbstractPoissonSolver{xBC, yBC, zBC, D} end

include("utils.jl")
include("solvers.jl")
include("algorithms.jl")

end # module
