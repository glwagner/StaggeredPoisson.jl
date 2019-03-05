module StaggeredPoisson

export
    PoissonSolver,
    solve_poisson!

using
    FFTW,
    CuArrays,
    GPUifyLoops

const solver_eltype = Complex{Float64} # Enforce Float64 for Poisson solver irrespective of grid type.

abstract type PoissonBC end
struct Periodic <: PoissonBC end
struct Neumann <: PoissonBC end

abstract type AbstractPoissonSolver{D, xBC, yBC, zBC} end

include("solvers.jl")
include("algorithms.jl")

end # module
