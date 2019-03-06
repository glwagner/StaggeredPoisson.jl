module StaggeredPoisson

export
    PoissonSolver,
    solve_poisson!,
    Periodic,
    Neumann,
    Dirichlet

using
    FFTW,
    CuArrays,
    GPUifyLoops

const solver_eltype = Complex{Float64} # Enforce Float64 type for solver

abstract type AbstractPoissonSolver{xBC, yBC, zBC, D} end

include("solvers.jl")
include("algorithms.jl")

end # module
