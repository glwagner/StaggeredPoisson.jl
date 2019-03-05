"""
    fwd_transforms!(a, solver)

Perform forward transforms on `a` (typically, the source term for 
Poisson's equation) for `solver`, depending on device type and boundary conditions.
"""
function fwd_transforms!(a, solver::AbstractPoissonSolver{D, xBC, yBC, zBC}) where {D<:CPU, xBC<:Periodic, yBC<:Periodic, zBC<:Neumann}
    solver.FFT_xy! * a
     solver.DCT_z! * a
    return nothing
end

function fwd_transforms!(a, solver::AbstractPoissonSolver{D, xBC, yBC, zBC}) where {D<:GPU, xBC<:Periodic, yBC<:Periodic, zBC<:Neumann}
    # Calculate DCTᶻ(f) in place using the FFT.
    solver.FFT_z! * a
    @. a *= solver.fft_to_dct_factors
    @. a = real(a)

    solver.FFT_xy! * a 
    return nothing
end

"""
    bwd_transforms!(a, solver)

Perform backward transforms on `a` (typically, the solution
to Poisson's equation) for `solver`, depending on device type and boundary conditions.
"""
function bwd_transforms!(a, solver::AbstractPoissonSolver{D, xBC, yBC, zBC}) where {D<:CPU, xBC<:Periodic, yBC<:Periodic, zBC<:Neumann}
    solver.IFFT! * a 
    solver.IDCT! * a 
    @. a = a / (2*size(a, 3))
    return nothing
end

function bwd_transforms!(a, solver::AbstractPoissonSolver{D, xBC, yBC, zBC}) where {D<:GPU, xBC<:Periodic, yBC<:Periodic, zBC<:Neumann}
    solver.IFFT_xy! * a 

    # Calculate IDCTᶻ(ϕ) in place using the FFT.
    @. a *= solver.idct_bfactors
    solver.IFFT_z! * a
    return nothing
end
 

"""
    solve_poisson!(ϕ, f, solver)

Solve Poisson's equation,

``
\triangle \phi = f
``

for `ϕ` with the source term `f`, using `solver`.
Boundary conditions and the type of compute device 
are stored in `solver`.
"""
function solve_poisson!(f, ϕ, solver)
    # Transform source term
    fwd_transforms!(f.data, solver)

    # Solve Poisson!
    @. ϕ.data = f.data / (solver.eigenvals.kx² + solver.eigenvals.ky² + solver.eigenvals.kz²)
    ϕ.data[1, 1, 1] = 0 # Set domain  mode to zero

    # Transform solution
    bwd_transforms!(ϕ.data, solver)
    return nothing
end
