const APS = AbstractPoissonSolver
const P = Periodic
const N = Neumann
const NDCT = NeumannDCT
const NFFT = NeumannFFT

"""
    fwd_transforms!(a, solver)

Perform forward transforms on `a` (typically, the source term for
Poisson's equation) for `solver`, depending on device type and boundary conditions.
"""
function fwd_transforms!(a, solver::APS{xBC, yBC, zBC}) where {xBC<:P, yBC<:P, zBC<:P}
    solver.xyz_fwd_transform! * a
    return nothing
end

function fwd_transforms!(a, solver::APS{xBC, yBC, zBC}) where {xBC<:P, yBC<:P, zBC<:NDCT}
    solver.xy_fwd_transform! * a
     solver.z_fwd_transform! * a
    return nothing
end

function fwd_transforms!(a, solver::APS{xBC, yBC, zBC}) where {xBC<:P, yBC<:P, zBC<:NFFT}
    # Calculate DCTᶻ(f) in place using the FFT.
    solver.z_fwd_transform! * a
    @. a *= solver.fft2dct
    @. a = real(a)

    solver.xy_fwd_transform! * a
    return nothing
end

"""
    inv_transforms!(a, solver)

Perform backward transforms on `a` (typically, the solution
to Poisson's equation) for `solver`, depending on device type and boundary conditions.
"""
function inv_transforms!(a, solver::APS{xBC, yBC, zBC}) where {xBC<:P, yBC<:P, zBC<:P}
    solver.xyz_inv_transform! * a
    @. a = real(a)
    return nothing
end

function inv_transforms!(a, solver::APS{xBC, yBC, zBC}) where {xBC<:P, yBC<:P, zBC<:NDCT}
    solver.xy_inv_transform! * a
    solver.z_inv_transform! * a
    @. a = real(a) / (2*solver.N[3])
    return nothing
end

function inv_transforms!(a, solver::APS{xBC, yBC, zBC}) where {xBC<:P, yBC<:P, zBC<:NFFT}
    solver.xy_inv_transform! * a
    # Do in-place IDCT with FFT in z-direction
    @. a *= solver.ifft2idct
    solver.z_inv_transform! * a
    return nothing
end


"""
    solve_poisson!(ϕ, f, solver)

Solve Poisson's equation,

``
\\triangle \\phi = f
``

for `ϕ` with the source term `f`, using `solver`.
Boundary conditions and the type of compute device
are stored in `solver`.
"""
function solve_poisson!(ϕ, f, solver)
    # Transform source term
    fwd_transforms!(f, solver)

    # Solve Poisson!
    @. ϕ = f / (solver.eigenvals.kx² + solver.eigenvals.ky² + solver.eigenvals.kz²)
    ϕ[1, 1, 1] = 0 # Set domain mode to zero

    # Transform solution
    inv_transforms!(ϕ, solver)
    return nothing
end

function solve_poisson!(ϕ, f, solver::PoissonSolver_PPN{NeumannFFT{R}}) where R <: Unpermuted
    # permute f
    solve_poisson!(ϕ, f, solver)
    return nothing
end

function solve_poisson(f, solver)
    ϕ = similar(f, cxeltype(f))
    ϕ .= 0
    solve_poisson!(ϕ, f, solver)
    return ϕ
end
