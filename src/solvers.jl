abstract type PoissonBoundaryCondition end
abstract type Neumann <: PoissonBoundaryCondition end

struct Periodic <: PoissonBoundaryCondition end
struct Dirichlet <: PoissonBoundaryCondition end
struct NeumannDCT <: Neumann end

abstract type RHSFormat end
struct Unpermuted <: RHSFormat end
struct Permuted <: RHSFormat end

"""
    NeumannFFT([D=CPU()], [N])

Construct a `NeumannFFT` boundary condition.
If the right-side input array is 'unpermuted' along the
Neumann direction, then providing the size `N` (and optionally the
device D) will calculate and store the permutation vector.
`D` defaults to `CPU()`.  If `N` is not provided, the
input array is assumed to be permuted.
"""
struct NeumannFFT{R<:RHSFormat,P} <: Neumann
    permutation::P
end

NeumannFFT() = NeumannFFT{Permuted, Nothing}(nothing)

#=
function NeumannFFT(D, N, L)
    # compute permutation.
    P = get_array_type(D)
    NeumannFFT{Unpermuted, Nothing}(nothing)
end

NeumannFFT(N, L) = NeumannFFT(CPU(), N, L)
=#

get_array_type(::CPU) = Array{solver_eltype, 3}
@hascuda get_array_type(::GPU) = CuArray{solver_eltype, 3}

"""
    PoissonEigenvalues(A, N, L, bcs)

Return the squared eigenvalues to Poisson's equation
discretized on a staggered grids with size `N` and extent `L`,
in array type `A`, on `grid`,
and for boundary conditions `bcs`.
"""
struct PoissonEigenvalues{A, xBC, yBC, zBC}
    kx²::A
    ky²::A
    kz²::A
end

function PoissonEigenvalues(A, N, L, bcs)
    kx² = convert(A, reshape(get_eigenvalues(bcs[1], N[1], L[1]), (N[1], 1, 1) ))
    ky² = convert(A, reshape(get_eigenvalues(bcs[2], N[2], L[2]), (1, N[2], 1) ))
    kz² = convert(A, reshape(get_eigenvalues(bcs[3], N[3], L[3]), (1, 1, N[3]) ))
    PoissonEigenvalues{A, bcs[1], bcs[2], bcs[3]}(kx², ky², kz²)
end

"""
    get_eigenvalues(bc, N, L)

Get the eigenvalues for Poisson's equation in a particular
direction for boundary condition `bc`, grid size `N` and extent `L`.
"""
get_eigenvalues(::Type{Periodic}, N, L)              = [ (2N * sin( (i-1)*π/N  ) / L)^2 for i=1:N ]
get_eigenvalues(::Type{T},  N, L) where T <: Neumann = [ (2N * sin( (i-1)*π/2N ) / L)^2 for i=1:N ]

#
# Solvers
#
# PoissonSolver_XYZ is a PoissonSolver with boundary conditions
# X, Y, and Z in the x, y, and z directions, respectively.

Base.size(solver::AbstractPoissonSolver) = solver.N
Base.size(solver::AbstractPoissonSolver, d) = solver.N[d]
domain(solver::AbstractPoissonSolver) = solver.L

struct PoissonSolver_PPP{D, A, T1, T2} <: AbstractPoissonSolver{Periodic, Periodic, Periodic, D}
    N                  :: NTuple{dim, Int}
    L                  :: NTuple{dim, solver_float_type}
    eigenvals          :: PoissonEigenvalues{A, Periodic, Periodic, Periodic}
    xyz_fwd_transform! :: T1
    xyz_inv_transform! :: T2
end

struct PoissonSolver_PPN{Nbc<:Neumann, D, Ak, As,
                         T1, T2, T3, T4} <: AbstractPoissonSolver{Periodic, Periodic, Nbc, D}
    N                 :: NTuple{dim, Int}
    L                 :: NTuple{dim, solver_float_type}
    neumann_bc        :: Nbc
    eigenvals         :: PoissonEigenvalues{Ak, Periodic, Periodic, Nbc}
    xy_fwd_transform! :: T1
    z_fwd_transform!  :: T2
    xy_inv_transform! :: T3
    z_inv_transform!  :: T4
    fft2dct           :: As
    ifft2idct         :: As
end

"""
    PoissonSolver([Dev=CPU()], xBC, yBC, zBC, N, L; kwargs...)

Return a `PoissonSolver` on device `Dev` for given boundary
conditions, grid size `N`, and grid length `L`.
"""
PoissonSolver(args...; kwargs...) = PoissonSolver(CPU(), args...; kwargs...)
function PoissonSolver(D, bcs::NTuple{dim, PoissonBoundaryCondition}, args...; kwargs...)
    return PoissonSolver(D, bcs[1], bcs[2], bcs[3], args...; kwargs...)
end

function PoissonSolver(D, ::Periodic, ::Periodic, ::Periodic, N, L;
                       planning_array=convert(get_array_type(D), zeros(N)),
                       planner_flag=FFTW.MEASURE)

    Ak = get_array_type(D)
    eigenvals = PoissonEigenvalues(Ak, N, L, (Periodic, Periodic, Periodic))
     FFT_xyz! = plan_fft!( planning_array, [1, 2, 3]; flags=planner_flag)
    IFFT_xyz! = plan_ifft!(planning_array, [1, 2, 3]; flags=planner_flag)

    PoissonSolver_PPP{D, Ak, typeof(FFT_xyz!), typeof(IFFT_xyz!)}(
        N, L, eigenvals, FFT_xyz!, IFFT_xyz!)
end


function PoissonSolver(D::CPU, ::Periodic, ::Periodic, ::NeumannDCT, N, L;
                       planning_array=convert(get_array_type(D), zeros(N)),
                       planner_flag=FFTW.MEASURE)

    Ak = get_array_type(D)
    eigenvals = PoissonEigenvalues(Ak, N, L, (Periodic, Periodic, NeumannDCT))

     FFT_xy! =      plan_fft!(  planning_array,               [1, 2]; flags=planner_flag)
      DCT_z! = FFTW.plan_r2r!(  planning_array, FFTW.REDFT10,      3; flags=planner_flag)
    IFFT_xy! =      plan_ifft!( planning_array,               [1, 2]; flags=planner_flag)
     IDCT_z! = FFTW.plan_r2r!(  planning_array, FFTW.REDFT01,      3; flags=planner_flag)

    PoissonSolver_PPN{
        NeumannDCT, D, Ak, Nothing,
        typeof(FFT_xy!), typeof(DCT_z!), typeof(IFFT_xy!), typeof(IDCT_z!)}(
        N, L, NeumannDCT(), eigenvals, FFT_xy!, DCT_z!, IFFT_xy!, IDCT_z!, nothing, nothing)
end

"""
    get_fft2dct_factors(N)

Return the exponential factors needed to calculate the
Discrete Cosine Transform and its inverse
from the Fast Fourier Transform and its inverse.
"""
function get_fft2dct_factors(N)
    fft2dct = 2*exp.(-im*π*(0:N-1)/2N)
    ifft2idct = exp.(im*π*(0:N-1)/2N)
    ifft2idct[1] *= 0.5
    return fft2dct, ifft2idct
end

function PoissonSolver(D, ::Periodic, ::Periodic, ::NeumannFFT, N, L;
                       planning_array=convert(get_array_type(D), zeros(N)),
                       planner_flag=FFTW.MEASURE)

    Ak = get_array_type(D)
    eigenvals = PoissonEigenvalues(Ak, N, L, (Periodic, Periodic, NeumannFFT))

    FFT_xy!  =  plan_fft!(planning_array, [1, 2] )
    FFT_z!   =  plan_fft!(planning_array,      3 )
    IFFT_xy! = plan_ifft!(planning_array, [1, 2] )
    IFFT_z!  = plan_ifft!(planning_array,      3 )

    Nx, Ny, Nz = N
    fft2dct, ifft2idct = get_fft2dct_factors(Nz)
      fft2dct = convert(Ak, reshape(fft2dct, (1, 1, Nz)))
    ifft2idct = convert(Ak, reshape(ifft2idct, (1, 1, Nz)))

    PoissonSolver_PPN{
        NeumannFFT, D, Ak, Nothing, typeof(FFT_xy!), typeof(FFT_z!), typeof(IFFT_xy!), typeof(FFT_z!)}(
        N, L, NeumannFFT(), eigenvals, FFT_xy!, FFT_z!, IFFT_xy!, IFFT_z!, fft2dct, ifft2idct)
end

PoissonSolver(::GPU, ::Periodic, ::Periodic, ::Neumann, args...; kwargs...) = PoissonSolver(
    GPU(), Periodic(), Periodic(), NeumanFFT(), args...; kwargs...)
