using Pkg; Pkg.activate("..")

using
    StaggeredPoisson,
    PyPlot

n = 64
N = (n, 1, 1)
L = (2π, 2π, 2π)
X0 = (-π, -π, -π)

solver = PoissonSolver(Periodic(), Periodic(), Periodic(), N, L)

d = 2π/10
c(x, y, z) = sin(2x)

x, y, z = makegrid(N, L, X0)
c0 = c.(x, y, z)

sol = solve_poisson(c0, solver)

# Plot
sqshow(a) = plot(dropdims(a, dims=(2, 3)))

fig, axs = subplots()
sqshow(c0)
sqshow(real.(sol))
