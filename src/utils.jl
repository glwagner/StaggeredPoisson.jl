function makegrid(N, L, X0=(0, 0, 0))
    x = X0[1] .+ reshape(range(0, step=L[1]/N[1], length=N[1]), (N[1], 1, 1) )
    y = X0[2] .+ reshape(range(0, step=L[2]/N[2], length=N[2]), (1, N[2], 1) )
    z = X0[3] .+ reshape(range(0, step=L[3]/N[3], length=N[3]), (1, 1, N[3]) )
    x, y, z
end

"""
    cxtype(T)

Returns `T` when `T` is `Complex`, or `Complex{T}` when `T` is `Real`.
"""
cxtype(::Type{T}) where T<:Number = T
cxtype(::Type{T}) where T<:Real = Complex{T}

"""
    innereltype(x)

Recursively determine the 'innermost' type in by the collection `x` (which may be, for example,
a collection of a collection).
"""
function innereltype(x)
  T = eltype(x)
  T <: AbstractArray ? innereltype(T) : return T
end

cxeltype(x) = cxtype(innereltype(x))
