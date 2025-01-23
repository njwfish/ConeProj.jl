using LinearAlgebra

function solvex(Rin::AbstractMatrix{T}, A::AbstractMatrix{T}, b::Vector{T}) where {T}
    R = UpperTriangular(Rin)
    q = A' * b
    x = R' \ q
    x = R \ x
    return x
end

function solvexeq(
    Rin::AbstractMatrix{T}, A::AbstractMatrix{T}, b::Vector{T}, C::AbstractMatrix{T}, d::Vector{T}
)  where {T}
    x = solvex(Rin, A, b)
    R = UpperTriangular(Rin)
    M = C / R
    _, U = qr(M')
    y = U' \ (d - C * x)
    y = U \ y
    z = R' \ (C' * y)
    z = R \ z
    x = x + z
    return x, y
end

# TODO: There should be a more efficient implementation of this. Empirically when we add a row/col
# to R that just adds a col to C, leaving the rest unchanged. There should be some way to compute this.
# If we can cheaply compute that additional column then we can keep a qless factorization of M' below by
# using the addrow functions above. I believe this will be faster than the full qr decomposition each time.
function solvexeq(
    Rin::AbstractMatrix{T}, A::AbstractMatrix{T}, b::Vector{T}, C::AbstractMatrix{T}, d::AbstractMatrix{T}
)  where {T}
    x = solvex(Rin, A, b)
    R = UpperTriangular(Rin)
    M = C / R
    _, U = qr(M')
    y = U' \ (d - C * x)
    y = U \ y
    z = R' \ (C' * y)
    z = R \ z
    x = x + z
    return x, y
end

function solvexeq(
    Rin::AbstractMatrix{T}, A::AbstractMatrix{T}, b::Vector{T}, Uin::AbstractMatrix{T}, C::AbstractMatrix{T}, d::AbstractMatrix{T}
)  where {T}
    x = solvex(Rin, A, b)
    R = UpperTriangular(Rin)
    U = UpperTriangular(Uin)
    y = U' \ (d - C * x)
    y = U \ y
    z = R' \ (C' * y)
    z = R \ z
    x = x + z
    return x, y
end
