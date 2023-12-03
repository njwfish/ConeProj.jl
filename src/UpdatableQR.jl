using LinearAlgebra

mutable struct UpdatableQR{T} <: Factorization{T}
    """
    Gives the qr factorization an (n, m) matrix as Q1*R1
    Q2 is such that Q := [Q1 Q2] is orthogonal and R is an (n, n) matrix where R1 "views into".
    """
    Q::Matrix{T}
    R::Matrix{T}
    n::Int
    m::Int

    Q1::SubArray{T, 2, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
    Q2::SubArray{T, 2, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
    R1::UpperTriangular{T, SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int},UnitRange{Int}}, false}}

    function UpdatableQR(A::AbstractMatrix{T}) where {T}
        n, m = size(A)
        @assert(m <= n, "Too many columns in the matrix.")

        F = qr(A)
        Q = F.Q*Matrix(I, n, n)
        R = zeros(T, n, n)
        R[1:m, 1:m] .= F.R

        new{T}(Q, R, n, m,
            view(Q, :, 1:m), view(Q, :, m+1:n),
            UpperTriangular(view(R, 1:m, 1:m)))
    end

end

function add_column!(F::Nothing, a::AbstractVector{T}) where {T}
    return UpdatableQR(reshape(a, length(a), 1))
end

function add_column!(F::UpdatableQR{T}, a::AbstractVector{T}) where {T}
    a1 = F.Q1'*a;
    a2 = F.Q2'*a;

    x = copy(a2)
    for i = length(x):-1:2
        G, r = givens(x[i-1], x[i], i-1, i)
        lmul!(G, x)
        lmul!(G, F.Q2')
    end

    F.R[1:F.m, F.m+1] .= a1
    F.R[F.m+1, F.m+1] = x[1]

    F.m += 1; update_views!(F)

    return a2
end

function add_column_householder!(F::UpdatableQR{T}, a::AbstractVector{T}) where {T}
    a1 = F.Q1'*a;
    a2 = F.Q2'*a;

    Z = qr(a2)
    LAPACK.gemqrt!('R','N', Z.factors, Z.T, F.Q2) # Q2 .= Q2*F.Q
    F.R[1:F.m, F.m+1] .= a1
    F.R[F.m+1, F.m+1] = Z.factors[1, 1]
    F.m += 1; update_views!(F)

    return Z
end

function remove_column!(F::UpdatableQR{T}, idx::Int) where {T}
    Q12 = view(F.Q, :, idx:F.m)
    R12 = view(F.R, idx:F.m, idx+1:F.m)

    for i in 1:size(R12, 1)-1
        G, r = givens(R12[i, i], R12[i + 1, i], i, i+1)
        lmul!(G, R12)
        rmul!(Q12, G')
    end

    for i in 1:F.m, j in idx:F.m-1
        F.R[i, j] = F.R[i, j+1]
    end
    F.R[:, F.m] .= zero(T)

    F.m -= 1; update_views!(F)

    return nothing 
end

function update_views!(F::UpdatableQR{T}) where {T}
    F.R1 = UpperTriangular(view(F.R, 1:F.m, 1:F.m))
    F.Q1 = view(F.Q, :, 1:F.m)
    F.Q2 = view(F.Q, :, F.m+1:F.n)
end


"""
Same as csne but only for x
"""
function solvex(Rin::AbstractMatrix{T}, A::AbstractMatrix{T}, b::Vector{T}) where {T}
    R = UpperTriangular(Rin)
    q = A' * b
    x = R' \ q
    x = R \ x
    return x
end

"""
Solve the corrected semi-normal equations `R'Rx=A'b`.

    x, r = csne(R, A, b) solves the least-squares problem

minimize  ||r||_2,  where  r := b - A*x

using the corrected semi-normal equation approach described by
Bjork (1987). Assumes that `R` is upper triangular.
"""
function csne(Rin::AbstractMatrix{T}, A::AbstractMatrix{T}, b::Vector{T}) where {T}

    R = UpperTriangular(Rin)
    q = A'*b
    x = R' \ q

    bnorm2 = sum(b.^2)
    xnorm2 = sum(x.^2)
    d2 = bnorm2 - xnorm2

    x = R \ x

    # Apply one step of iterative refinement.
    r = b - A*x
    q = A'*r
    dx = R' \ q
    dx = R  \ dx
    x += dx
    r = b - A*x
    return (x, r)
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
