using ConeProj
using Random
using LinearAlgebra
using Test

function sim(rng, n=100, p=100, k=1)
    A = rand(rng, n, p)
    b = rand(rng, n)
    C = rand(rng, k, p)
    d = rand(rng, k)
    C[:, 1:50] .= 0
    return(A, b, C, d)
end

@testset "ConeProj.jl" begin
    rng = MersenneTwister(3);
    A, b, C, d = sim(rng);

    # nn constrained p = 0
    coefs, passive_set, R, status = coneB(A, b);
    @test status == 1
    _, r = qr(A[:, passive_set])
    _coefs = solvex(r, A[:, passive_set], b)
    @test all(A' * (b - A[:, passive_set] * _coefs) .<= 1e-8)

    # eq and nn constrained p = 0
    coefs, passive_set, R, status = coneBEq(A, b, C, d);
    @test status == 1
    _, r = qr(A[:, passive_set])
    _coefs, lambd = solvexeq(r, A[:, passive_set], b, C[:, passive_set], d)
    @test all(A' * (b - A[:, passive_set] * _coefs) + C' * lambd .<= 1e-8)

    # nn constrained p > 0
    coefs, passive_set, R, status = nnls(A, b, p=1);
    @test status == 1
    _, r = qr(A[:, passive_set])
    _coefs = solvex(r, A[:, passive_set], b)
    @test all(A' * (b - A[:, passive_set] * _coefs) .<= 1e-8)

    # eq and nn constrained p > 0
    coefs, passive_set, R, status = ecnnls(A, b, C, d, p=1);
    @test status == 1
    _, r = qr(A[:, passive_set])
    _coefs, lambd = solvexeq(r, A[:, passive_set], b, C[:, passive_set], d)
    @test all(A' * (b - A[:, passive_set] * _coefs) + C' * lambd .<= 1e-8)
end
