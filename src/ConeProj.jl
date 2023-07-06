module ConeProj


include("QRupdate.jl")

using LinearAlgebra
export nnls, ecnnls, solvex, solvexeq

# there is some way to switch to views here which may speed things up

function nnls(A, b; p=0, passive_set=nothing, R=nothing, tol=1e-8, maxit=nothing)
    optimal = true
    n, = size(b)
    _, m = size(A)
    m = m - p

    if maxit == nothing
        maxit = (m + p)^2
    end

    bhat = zeros(n)
    coefs = zeros(m + p)

    if (passive_set == nothing) | (R == nothing)
        if p == 0
            passive_set = Vector{Int}()
            R = zeros(0, 0)
        else
            passive_set = Vector(1:p)
            _, R = qr(A[:, passive_set])
        end
    end

    if length(passive_set) > 0
        coefs[passive_set] = solvex(R, A[:, passive_set], b)
        bhat = A[:, passive_set] * coefs[passive_set]
    end

    proj_resid = A' * (b - bhat) / n
    max_ind = partialsortperm(proj_resid, 1, rev=true)
    if proj_resid[max_ind] <= 2 * tol
        return(coefs, passive_set, R, optimal)
    end
    coefs[passive_set] .= 0
    
    if length(passive_set) == p
        R = qraddcol(A[:, passive_set], R, A[:, max_ind])
        push!(passive_set, max_ind)
    end

    for i in 1:maxit
        A_passive = A[:, passive_set]
        coef_passive = solvex(R, A_passive, b)
        if length(coef_passive) > p
            min_ind = p + partialsortperm(coef_passive[p+1:end], 1, rev=false)
            if coef_passive[min_ind] < -tol             
                R = qrdelcol(R, min_ind)
                deleteat!(passive_set, min_ind)
            else
                bhat = A_passive * coef_passive
                proj_resid = A' * (b - bhat) / n
                max_ind = partialsortperm(proj_resid, 1, rev=true)
                if proj_resid[max_ind] < 2 * tol
                    coefs[passive_set] = coef_passive
                    @goto done
                end
                R = qraddcol(A_passive, R, A[:, max_ind])
                push!(passive_set, max_ind)
            end
        else 
            coefs[passive_set] = coef_passive
            @goto done
        end
    end
    coefs[passive_set] = solvex(R, A[:, passive_set], b)
    optimal = false
    @label done
    return(coefs, passive_set, R, optimal)
end

function ecnnls(A, b, C, d; p=0, passive_set=nothing, R=nothing, constraint_set=nothing, tol=1e-8, maxit=nothing)
    optimal = true
    n, = size(b)
    q = size(d)
    _, m = size(A)
    m = m - p
    obs = 1:(m+p)
    
    if maxit == nothing
        maxit = (m + p)^2
    end

    bhat = zeros(n)
    coefs = zeros(m + p)
    lambd = zeros(q)


    if (passive_set == nothing) | (R == nothing)
        if p == 0
            passive_set = Vector{Int}()
            R = zeros(0, 0)
        else
            passive_set = Vector(1:p)
            _, R = qr(A[:, passive_set])
            coefs[passive_set] = solvex(R, A[:, passive_set], b)
            bhat = A[:, passive_set] * coefs[passive_set]
        end
    else
        coefs[passive_set], lambd = solvexeq(R, A[:, passive_set], b, C[:, passive_set], d)
        bhat = A[:, passive_set] * coefs[passive_set]
    end

    # compute projected residual
    proj_resid = (A' * (b - bhat) + C' * lambd) / n
    if all(lambd .== 0)
        feasible_constraint_set = findall(vec(C) .> 0)
        max_ind = feasible_constraint_set[
            partialsortperm(proj_resid[feasible_constraint_set], 1, rev=true)
        ]
        constraint_set = [max_ind]
        R = qraddcol(A[:, passive_set], R, A[:, max_ind])
        push!(passive_set, max_ind)
        # _, r = qr(A)
        # Cp = (pinv(r) * C')'
        # print(size(Cp))
        # _, constraint_set, _, _ = nnls(Cp, d, p=p)
        # print(size(constraint_set))
        # passive_set = union(1:p, constraint_set)
        # _, R = qr(A[:, passive_set])
    else
        if maximum(proj_resid) <= (2 * tol)
            return(bhat, coefs, passive_set, R)
        end
    end
    coefs[passive_set] .= 0

    for it in 1:maxit
        A_passive = A[:, passive_set]
        coef_passive, lambd = solvexeq(R, A_passive, b, C[:, passive_set], d)
        if length(coef_passive) > p
            min_ind = p + partialsortperm(coef_passive[p+1:end], 1, rev=false)
            if coef_passive[min_ind] < -tol 
                R = qrdelcol(R, min_ind)
                deleteat!(passive_set, min_ind)
            else
                bhat = A_passive * coef_passive
                proj_resid = (A' * (b - bhat) + C' * lambd) / n
                max_ind = partialsortperm(proj_resid, 1, rev=true)
                if proj_resid[max_ind] < 2 * tol
                    coefs[passive_set] = coef_passive
                    @goto done
                end
                R = qraddcol(A_passive, R, A[:, max_ind])
                push!(passive_set, max_ind)
            end
        else 
            coefs[passive_set] = coef_passive
            @goto done
        end
    end
    coefs[passive_set], lambd = solvexeq(R, A[:, passive_set], b, C[:, passive_set], d)
    optimal = false
    @label done
    return(coefs, passive_set, R, optimal)
end

end