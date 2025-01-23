module ConeProj


include("QRupdate.jl")
include("UpdatableQR.jl")

using LinearAlgebra
export nnls, ecnnls, solvex, solvexeq, mat_nnls, mat_ecnnls, nmf, UpdatableQR

# there is some way to switch to views here which may speed things up

function nnls(A, b; p=0, passive_set=nothing, uqr=nothing, tol=1e-8, maxit=nothing)
    optimal = true
    n, = size(b)
    _, m = size(A)
    m = m - p

    if maxit == nothing
        maxit = (m + p)^2
    end

    bhat = zeros(n)
    coefs = zeros(m + p)

    if (passive_set == nothing) 
        if p == 0
            passive_set = Vector{Int}()
            uqr = nothing
        else
            passive_set = Vector(1:p)
            uqr = UpdatableQR(A[:, passive_set])
        end
    elseif (uqr == nothing)
        uqr = UpdatableQR(A[:, passive_set])
    end

    if length(passive_set) > 0
        coefs[passive_set] = solvex(uqr.R1, A[:, passive_set], b)
        bhat = A[:, passive_set] * coefs[passive_set]
    end

    proj_resid = A' * (b - bhat) / n
    max_ind = partialsortperm(proj_resid, 1, rev=true)
    if proj_resid[max_ind] <= 2 * tol
        print("done without changing passive set")
        @goto done
    end
    coefs[passive_set] .= 0
    
    if length(passive_set) == p
        if uqr == nothing
            uqr = UpdatableQR(A[:, max_ind:max_ind])
        else
            add_column!(uqr, A[:, max_ind])
        end
        push!(passive_set, max_ind)
    end

    for i in 1:maxit
        A_passive = A[:, passive_set]
        coef_passive = solvex(uqr.R1, A_passive, b)
        if length(coef_passive) > p
            min_ind = p + partialsortperm(coef_passive[p+1:end], 1, rev=false)
            if coef_passive[min_ind] < -tol             
                remove_column!(uqr, min_ind)
                deleteat!(passive_set, min_ind)
            else
                bhat = A_passive * coef_passive
                proj_resid = A' * (b - bhat) / n
                max_ind = partialsortperm(proj_resid, 1, rev=true)
                if proj_resid[max_ind] < 2 * tol
                    coefs[passive_set] = coef_passive
                    @goto done
                end
                add_column!(uqr, A[:, max_ind])
                push!(passive_set, max_ind)
            end
        else 
            coefs[passive_set] = coef_passive
            bhat = A[:, passive_set] * coefs[passive_set]
            @goto done
        end
    end
    coefs[passive_set] = solvex(uqr.R1, A[:, passive_set], b)
    bhat = A[:, passive_set] * coefs[passive_set]
    optimal = false
    @label done
    return(coefs, bhat, passive_set, uqr, optimal)
end

#TODO #3 Investiage the issue even with the 1D case where we end up with negative coefficients 
function ecnnls(A, b, C, d; p=0, passive_set=nothing, uqr=nothing, tol=1e-8, maxit=nothing, debug=false, debug_proj=false)
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


    if (passive_set == nothing) 
        if p == 0
            passive_set = Vector{Int}()
            uqr = nothing
        else
            passive_set = Vector(1:p)
            uqr = UpdatableQR(A[:, passive_set])
            coefs[passive_set] = solvex(uqr.R1, A[:, passive_set], b)
            bhat = A[:, passive_set] * coefs[passive_set]
        end
    else
        if (uqr == nothing)
            uqr = UpdatableQR(A[:, passive_set])
        end
        coefs[passive_set], lambd = solvexeq(uqr.R1, A[:, passive_set], b, C[:, passive_set], d)
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
        if uqr == nothing
            uqr = UpdatableQR(A[:, max_ind:max_ind])
        else
            add_column!(uqr, A[:, max_ind])
        end
        push!(passive_set, max_ind)
        # _, r = qr(A)
        # Cp = (pinv(r) * C')'
        # print(size(Cp))
        # _, constraint_set, _, _ = nnls(Cp, d, p=p)
        # print(size(constraint_set))
        # passive_set = union(1:p, constraint_set)
        # _, R = qr(A[:, passive_set])
    else
        min_ind = p + partialsortperm(coefs[passive_set][p+1:end], 1, rev=false)
        if coefs[passive_set][min_ind] < -tol 
            # println("removing at start", sort(coefs[passive_set]))
            coefs[passive_set[min_ind]] = 0
            remove_column!(uqr, min_ind)
            deleteat!(passive_set, min_ind)
        elseif maximum(proj_resid) <= (2 * tol)
            # println("returning at start", sort(coefs[passive_set]))
            @goto done
        end
    end
    coefs[passive_set] .= 0

    for it in 1:maxit
        A_passive = A[:, passive_set]
        coef_passive, lambd = solvexeq(uqr.R1, A_passive, b, C[:, passive_set], d)
        if debug
            println("iter: $it")
            println("\tpassive_set: $passive_set")
            println("\tcoefs: $coef_passive")
            println("\tlambd: $lambd")
            
        end
        if length(coef_passive) > p
            min_ind = p + partialsortperm(coef_passive[p+1:end], 1, rev=false)
            # println("min_ind", min_ind, " ", coef_passive[min_ind], " ", sort(coef_passive))
            if coef_passive[min_ind] < -tol 
                remove_column!(uqr, min_ind)
                deleteat!(passive_set, min_ind)
            else
                bhat = A_passive * coef_passive
                proj_resid = (A' * (b - bhat) + C' * lambd) / n
                max_ind = partialsortperm(proj_resid, 1, rev=true)
                if debug_proj
                    println("\tproj_resid: $proj_resid[$max_ind] $proj_resid")
                end
                if proj_resid[max_ind] < 2 * tol
                    coefs[passive_set] = coef_passive
                    @goto done
                end
                add_column!(uqr, A[:, max_ind])
                push!(passive_set, max_ind)
            end
        else 
            coefs[passive_set] = coef_passive
            bhat = A[:, passive_set] * coefs[passive_set]
            @goto done
        end
    end
    coefs[passive_set], lambd = solvexeq(uqr.R1, A[:, passive_set], b, C[:, passive_set], d)
    bhat = A[:, passive_set] * coefs[passive_set]
    optimal = false
    @label done
    return(coefs, bhat, passive_set, uqr, optimal)
end

end
