function mat_ecnnls(A, B, C, d)
    X = zeros(size(A, 2), size(B, 2))
    for i in 1:size(B, 2)
        X[:, i] = ecnnls(A, B[:, i], C, d)[1]
    end
    return X
end

function mat_nnls(A, B)
    X = zeros(size(A, 2), size(B, 2))
    for i in 1:size(B, 2)
      X[:, i] = nnls(A, B[:, i])[1]
    end
    return X
  end

#= function mat_nnls(A, B; warm_start_info=nothing, save_warm_start=false)
    if isnothing(warm_start_info)
        warm_start = false
        if save_warm_start
            warm_start_info = Vector{Int}[]
        end
    else
        warm_start = true
    end
    X = zeros(size(A, 2), size(B, 2))
    for i in 1:size(B, 2)
        if warm_start
            passive_set_init = warm_start_info[i]
            X[:, i], _, passive_set, _, _ = nnls(A, B[:, i], passive_set=passive_set_init)
        else
            X[:, i], _, passive_set, _, _ = nnls(A, B[:, i])
        end
        if save_warm_start
            if warm_start
                warm_start_info[i] = passive_set
            else
                push!(warm_start_info, passive_set)
            end
        end
    end
    if save_warm_start
        return X, warm_start_info
    else
        return X
    end
end =#

function mat_ecnnls(A, B, C, d, F)
    X = zeros(size(A, 2), size(B, 2))
    for i in 1:size(B, 2)
        X[1:end.!=F[i], i] = ecnnls(A[:, 1:end.!=F[i]], B[:, i], C, d)[1]
    end
    return X
end

function mat_nnls(A, B, F)
    X = zeros(size(A, 2), size(B, 2))
    for i in 1:size(B, 2)
        X[1:end.!=F[i], i] = nnls(A[:, 1:end.!=F[i]], B[:, i])[1]
    end
    return X
end


function nmf(D, A; tol=1e-8, maxiter=1000, debug=false,)
    W = mat_nnls(A, D)
    for i in 1:maxiter
        A = mat_nnls(W', D')
        A = A';
        W = mat_nnls(A, D)
        kkt = maximum(A' * (D - A*W) / p)
        if kkt < tol
            break
        end
        if debug
            println("Iteration $i, kkt = $kkt")
        end
    end
    return A, W
end

function nmf(D, A, C, d; tol=1e-8, maxiter=1000, debug=false)
    W = mat_ecnnls(A, D, C, d)
    for i in 1:maxiter
        A = mat_nnls(W', D')'
        A = A ./ sum(A, dims=2)
        W = mat_ecnnls(A, D, C, d)
        kkt = maximum(A' * (D - A*W) / p)
        if kkt < tol
            break
        end
        if debug
            println("Iteration $i, kkt = $kkt")
        end
    end
    return A, W
end

function nmf(D, A, C, d, F; tol=1e-8, maxiter=1000, debug=false)
    W = mat_ecnnls(A, D, C, d, F)
    for i in 1:maxiter
        A = mat_nnls(W', D', F)'
        A = A ./ sum(A, dims=2)
        W = mat_ecnnls(A, D, C, d, F)
        kkt = maximum(A' * (D - A*W) / p)
        if kkt < tol
            break
        end
        if debug
            println("Iteration $i, kkt = $kkt")
        end
    end
    return A, W
end