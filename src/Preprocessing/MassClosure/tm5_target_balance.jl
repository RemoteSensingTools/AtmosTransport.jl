"""A pinned principal-minor factorization of the compressed RG Laplacian."""
struct DirectRingPoisson{F}
    factor::F
    nc::Int
end

function DirectRingPoisson(L::CompressedLaplacian)
    nc=length(L.degree)
    ii=Int[]
    jj=Int[]
    vv=Float64[]
    for c in 1:nc
        push!(ii,c)
        push!(jj,c)
        push!(vv,Float64(L.degree[c]))
        for t in L.row_ptr[c]:L.row_ptr[c+1]-1
            push!(ii,c)
            push!(jj,Int(L.col_idx[t]))
            push!(vv,-Float64(L.weights[t]))
        end
    end
    matrix=sparse(ii,jj,vv,nc,nc)
    return DirectRingPoisson(cholesky(Symmetric(matrix[1:nc-1,1:nc-1])),nc)
end

function solve_compressed_poisson_direct!(psi,rhs,solver::DirectRingPoisson; projection::Val=Val(false))
    project_mean_zero!(rhs, projection)
    psi[1:end-1] .= solver.factor \ view(rhs,1:solver.nc-1)
    psi[end]=0
    project_mean_zero!(psi, projection)
    return psi
end

"""Solve one RG layer against an outflow target, including the vertical forcing."""
function balance_compressed_level!(flux,target,left,right,solver::DirectRingPoisson)
    residual=-Float64.(target)
    for f in eachindex(left)
        l=left[f]
        r=right[f]
        l>0 && (residual[l]+=flux[f])
        r>0 && (residual[r]-=flux[f])
    end
    psi=zeros(solver.nc)
    solve_compressed_poisson_direct!(psi,residual,solver)
    for f in eachindex(left)
        l=left[f]
        r=right[f]
        l>0 && r>0 && (flux[f]+=psi[r]-psi[l])
    end
    return nothing
end
