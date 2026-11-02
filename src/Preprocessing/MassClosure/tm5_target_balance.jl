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

"""Pinned principal-minor CS graph Cholesky, reused across layers and windows."""
struct DirectCSPoisson{F}
    factor::F
    nc::Int
end

function DirectCSPoisson(ft::CSGlobalFaceTable)
    degree = cs_cell_face_degree(ft)
    ii, jj, vv = collect(1:ft.nc), collect(1:ft.nc), Float64.(degree)
    for f in 1:ft.nf
        l, r = Int(ft.face_left[f]), Int(ft.face_right[f])
        append!(ii, (l, r))
        append!(jj, (r, l))
        append!(vv, (-1.0, -1.0))
    end
    matrix = sparse(ii, jj, vv, ft.nc, ft.nc)
    return DirectCSPoisson(cholesky(Symmetric(matrix[1:end-1, 1:end-1])), ft.nc)
end

function solve_compressed_poisson_direct!(psi,rhs,solver::Union{DirectRingPoisson, DirectCSPoisson}; projection::Val=Val(false))
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

"""Balance a CS layer against explicit outward divergence, keeping seam mirrors."""
function balance_cs_level!(k::Int, am, bm, target::_ArrayTarget, ft::CSGlobalFaceTable,
                           solver::DirectCSPoisson, scratch::CSPoissonScratch)
    cs_face_divergence!(scratch.div, am, bm, ft, k)
    for c in 1:ft.nc
        scratch.rhs[c] = scratch.div[c]-target.values[c, k]
    end
    solve_compressed_poisson_direct!(scratch.psi, scratch.rhs, solver)
    apply_cs_flux_correction!(am, bm, scratch.psi, ft, k)
    return nothing
end
