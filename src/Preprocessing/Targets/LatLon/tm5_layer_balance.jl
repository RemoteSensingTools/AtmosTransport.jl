"""Convert flat ring-face storage to LL directional arrays, keeping the periodic mirror."""
function directional_fluxes(u,v,grid::LatLonTargetGeometry)
    nx=nlon(grid)
    ny=nlat(grid)
    nz=size(u,2)
    am=zeros(nx+1,ny,nz)
    bm=reshape(copy(v),nx,ny+1,nz)
    am[1:nx,:,:] .= reshape(u,nx,ny,nz)
    am[end,:,:] .= am[1,:,:]
    return am,bm
end

function tm5_horizontal_convergence(u,v,grid::LatLonTargetGeometry)
    am,bm=directional_fluxes(u,v,grid)
    return reshape(am[1:end-1,:,:].-am[2:end,:,:].+bm[:,1:end-1,:].-bm[:,2:end,:],
        nlon(grid)*nlat(grid),size(u,2))
end


function balance_tm5_layers!(u,v,target,grid::LatLonTargetGeometry,solver::LLPoissonWorkspace)
    am,bm=directional_fluxes(u,v,grid)
    balance_mass_fluxes!(am,bm,reshape(copy(target),nlon(grid),nlat(grid),size(u,2)),solver;log_summary=false)
    u .= reshape(view(am,1:nlon(grid),:,:),size(u))
    v .= reshape(bm,size(v))
    return nothing
end


# TM5 guards measure the column pressure-tendency mismatch / mean PS.
# Remove only the known global water tendency introduced by the dry pin.
# A changing global dry mass remains an error; cell-area normalization cannot amplify this mode.
function tm5_balance_diagnostics(u,v,a,b,grid,seconds,areas)
    conv=tm5_horizontal_convergence(u,v,grid)
    mismatch=vec(sum(conv;dims=2)).-(vec(sum(b.m-a.m;dims=2))./seconds)
    water = sum(b.m .* b.q .- a.m .* a.q) / seconds
    mismatch .+= water .* (areas ./ sum(areas))
    relative=mismatch.*STANDARD_GRAVITY./areas./((a.ps+b.ps)./2)
    return (;rms=sqrt(sum(abs2,relative)/length(relative)),maximum=maximum(abs,relative))
end

"""Balance copies of both endpoint rates against one tendency, dry each, and average (Q2)."""
function balance_window_layers!(::TM5SpectralFluxes,a,b,grid,solver,seconds,areas)
    seconds>0 || throw(ArgumentError("window duration must be positive"))
    dm=(b.m-a.m)./seconds
    diagnostics=map((a,b)) do endpoint
        u=copy(endpoint.u)
        v=copy(endpoint.v)
        pre=tm5_balance_diagnostics(u,v,a,b,grid,seconds,areas)
        pre.rms<=1e-4 && pre.maximum<=1e-3 || error("TM5 pre-balance guard failed: $pre")
        target=dm.-diff(endpoint.W;dims=2)
        # The layer Poisson equation cannot change its global mode. Remove it
        # in pressure space (area weights), before the solver projects in index
        # space. Final dry column repair imposes the stored dry endpoint budget.
        target .-= (areas ./ sum(areas)) * sum(target; dims=1)
        post=pre
        passes=0
        for pass in 1:3
            balance_tm5_layers!(u,v,target,grid,solver)
            post=tm5_balance_diagnostics(u,v,a,b,grid,seconds,areas)
            passes=pass
            post.rms<=1e-7 && post.maximum<=1e-6 && break
        end
        post.rms<=1e-7 && post.maximum<=1e-6 || error("TM5 post-balance guard failed after three passes: $post")
        layer=tm5_horizontal_convergence(u,v,grid).-target
        dry_target_fluxes!(u,v,endpoint.q,grid)
        (;u,v,pre,post,passes,max_layer_residual=maximum(abs,layer))
    end
    return (;u=(diagnostics[1].u+diagnostics[2].u)./2,
        v=(diagnostics[1].v+diagnostics[2].v)./2,
        m=a.m.*(1 .-a.q),m_next=b.m.*(1 .-b.q),diagnostics)
end

function dry_target_fluxes!(u,v,q,grid::LatLonTargetGeometry)
    am,bm=directional_fluxes(u,v,grid)
    m=ones(nlon(grid),nlat(grid),size(q,2))
    apply_dry_basis_native!(m,am,bm,reshape(q,size(m)))
    u .= reshape(view(am,1:nlon(grid),:,:),size(u))
    v .= reshape(bm,size(v))
    return nothing
end
