using Test, AtmosTransport, Random
const MC = AtmosTransport.Preprocessing.MassClosure
const G = AtmosTransport.Grids

@testset "CS endpoint and explicit targets are bitwise equal" begin
    for FT in (Float32, Float64), convention in (G.GnomonicPanelConvention(), G.GEOSNativePanelConvention())
        Random.seed!(9)
        n, nz = 4, 2
        ft = G.build_cs_global_face_table(n, G.panel_connectivity_for(convention))
        degree = MC.cs_cell_face_degree(ft)
        am = ntuple(_ -> randn(FT, n+1, n, nz), 6)
        bm = ntuple(_ -> randn(FT, n, n+1, nz), 6)
        aa, bb = deepcopy(am), deepcopy(bm)
        m = ntuple(_ -> rand(FT, n, n, nz), 6)
        mn = ntuple(_ -> rand(FT, n, n, nz), 6)
        target = zeros(ft.nc, nz)
        for k in 1:nz, c in 1:ft.nc
            i, j, p = G.cs_panel_ij(c, n)
            target[c, k] = (Float64(m[p][i, j, k])-Float64(mn[p][i, j, k]))*0.125
        end
        for k in 1:nz
            kw = (; tol=1e-12, max_iter=2000, project_every=10)
            d1 = MC.balance_cs_level!(k, am, bm, m, mn, ft, degree, MC.CSPoissonScratch(ft.nc), 0.125; kw...)
            d2 = MC.balance_cs_level!(k, aa, bb, MC._ArrayTarget(target), m, ft, degree, MC.CSPoissonScratch(ft.nc); kw...)
            @test isequal(d1, d2)
        end
        @test all(reinterpret(UInt8, am[p]) == reinterpret(UInt8, aa[p]) for p in 1:6)
        @test all(reinterpret(UInt8, bm[p]) == reinterpret(UInt8, bb[p]) for p in 1:6)
    end
end

const PP = AtmosTransport.Preprocessing
const CO = PP.Common
const CS = PP.CubedSphereTarget
const LL = PP.LatLonTarget
const GRAV = AtmosTransport.Parameters.STANDARD_GRAVITY

@testset "Direct CS layer forcing, guards, seams and dry closure" begin
    for convention in ("gnomonic", "geos_native")
        grid = PP.build_target_geometry(Dict("type"=>"cubed_sphere", "Nc"=>4,
            "panel_convention"=>convention), Float64)
        ft, nz = grid.face_table, 3
        areas = G.cell_areas(grid.mesh)
        factor = MC.DirectCSPoisson(ft)
        Random.seed!(10)
        u = randn(ft.nf,nz).*1e6
        v = zeros(0,nz)
        # Internal interfaces have arbitrary, independent W. Column convergence
        # remains modest in pressure units so the pre-check is meaningful.
        W = hcat(zeros(ft.nc), randn(ft.nc,2).*1e5, zeros(ft.nc))
        m = (areas./GRAV)*[10000. 30000. 60000.]
        endpoint = (; u,v,W,m,q=zeros(ft.nc,nz),ps=fill(100000.,ft.nc))
        window = CO.balance_window_layers!(PP.TM5SpectralFluxes(),endpoint,endpoint,
            grid,factor,3600.,areas)
        target = -diff(W;dims=2)
        target .-= (areas./sum(areas))*sum(target;dims=1)
        residual = LL.tm5_horizontal_convergence(window.u,window.v,grid)-target
        @test maximum(abs,residual) < 1e-6
        @test all(d->d.post.rms<=1e-7 && d.post.maximum<=1e-6 && 1<=d.passes<=3, window.diagnostics)
        am,bm = CS.tm5_cs_directional_fluxes(window.u,grid)
        # Mirrors must be exact for every orientation, including sign flips.
        for f in 1:ft.nf
            p=ft.mirror_panel[f]; p==0 && continue
            canonical = ft.face_dir[f]==1 ? am[ft.face_panel[f]] : bm[ft.face_panel[f]]
            mirror = ft.mirror_dir[f]==1 ? am[p] : bm[p]
            @test mirror[ft.mirror_idx_i[f],ft.mirror_idx_j[f],:] ==
                ft.mirror_sign[f].*canonical[ft.face_idx_i[f],ft.face_idx_j[f],:]
        end
        broken = merge(endpoint,(;u=u.*1e10))
        @test_throws ErrorException CO.balance_window_layers!(PP.TM5SpectralFluxes(),broken,
            broken,grid,factor,3600.,areas)
        dry = copy(u)
        LL.dry_target_fluxes!(dry,v,fill(0.01,ft.nc,nz),grid)
        @test dry ≈ u.*0.99 rtol=1e-15
        # Same arbitrary face rates: direct and PCG impose the same target.
        direct_am,direct_bm = CS.tm5_cs_directional_fluxes(u,grid)
        pcg_am,pcg_bm = deepcopy(direct_am),deepcopy(direct_bm)
        outward = MC._ArrayTarget(-target)
        masses = ntuple(p->reshape(copy(m[(p-1)*grid.Nc^2+1:p*grid.Nc^2,:]),grid.Nc,grid.Nc,nz),6)
        degree = MC.cs_cell_face_degree(ft)
        for k in 1:nz
            MC.balance_cs_level!(k,direct_am,direct_bm,outward,ft,factor,MC.CSPoissonScratch(ft.nc))
            MC.balance_cs_level!(k,pcg_am,pcg_bm,outward,masses,ft,degree,MC.CSPoissonScratch(ft.nc);
                tol=1e-12,max_iter=2000,project_every=10)
        end
        @test all(isapprox(direct_am[p],pcg_am[p];rtol=1e-10,atol=1e-5) for p in 1:6)
        @test all(isapprox(direct_bm[p],pcg_bm[p];rtol=1e-10,atol=1e-5) for p in 1:6)
    end
end

# A test-local solver verifies the production guard/pass loop without adding
# a failure hook to source code.
struct DelayedCSBalance{S}
    solver::S
    calls::Base.RefValue{Int}
    recover::Bool
end
function LL.balance_tm5_layers!(u,v,target,grid::CO.CubedSphereTargetGeometry,delayed::DelayedCSBalance)
    delayed.calls[] += 1
    delayed.recover && delayed.calls[] % 3 == 0 && LL.balance_tm5_layers!(u,v,target,grid,delayed.solver)
    return nothing
end
@testset "TM5 CS three-pass policy and exhausted post guard" begin
    grid=PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>2),Float64)
    ft=grid.face_table;areas=G.cell_areas(grid.mesh)
    u=reshape(sin.(1:ft.nf),:,1);v=zeros(0,1)
    endpoint=(;u,v,m=reshape(areas.*(100000/GRAV),:,1),q=zeros(ft.nc,1),
        W=zeros(ft.nc,2),ps=fill(100000.,ft.nc))
    pre=LL.tm5_balance_diagnostics(u,v,endpoint,endpoint,grid,3600.,areas)
    u .*= 5e-6/pre.rms
    for recover in (false,true)
        delayed=DelayedCSBalance(MC.DirectCSPoisson(ft),Ref(0),recover)
        if recover
            result=CO.balance_window_layers!(PP.TM5SpectralFluxes(),endpoint,endpoint,grid,delayed,3600.,areas)
            @test all(d->d.passes==3,result.diagnostics)
            @test delayed.calls[]==6
        else
            @test_throws ErrorException CO.balance_window_layers!(PP.TM5SpectralFluxes(),endpoint,endpoint,grid,delayed,3600.,areas)
            @test delayed.calls[]==3
        end
    end
end
