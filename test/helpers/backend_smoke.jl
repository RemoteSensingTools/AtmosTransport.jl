# Small complete models shared by CPU validation and opt-in backend diagnostics.
module BackendSmoke
using AtmosTransport, Adapt, Test
using AtmosTransport.State: CubedSphereField
using AtmosTransport.Operators.Advection: FV3ScalarProfile
const AT = AtmosTransport

function smoke_model(FT, cs, convection; scheme = UpwindScheme(), backend = "cpu")
    # Use the same configuration resolution and runtime validation as the
    # production runner, then allocate model workspaces on the selected device.
    arch = AT.Models.DrivenRunner._cfg_architecture(
        Dict("architecture" => Dict("backend" => backend)))
    AT.Architectures.ensure_runtime!(arch)
    AT.Architectures.assert_float_type!(arch, FT)
    adapter = AT.Architectures.array_adapter(arch)
    nx, ny, nz, hp = 4, 3, 6, 3
    mesh = cs ? CubedSphereMesh(; FT, Nc = nx, Hp = hp) : LatLonMesh(; FT, Nx = nx, Ny = ny)
    vc = HybridSigmaPressure(fill(FT(100), nz + 1), FT.(range(0, 1; length = nz + 1)))
    grid = AtmosGrid(mesh, vc, arch; FT)
    shape = cs ? (nx + 2hp, nx + 2hp, nz) : (nx, ny, nz)
    mass = fill(FT(1e16), shape)
    uniform = mass .* FT(4e-4)
    gradient = [mass[i,j,k] * FT(1e-4 * (1 + 0.1i + 0.2j + 0.3k))
                for i in axes(mass,1), j in axes(mass,2), k in 1:nz]
    state = cs ? CubedSphereState(DryBasis, mesh, ntuple(_ -> copy(mass), 6);
                                  uniform = ntuple(_ -> copy(uniform), 6),
                                  gradient = ntuple(_ -> copy(gradient), 6)) :
                 CellState(mass; uniform, gradient)
    fluxes = allocate_face_fluxes(mesh, nz; FT, basis = DryBasis)
    # Closed four-cell circulation: nonzero horizontal transport, zero net
    # divergence and no boundary flux. Both topologies must conserve mass.
    for p in 1:(cs ? 6 : 1)
        am, bm = cs ? (fluxes.am[p], fluxes.bm[p]) : (fluxes.am, fluxes.bm)
        h = cs ? hp : 0
        am[h+2,h+1,:] .= FT(1e14); am[h+2,h+2,:] .= -FT(1e14)
        bm[h+1,h+2,:] .= -FT(1e14); bm[h+2,h+2,:] .= FT(1e14)
        # Closed x-z circulation between levels 2 and 3. Vertical transport
        # is nonzero, with horizontal return flow preserving zero divergence
        # and closed top/bottom boundaries (including for the uniform tracer).
        cm = cs ? fluxes.cm[p] : fluxes.cm
        am[h+3,h+3,2] += FT(1e14); am[h+3,h+3,3] -= FT(1e14)
        cm[h+2,h+3,3] = -FT(1e14); cm[h+3,h+3,3] = FT(1e14)
    end
    cols = cs ? (nx, nx) : (nx, ny)
    rates = ntuple(_ -> zeros(FT, cols..., nz), 4)
    rates[1][:,:,nz] .= FT(0.02); rates[2][:,:,2] .= FT(0.02)
    # Include a closed downdraft to cover general LU, not just Hessenberg LU.
    rates[3][:,:,3] .= FT(0.005); rates[4][:,:,nz] .= FT(0.005)
    fields = NamedTuple{(:entu, :detu, :entd, :detd)}(cs ? map(a -> ntuple(_ -> copy(a), 6), rates) : rates)
    cmfmc = zeros(FT, cols..., nz + 1)
    cmfmc[:,:,3:nz-1] .= FT(0.01)
    dtrain = max.(zero(FT), cmfmc[:,:,2:end] .- cmfmc[:,:,1:end-1])
    forcing = ConvectionForcing(cs ? ntuple(_ -> copy(cmfmc),6) : cmfmc,
                                cs ? ntuple(_ -> copy(dtrain),6) : dtrain, fields)
    kz = ProfileKzField(fill(FT(10), nz))
    diffusion = ImplicitVerticalDiffusion(; kz_field = cs ? CubedSphereField(ntuple(_ -> kz,6)) : kz)
    state = Adapt.adapt(adapter, state)
    fluxes = Adapt.adapt(adapter, fluxes)
    forcing = Adapt.adapt(adapter, forcing)
    diffusion = Adapt.adapt(adapter, diffusion)
    model = TransportModel(state, fluxes, grid, scheme; convection, convection_forcing = forcing, diffusion)
    AT.Models.DrivenRunner._assert_gpu_residency!(model.state, arch)
    dz = model.workspace.diffusion_ws.layer_thickness
    foreach(a -> fill!(a, FT(100)), cs ? dz : (dz,))
    return model
end

function host_fields(model)
    state = model.state
    cs = state isa CubedSphereState
    arrays = cs ? state.tracers_raw : (state.tracers_raw,)
    masses = cs ? state.air_mass : (state.air_mass,)
    h = cs ? state.halo_width : 0
    inside(a) = Array(a)[h+1:end-h, h+1:end-h, ntuple(_ -> Colon(), ndims(a)-2)...]
    return map(inside, arrays), map(inside, masses)
end

function check_backend_smoke(array_type; float_types = (Float32, Float64), backend = "cpu")
    for FT in float_types, cs in (false, true),
        op in (TM5Convection(), TM5Convection(; use_collab_lu = true), CMFMCConvection()),
        scheme in (UpwindScheme(), cs ? PPMScheme(; vertical = FV3ScalarProfile()) : PPMScheme())
        @testset "$FT cs=$cs $(typeof(op)) collab=$(op isa TM5Convection && op.use_collab_lu) $(typeof(scheme))" begin
            cpu = smoke_model(FT, cs, op; scheme)
            candidate = smoke_model(FT, cs, op; scheme, backend)
            initial, _ = host_fields(cpu)
            @test AT.Architectures.backend_name(candidate.grid.arch) === Symbol(backend)
            @test AT.Architectures.array_adapter(candidate.grid.arch) === array_type
            @test AT.Architectures.array_adapter_for(candidate.state.air_mass) === array_type
            cm = cs ? candidate.fluxes.cm : (candidate.fluxes.cm,)
            @test all(a -> any(!iszero, Array(a)), cm)
            # This paired run makes disabling vertical transport observable,
            # even while convection and diffusion also alter the gradient.
            no_vertical = smoke_model(FT, cs, op; scheme)
            foreach(a -> fill!(a, zero(FT)), cs ? no_vertical.fluxes.cm : (no_vertical.fluxes.cm,))
            step!(no_vertical, FT(60))
            step!(cpu, FT(60))
            with_z, _ = host_fields(cpu)
            without_z, _ = host_fields(no_vertical)
            @test all(p -> with_z[p][:,:,:,2] != without_z[p][:,:,:,2], eachindex(with_z))
            step!(candidate, FT(60))
            for _ in 2:3
                step!(cpu, FT(60)); step!(candidate, FT(60))
            end
            expected, air = host_fields(cpu)
            actual, device_air = host_fields(candidate)
            tol = FT === Float32 ? 32eps(FT) : 2e-12
            for p in eachindex(actual)
                @test all(isfinite, actual[p])
                @test actual[p] ≈ expected[p] rtol = tol
                @test device_air[p] ≈ air[p] rtol = tol
                for t in 1:2
                    @test sum(Float64, actual[p][:,:,:,t]) ≈ sum(Float64, initial[p][:,:,:,t]) rtol = tol
                end
                @test actual[p][:,:,:,1] ./ device_air[p] ≈ fill(FT(4e-4), size(air[p])) rtol = tol
                @test actual[p][:,:,:,2] != initial[p][:,:,:,2]
            end
        end
    end
end
end
