# Set an explicit CUDA_VISIBLE_DEVICES and opt in before running this file.
using Test
if get(ENV, "ATMOSTR_RUN_CS_SEAMS_GPU_TESTS", "0") != "1"
    @info "Skipping opt-in cubed-sphere seam GPU tests"
else
    using CUDA, AtmosTransport
    CUDA.allowscalar(false)
    expected = get(ENV, "ATMOSTR_CS_SEAMS_GPU_NAME", "A100")
    isempty(expected) && error("ATMOSTR_CS_SEAMS_GPU_NAME must identify a device")
    @assert occursin(expected, CUDA.name(CUDA.device())) "Unexpected GPU for seam tests"
    include(joinpath(@__DIR__, "..", "helpers", "cs_seam_exchange.jl"))

    @testset "CUDA conservative PPM palindrome through 65 tracers" begin
        for FT in (Float32, Float64), Nc in (5, 35), Nt in (1, 7, 32, 65), convention in
                (CSSeamGrids.GnomonicPanelConvention(), CSSeamGrids.GEOSNativePanelConvention())
            mesh, mass, tracer, am, bm = cs_seam_fixture(FT, Nc, 3, Nt, convention)
            N = Nc + 2mesh.Hp
            cm = ntuple(_ -> zeros(FT, N, N, 4), 6)
            device_mass, device_tracer = map(CuArray, mass), map(CuArray, tracer)
            device_am, device_bm, device_cm = map(CuArray, am), map(CuArray, bm), map(CuArray, cm)
            cpu_ws = CSSeamAdv.CSAdvectionWorkspace(mesh, mass[1]; n_tracers=Nt)
            gpu_ws = CSSeamAdv.CSAdvectionWorkspace(mesh, device_mass[1]; n_tracers=Nt)
            initial = [sum(sum(Float64, cs_seam_interior(view(a, :, :, :, t), mesh)) for a in tracer) for t in 1:Nt]
            scale = [sum(sum(abs, Float64.(cs_seam_interior(view(a, :, :, :, t), mesh))) for a in tracer) for t in 1:Nt]
            for _ in 1:2    # two model steps, each opening with the step-start halo refresh
                CSSeamAdv.fill_cs_step_start_halos!(tracer, mass, mesh)
                CSSeamAdv.fill_cs_step_start_halos!(device_tracer, device_mass, mesh)
                CSSeamAdv.strang_split_cs_mt!(tracer, mass, am, bm, cm, mesh, PPMScheme(), cpu_ws; subcycle_count=1)
                CSSeamAdv.strang_split_cs_mt!(device_tracer, device_mass, device_am, device_bm,
                    device_cm, mesh, PPMScheme(), gpu_ws; subcycle_count=1)
            end
            actual, actual_mass = map(Array, device_tracer), map(Array, device_mass)
            tol = FT == Float32 ? 3e-5 : 3e-13
            @test cs_seam_maxdiff(actual_mass, mass, mesh) < tol
            for t in 1:Nt
                panels = map(a -> view(a, :, :, :, t), actual)
                @test cs_seam_maxdiff(panels, map(a -> view(a, :, :, :, t), tracer), mesh) < tol
                @test all(p -> all(isfinite, cs_seam_interior(p, mesh)), panels)
                final = sum(sum(Float64, cs_seam_interior(p, mesh)) for p in panels)
                @test abs(final - initial[t]) / scale[t] < (FT == Float32 ? 3e-7 : 3e-14)
            end
        end
    end
    @testset "CUDA paired-seam adjoints match CPU" begin
        Adj = AtmosTransport.Adjoints
        for FT in (Float32, Float64), axis in (1, 2), convention in
                (CSSeamGrids.GnomonicPanelConvention(), CSSeamGrids.GEOSNativePanelConvention()),
                scheme in (UpwindScheme(), PPMScheme(CSSeamAdv.NoLimiter()), PPMScheme())
            mesh, m, packed, am, bm = cs_seam_fixture(FT, 5, 3, 2, convention)
            rm = map(a -> a[:, :, :, 2], packed)
            rng = MersenneTwister(2391)
            gradient = map(a -> zero(a), rm)
            for p in 1:6
                cs_seam_interior(gradient[p], mesh) .= randn(rng, FT, mesh.Nc, mesh.Nc, 3)
            end
            flux, symbol = axis == 1 ? (am, :x) : (bm, :y)
            gm, gr, gf = map(CuArray, m), map(CuArray, rm), map(CuArray, flux)
            actual = map(CuArray, gradient)
            cpu_ws = Adj.CSAdjointWorkspace(mesh, m[1])
            gpu_ws = Adj.CSAdjointWorkspace(mesh, gm[1])
            if scheme isa PPMScheme{MonotoneLimiter}
                # The limiter branches the tape records: one fused launch for
                # the six panels. Keep the exact check as a diagnostic: if
                # GPU FMA changes a near-tie predicate, inspect the differing
                # cell's profile/slope predicates and its reverse weight
                # before attributing a gradient bug (see TP5 GPU acceptance).
                codes = CSSeamAdv.record_ppm_limiter_codes!(map(a -> similar(a, UInt8), m), m, rm,
                                                            symbol, scheme)
                gcodes = CSSeamAdv.record_ppm_limiter_codes!(map(a -> similar(a, UInt8), gm), gm, gr,
                                                             symbol, scheme)
                host_codes = map(Array, gcodes)
                for p in 1:6
                    mismatches = findall(host_codes[p] .!= codes[p])
                    if !isempty(mismatches)
                        @info "Limiter code mismatch: inspect near-tie predicates and reverse weights" FT axis p cells = first(mismatches, 10) count = length(mismatches)
                    end
                    @test host_codes[p] == codes[p]
                end
                CSSeamAdv._adjoint_scheme_sweep!(gradient, m, codes, flux, symbol, scheme, mesh, cpu_ws, FT(1))
                CSSeamAdv._adjoint_scheme_sweep!(actual, gm, gcodes, gf, symbol, scheme, mesh, gpu_ws, FT(1))
            else
                CSSeamAdv._adjoint_scheme_sweep!(gradient, m, flux, symbol, scheme, mesh, cpu_ws, FT(1))
                CSSeamAdv._adjoint_scheme_sweep!(actual, gm, gf, symbol, scheme, mesh, gpu_ws, FT(1))
            end
            Adj._adjoint_fill_panel_halos!(gradient, mesh)
            Adj._adjoint_fill_panel_halos!(actual, mesh)
            @test cs_seam_maxdiff(map(Array, actual), gradient, mesh) < (FT == Float32 ? 3e-5 : 2e-12)
        end
    end

end
