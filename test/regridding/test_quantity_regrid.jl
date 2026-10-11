# ---------------------------------------------------------------------------
# test_quantity_regrid.jl
#
# Regridding by quantity kind (`src/Regridding/quantity_regrid.jl`):
#   1. extensive fields (per-cell totals) keep their global sum to 1e-13 on
#      LL ↔ CS and through `IdentityRegrid`; intensive fields are the plain
#      three-argument call; vector and flux kinds throw;
#   2. `regrid_conservation_report`, which checks both leading dimensions;
#   3. `regrid_2d_to_cs_panels!` / `regrid_3d_to_cs_panels!` give the same
#      bits as before the per-kind density correction moved from
#      Preprocessing (`_density_correct!`) to `apply_regridder!(…, kind)`:
#      the legacy helpers below are verbatim copies from commit 5a9930fb;
#   4. the `scratch` keyword: a caller-provided density buffer gives the same
#      bits, is validated, and keeps the preprocessing helpers allocation-free
#      after warm-up.
# ---------------------------------------------------------------------------

using Test
using Random
using AtmosTransport
using AtmosTransport.Regridding
using AtmosTransport.Quantities: IntensiveCellField, ExtensiveCellField,
                                 HorizontalVectorField, HorizontalFluxField
using AtmosTransport.Grids: GnomonicPanelConvention, GMAOCubedSphereDefinition

const QPP = AtmosTransport.Preprocessing

qr_bits(x::Float32) = reinterpret(UInt32, x)
qr_bits(x::Float64) = reinterpret(UInt64, x)
qr_bitwise(a, b) = size(a) == size(b) && all(qr_bits.(a) .== qr_bits.(b))

# --- cs_transport_helpers.jl at 5a9930fb, verbatim (renamed) -----------------
legacy_density_correct!(buf, ::Any, ::Any, ::IntensiveCellField) = buf

function legacy_density_correct!(buf::AbstractMatrix, areas, op::F,
                                 ::ExtensiveCellField) where F
    @inbounds for k in axes(buf, 2), i in eachindex(areas)
        buf[i, k] = op(buf[i, k], areas[i])
    end
    return buf
end

function legacy_density_correct!(buf::AbstractVector, areas, op::F,
                                 ::ExtensiveCellField) where F
    @inbounds for i in eachindex(areas)
        buf[i] = op(buf[i], areas[i])
    end
    return buf
end

function legacy_regrid_3d_to_cs_panels!(panels, regridder, src_3d, ws, Nc, kind)
    Nz = size(src_3d, 3)
    copyto!(ws.src_flat_3d, reshape(src_3d, size(ws.src_flat_3d)...))
    legacy_density_correct!(ws.src_flat_3d, regridder.src_areas, /, kind)
    apply_regridder!(ws.dst_flat_3d, regridder, ws.src_flat_3d)
    legacy_density_correct!(ws.dst_flat_3d, regridder.dst_areas, *, kind)
    return QPP.unpack_flat_to_panels_3d!(panels, ws.dst_flat_3d, Nc, Nz)
end

function legacy_regrid_2d_to_cs_panels!(panels, regridder, src_2d, ws, Nc, kind)
    copyto!(ws.src_flat_2d, reshape(src_2d, size(ws.src_flat_2d)...))
    legacy_density_correct!(ws.src_flat_2d, regridder.src_areas, /, kind)
    apply_regridder!(ws.dst_flat_2d, regridder, ws.src_flat_2d)
    legacy_density_correct!(ws.dst_flat_2d, regridder.dst_areas, *, kind)
    return QPP.unpack_flat_to_panels_2d!(panels, ws.dst_flat_2d, Nc)
end

# Bytes allocated by warm calls of the preprocessing regrid helpers.
function qr_warm_allocations(panels3, panels2, reg, data3, data2, ws, Nc, kind)
    QPP.regrid_3d_to_cs_panels!(panels3, reg, data3, ws, Nc, kind)
    QPP.regrid_2d_to_cs_panels!(panels2, reg, data2, ws, Nc, kind)
    a3 = @allocated QPP.regrid_3d_to_cs_panels!(panels3, reg, data3, ws, Nc, kind)
    a2 = @allocated QPP.regrid_2d_to_cs_panels!(panels2, reg, data2, ws, Nc, kind)
    return a3, a2
end

@testset "regridding by quantity kind" begin
    Nx, Ny, Nc, Nz = 36, 18, 4, 3
    ll = LatLonMesh(Nx = Nx, Ny = Ny)
    cs = CubedSphereMesh(Nc = Nc, convention = GnomonicPanelConvention())
    r = build_regridder(ll, cs; normalize = false)
    rt = build_regridder(cs, ll; normalize = false)
    rng = MersenneTwister(42)

    @testset "extensive totals conserved to 1e-13" begin
        for reg in (r, rt)
            n_src, n_dst = length(reg.src_areas), length(reg.dst_areas)
            mass = 1e12 .* (0.5 .+ rand(rng, n_src))
            mass_copy = copy(mass)
            out = zeros(n_dst)
            apply_regridder!(out, reg, mass, ExtensiveCellField())
            @test mass == mass_copy                               # src untouched
            @test abs(sum(out) / sum(mass) - 1) < 1e-13
            rep = regrid_conservation_report(reg, mass, out, ExtensiveCellField())
            @test rep.rel_error < 1e-13
            @test rep.src_total == sum(Float64, mass) && rep.dst_total == sum(Float64, out)
            # Several columns at once; each column is the vector call.
            masses = 1e12 .* (0.5 .+ rand(rng, n_src, Nz))
            outs = zeros(n_dst, Nz)
            apply_regridder!(outs, reg, masses, ExtensiveCellField())
            @test regrid_conservation_report(reg, masses, outs, ExtensiveCellField()).rel_error < 1e-13
            for k in 1:Nz
                one_col = zeros(n_dst)
                apply_regridder!(one_col, reg, masses[:, k], ExtensiveCellField())
                @test one_col ≈ outs[:, k] rtol = 1e-14
            end
        end
        # Identity: the density round trip returns the cell totals.
        ident = build_regridder(cs, CubedSphereMesh(Nc = Nc, convention = GnomonicPanelConvention()))
        @test ident isa IdentityRegrid
        mass = 1e12 .* (0.5 .+ rand(rng, 6 * Nc^2))
        out = zeros(6 * Nc^2)
        apply_regridder!(out, ident, mass, ExtensiveCellField())
        @test out ≈ mass rtol = 4eps()
    end

    @testset "intensive, vector and flux kinds" begin
        q = randn(rng, Nx * Ny, Nz)
        plain = zeros(6 * Nc^2, Nz)
        tagged = zeros(6 * Nc^2, Nz)
        apply_regridder!(plain, r, q)
        apply_regridder!(tagged, r, q, IntensiveCellField())
        @test qr_bitwise(plain, tagged)
        rep = regrid_conservation_report(r, q, tagged, IntensiveCellField())
        @test rep.rel_error < 1e-12
        @test rep.src_total ≈ sum(q .* r.src_areas)
        @test_throws ArgumentError apply_regridder!(plain, r, q, HorizontalVectorField())
        @test_throws ArgumentError apply_regridder!(plain, r, q, HorizontalFluxField())
        @test_throws ArgumentError regrid_conservation_report(r, q, plain, HorizontalFluxField())
        @test_throws DimensionMismatch apply_regridder!(zeros(5), r, zeros(Nx * Ny), ExtensiveCellField())
        # Both kinds check both leading dimensions against the regridder.
        for kind in (IntensiveCellField(), ExtensiveCellField())
            @test_throws DimensionMismatch regrid_conservation_report(r, zeros(5), zeros(6 * Nc^2), kind)
            @test_throws DimensionMismatch regrid_conservation_report(r, zeros(Nx * Ny), zeros(5), kind)
            @test_throws DimensionMismatch regrid_conservation_report(r, zeros(Nx * Ny, 2), zeros(5, 2), kind)
        end
        zero_rep = regrid_conservation_report(r, zeros(Nx * Ny), zeros(6 * Nc^2), ExtensiveCellField())
        @test zero_rep.rel_error == 0.0
        @test regrid_conservation_report(r, zeros(Nx * Ny), ones(6 * Nc^2),
                                         ExtensiveCellField()).rel_error == Inf
    end

    @testset "regrid_*_to_cs_panels! bit-identical to the pre-move helpers" begin
        n_src, n_dst = Nx * Ny, 6 * Nc^2
        for FT in (Float32, Float64), kind in (IntensiveCellField(), ExtensiveCellField())
            ws_new = QPP.allocate_cs_preprocess_workspace(Nc, Nx, Ny, Nz, n_src, n_dst, FT)
            ws_old = QPP.allocate_cs_preprocess_workspace(Nc, Nx, Ny, Nz, n_src, n_dst, FT)
            for data in (FT.(1e9 .+ 1e13 .* rand(rng, Nx, Ny, Nz)),       # air-mass-like
                         FT.(randn(rng, Nx, Ny, Nz)))                     # signed
                new3 = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
                old3 = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
                QPP.regrid_3d_to_cs_panels!(new3, r, data, ws_new, Nc, kind)
                legacy_regrid_3d_to_cs_panels!(old3, r, data, ws_old, Nc, kind)
                @test all(qr_bitwise(new3[p], old3[p]) for p in 1:6)

                new2 = ntuple(_ -> zeros(FT, Nc, Nc), 6)
                old2 = ntuple(_ -> zeros(FT, Nc, Nc), 6)
                QPP.regrid_2d_to_cs_panels!(new2, r, data[:, :, 1], ws_new, Nc, kind)
                legacy_regrid_2d_to_cs_panels!(old2, r, data[:, :, 1], ws_old, Nc, kind)
                @test all(qr_bitwise(new2[p], old2[p]) for p in 1:6)
            end
        end
        # The identity passthrough (same-mesh preprocessing) as well.
        ident = build_regridder(cs, CubedSphereMesh(Nc = Nc, convention = GnomonicPanelConvention()))
        for FT in (Float32, Float64)
            ws_new = QPP.allocate_cs_preprocess_workspace(Nc, 6 * Nc, Nc, Nz, n_dst, n_dst, FT)
            ws_old = QPP.allocate_cs_preprocess_workspace(Nc, 6 * Nc, Nc, Nz, n_dst, n_dst, FT)
            data = FT.(1e9 .+ 1e13 .* rand(rng, 6 * Nc, Nc, Nz))
            new3 = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
            old3 = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
            QPP.regrid_3d_to_cs_panels!(new3, ident, data, ws_new, Nc, ExtensiveCellField())
            legacy_regrid_3d_to_cs_panels!(old3, ident, data, ws_old, Nc, ExtensiveCellField())
            @test all(qr_bitwise(new3[p], old3[p]) for p in 1:6)
        end
        @test_throws ArgumentError QPP.regrid_2d_to_cs_panels!(
            ntuple(_ -> zeros(Nc, Nc), 6), r, zeros(Nx, Ny),
            QPP.allocate_cs_preprocess_workspace(Nc, Nx, Ny, 1, n_src, n_dst, Float64),
            Nc, HorizontalVectorField())
        @test !isdefined(QPP, :_density_correct!)                # one implementation
    end

    @testset "caller-provided density scratch" begin
        n_src, n_dst = Nx * Ny, 6 * Nc^2
        for FT in (Float32, Float64)
            mass = FT.(1e9 .+ 1e13 .* rand(rng, n_src, Nz))
            fresh = zeros(FT, n_dst, Nz)
            reused = zeros(FT, n_dst, Nz)
            scratch = zeros(FT, n_src)
            apply_regridder!(fresh, r, mass, ExtensiveCellField())
            apply_regridder!(reused, r, mass, ExtensiveCellField(); scratch)
            @test qr_bitwise(fresh, reused)
            fresh1, reused1 = zeros(FT, n_dst), zeros(FT, n_dst)
            apply_regridder!(fresh1, r, mass[:, 1], ExtensiveCellField())
            apply_regridder!(reused1, r, mass[:, 1], ExtensiveCellField(); scratch)
            @test qr_bitwise(fresh1, reused1)
            # Intensive regrids accept and ignore it.
            plain, tagged = zeros(FT, n_dst, Nz), zeros(FT, n_dst, Nz)
            apply_regridder!(plain, r, mass)
            apply_regridder!(tagged, r, mass, IntensiveCellField(); scratch)
            @test qr_bitwise(plain, tagged)
        end
        mass = rand(rng, n_src, Nz)
        out = zeros(6 * Nc^2, Nz)
        @test_throws DimensionMismatch apply_regridder!(out, r, mass, ExtensiveCellField();
                                                        scratch = zeros(n_src - 1))
        @test_throws ArgumentError apply_regridder!(out, r, mass, ExtensiveCellField();
                                                    scratch = zeros(Float32, n_src))
        @test_throws ArgumentError apply_regridder!(out, r, mass, ExtensiveCellField();
                                                    scratch = view(mass, :, 1))          # aliases src
        out_ll = zeros(n_src)                                     # CS → LL: n_dst > n_src
        @test_throws ArgumentError apply_regridder!(out_ll, rt, rand(rng, n_dst), ExtensiveCellField();
                                                    scratch = view(out_ll, 1:n_dst))     # aliases dst
        @test_throws ArgumentError apply_regridder!(out, r, mass, HorizontalFluxField();
                                                    scratch = zeros(n_src))

        # The preprocessing workspace carries the scratch: warm calls allocate
        # nothing, for both kinds (before, each extensive call allocated a
        # density vector of n_src values).
        for FT in (Float32, Float64), kind in (IntensiveCellField(), ExtensiveCellField())
            ws = QPP.allocate_cs_preprocess_workspace(Nc, Nx, Ny, Nz, n_src, n_dst, FT)
            @test length(ws.density_scratch) == n_src
            data3 = FT.(1e9 .+ 1e13 .* rand(rng, Nx, Ny, Nz))
            panels3 = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
            panels2 = ntuple(_ -> zeros(FT, Nc, Nc), 6)
            @test qr_warm_allocations(panels3, panels2, r, data3, data3[:, :, 1], ws, Nc, kind) == (0, 0)
            @test_throws DimensionMismatch QPP.regrid_2d_to_cs_panels!(panels2, r, zeros(FT, Nx, Ny + 1),
                                                                      ws, Nc, kind)
            @test_throws DimensionMismatch QPP.regrid_3d_to_cs_panels!(panels3, r, zeros(FT, Nx, Ny, Nz + 1),
                                                                      ws, Nc, kind)
        end
    end
end
