# Analytic and file ICs, conservative hybrid remapping and dry/moist packing.
# Flux-loader contracts live in test_surface_flux_ingestion.jl and native flux
# tests; the C90 real-data invariant lives in test/real_data/.
using Test
using Dates
using TOML
using Printf: @sprintf
import NCDatasets: NCDataset, defDim, defVar, nomissing

using AtmosTransport
using .AtmosTransport.Grids: cell_index, nrings, ring_longitudes
using .AtmosTransport.Models.InitialConditionIO: build_surface_flux_source,
                                                build_surface_flux_sources

const FT = Float64
const ICIO = AtmosTransport.Models.InitialConditionIO

_synthetic_file_ic_value(lon_deg, lat_deg) =
    3.5e-4 + 1e-6 * mod(lon_deg, 360.0) + 2e-6 * lat_deg

function _write_synthetic_ic_file(path::AbstractString; with_humidity::Bool = false,
                                  humidity_name::AbstractString = "QV",
                                  humidity_standard_name::Bool = true)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", 4)
        defDim(ds, "latitude", 2)
        defDim(ds, "level", 3)
        defDim(ds, "hlevel", 4)

        vlon = defVar(ds, "longitude", Float64, ("longitude",))
        vlat = defVar(ds, "latitude", Float64, ("latitude",))
        vlev = defVar(ds, "level", Float64, ("level",))
        vhlev = defVar(ds, "hlevel", Float64, ("hlevel",))
        vap = defVar(ds, "ap", Float64, ("hlevel",), attrib = Dict("units" => "Pa"))
        vbp = defVar(ds, "bp", Float64, ("hlevel",))
        vco2 = defVar(ds, "CO2", Float64, ("longitude", "latitude", "level"))
        vqv = with_humidity ?
              defVar(ds, humidity_name, Float64,
                     ("longitude", "latitude", "level")) : nothing

        lon = [-135.0, -45.0, 45.0, 135.0]
        lat = [45.0, -45.0]  # descending on purpose
        vlon[:] = lon
        vlat[:] = lat
        vlev[:] = [0.0, 1.0, 2.0]
        vhlev[:] = [0.0, 1.0, 2.0, 3.0]
        vap[:] = [0.0, 0.0, 0.0, 0.0]
        vbp[:] = [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]
        raw = Array{Float64}(undef, 4, 2, 3)
        for k in 1:3, j in 1:2, i in 1:4
            raw[i, j, k] = _synthetic_file_ic_value(lon[i], lat[j])
        end
        vco2[:, :, :] = raw
        vco2.attrib["units"] = "mol mol-1"
        if vqv !== nothing
            for k in 1:3
                vqv[:, :, k] .= (0.35, 0.08, 0.001)[k]
            end
            humidity_standard_name &&
                (vqv.attrib["standard_name"] = "specific_humidity")
            vqv.attrib["units"] = "kg kg-1"
        end
    finally
        close(ds)
    end
    return nothing
end

function _write_synthetic_cs_native_ic_file(path::AbstractString; Nc=2, Nz=2)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "Xdim", Nc)
        defDim(ds, "Ydim", Nc)
        defDim(ds, "nf", 6)
        defDim(ds, "lev", Nz)
        defDim(ds, "ilev", Nz + 1)
        tracer = defVar(ds, "SIGNED", Float64,
                        ("Xdim", "Ydim", "nf", "lev"))
        # Equal level counts must retain the exact-copy path even when the
        # native file carries otherwise usable hybrid metadata.
        defVar(ds, "ap", Float64, ("ilev",), attrib = Dict("units" => "Pa"))[:] = zeros(Nz + 1)
        defVar(ds, "bp", Float64, ("ilev",))[:] = collect(range(1.0, 0.0; length = Nz + 1))
        values = Array{Float64}(undef, Nc, Nc, 6, Nz)
        for k in 1:Nz, p in 1:6, j in 1:Nc, i in 1:Nc
            values[i, j, p, k] = (isodd(i + j + p + k) ? -1.0 : 1.0) *
                                  (1e-6 + 1e-8 * (i + 2j + 3p + 4k))
        end
        tracer[:, :, :, :] = values
    finally
        close(ds)
    end
    return nothing
end

# A cs_native IC whose level k (file order) holds `level_values[k]` everywhere,
# with optional hybrid coefficients: `ap`/`bp` = `nothing` omits the variable,
# `ap_units` = `nothing` omits its units attribute.
function _write_hybrid_cs_native_ic_file(path::AbstractString; Nc, level_values,
                                         ap, bp, ap_units = "Pa")
    Nsrc = length(level_values)
    NCDataset(path, "c") do ds
        for (name, n) in (("Xdim", Nc), ("Ydim", Nc), ("nf", 6),
                          ("lev", Nsrc), ("ilev", Nsrc + 1))
            defDim(ds, name, n)
        end
        tracer = defVar(ds, "TRACER", Float64, ("Xdim", "Ydim", "nf", "lev"))
        if ap !== nothing
            attrib = ap_units === nothing ? Dict{String, Any}() :
                                            Dict{String, Any}("units" => ap_units)
            defVar(ds, "ap", Float64, ("ilev",); attrib)[:] = ap
        end
        bp === nothing || (defVar(ds, "bp", Float64, ("ilev",))[:] = bp)
        for k in 1:Nsrc
            tracer[:, :, :, k] .= level_values[k]
        end
    end
    return path
end

function _write_mismatched_cs_native_ic_file(path::AbstractString; Nc=2)
    Nsrc = 3
    NCDataset(path, "c") do ds
        for (name, n) in (("Xdim", Nc), ("Ydim", Nc), ("nf", 6),
                          ("lev", Nsrc), ("ilev", Nsrc + 1))
            defDim(ds, name, n)
        end
        raw = defVar(ds, "TRACER", Float64, ("Xdim", "Ydim", "nf", "lev"))
        humidity = defVar(ds, "QV", Float64, ("Xdim", "Ydim", "nf", "lev"))
        defVar(ds, "ap", Float64, ("ilev",), attrib = Dict("units" => "Pa"))[:] = zeros(Nsrc + 1)
        defVar(ds, "bp", Float64, ("ilev",))[:] = [1.0, 2 / 3, 1 / 3, 0.0]
        for k in 1:Nsrc
            raw[:, :, :, k] .= (6.0, 3.0, 0.0)[k]
            humidity[:, :, :, k] .= (0.0, 0.5, 0.0)[k]
        end
    end
    return nothing
end

@testset "InitialConditionIO" begin

    # ---------------------------- LL uniform + blob ------------------------
    @testset "build_initial_mixing_ratio — LatLonMesh" begin
        mesh = LatLonMesh(; Nx = 4, Ny = 3,
                          longitude = (0.0, 360.0),
                          latitude  = (-90.0, 90.0))
        Nz = 2
        air_mass = ones(FT, 4, 3, Nz)   # shape-only; builder does not read values

        # uniform
        q = build_initial_mixing_ratio(air_mass, mesh,
                                       Dict("kind" => "uniform",
                                            "background" => 1.5e-4))
        @test size(q) == size(air_mass)
        @test all(q .== FT(1.5e-4))
        grid = AtmosGrid(mesh, HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0]), CPU(); FT)
        @test build_initial_mixing_ratio(air_mass, grid,
            Dict("kind" => "uniform", "background" => 1.5e-4)) == q

        q_step = build_initial_mixing_ratio(air_mass, mesh,
                                            Dict("kind" => "latitude_step",
                                                 "south_value" => 4.0e-4,
                                                 "north_value" => 4.4e-4,
                                                 "split_lat_deg" => 0.0))
        @test size(q_step) == size(air_mass)
        @test all(q_step[:, 1, :] .== FT(4.0e-4))
        @test all(q_step[:, 2:3, :] .== FT(4.4e-4))

        # gaussian_blob — non-uniform profile centered at (0, 0)
        q2 = build_initial_mixing_ratio(air_mass, mesh,
                                        Dict("kind" => "gaussian_blob",
                                             "lon0_deg" => 0.0, "lat0_deg" => 0.0,
                                             "sigma_lon_deg" => 10.0,
                                             "sigma_lat_deg" => 10.0,
                                             "amplitude" => 1.0e-3,
                                             "background" => 4.0e-4))
        @test size(q2) == size(air_mass)
        @test minimum(q2) ≥ FT(4.0e-4)
        @test maximum(q2) > minimum(q2)   # non-trivial profile
    end

    # ---------------------------- RG uniform + blob ------------------------
    @testset "build_initial_mixing_ratio — ReducedGaussianMesh" begin
        latitudes = [-75.0, -25.0, 25.0, 75.0]
        nlon_per_ring = [4, 8, 8, 4]
        mesh = ReducedGaussianMesh(latitudes, nlon_per_ring; FT = FT)
        Nz = 2
        ncells_ = ncells(mesh)
        air_mass = ones(FT, ncells_, Nz)

        q = build_initial_mixing_ratio(air_mass, mesh,
                                       Dict("kind" => "uniform",
                                            "background" => 2.0e-4))
        @test size(q) == (ncells_, Nz)
        @test all(q .== FT(2.0e-4))

        q_step = build_initial_mixing_ratio(air_mass, mesh,
                                            Dict("kind" => "hemisphere_step",
                                                 "south_value" => 4.0e-4,
                                                 "north_value" => 4.4e-4))
        @test all(q_step[1:12, :] .== FT(4.0e-4))
        @test all(q_step[13:end, :] .== FT(4.4e-4))

        q2 = build_initial_mixing_ratio(air_mass, mesh,
                                        Dict("kind" => "gaussian_blob",
                                             "lon0_deg" => 0.0, "lat0_deg" => 0.0,
                                             "sigma_lon_deg" => 15.0,
                                             "sigma_lat_deg" => 15.0,
                                             "amplitude" => 5.0e-4,
                                             "background" => 4.0e-4))
        @test minimum(q2) ≥ FT(4.0e-4)
        @test maximum(q2) > minimum(q2)
    end

    # ---------------------------- LL grid-level dispatch -------------------

    @testset "file-based initial conditions support latlon and reduced grids" begin
        mktempdir() do dir
            ic_path = joinpath(dir, "test_ic.nc")
            _write_synthetic_ic_file(ic_path)
            humid_ic_path = joinpath(dir, "test_ic_humid.nc")
            _write_synthetic_ic_file(humid_ic_path; with_humidity = true)

            init_cfg = Dict{String, Any}(
                "kind" => "file",
                "file" => ic_path,
                "variable" => "CO2",
            )
            humid_cfg = merge(init_cfg, Dict("file" => humid_ic_path,
                                              "humidity_variable" => "QV"))
            humid_source = ICIO._load_file_initial_condition_source(humid_cfg, Float64, 2)
            @test humid_source.qv !== nothing
            @test humid_source.qv[1, 1, :] == [0.35, 0.08, 0.001]

            vertical = HybridSigmaPressure(Float64[0, 0, 0], Float64[0, 0.5, 1])

            latlon_mesh = LatLonMesh(; FT = Float64, Nx = 4, Ny = 2)
            latlon_grid = AtmosGrid(latlon_mesh, vertical, CPU(); FT = Float64)
            air_mass_ll = ones(Float64, 4, 2, 2)
            ps_ll = fill(9.0e4, 4, 2)
            q_ll = build_initial_mixing_ratio(air_mass_ll, latlon_grid, init_cfg;
                                              surface_pressure = ps_ll)
            for j in 1:2, i in 1:4
                expected = _synthetic_file_ic_value(latlon_mesh.λᶜ[i], latlon_mesh.φᶜ[j])
                @test q_ll[i, j, 1] ≈ expected atol = 1e-12
                @test q_ll[i, j, 2] ≈ expected atol = 1e-12
            end

            reduced_mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 4]; FT = Float64)
            reduced_grid = AtmosGrid(reduced_mesh, vertical, CPU(); FT = Float64)
            air_mass_rg = ones(Float64, ncells(reduced_mesh), 2)
            ps_rg = fill(9.0e4, ncells(reduced_mesh))
            q_rg = build_initial_mixing_ratio(air_mass_rg, reduced_grid, init_cfg;
                                              surface_pressure = ps_rg)
            for j in 1:nrings(reduced_mesh)
                lons = ring_longitudes(reduced_mesh, j)
                lat = reduced_mesh.latitudes[j]
                for i in eachindex(lons)
                    c = cell_index(reduced_mesh, i, j)
                    expected = _synthetic_file_ic_value(lons[i], lat)
                    @test q_rg[c, 1] ≈ expected atol = 1e-12
                    @test q_rg[c, 2] ≈ expected atol = 1e-12
                end
            end
        end
    end

    # ---------------------------- pack_initial_tracer_mass ------------------

    # --------- plan 40 Commit 1c: CubedSphere IC + packer ---------------
    @testset "structured dry/moist tracer-mass conversion" begin
        meshes = (LatLonMesh(; Nx = 4, Ny = 3),
                  ReducedGaussianMesh([-75.0, -25.0, 25.0, 75.0], [4, 8, 8, 4]; FT))
        for mesh in meshes
            grid = AtmosGrid(mesh, HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0]), CPU(); FT)
            dims = mesh isa LatLonMesh ? (4, 3, 2) : (ncells(mesh), 2)
            air = fill(FT(1.2e10), dims)
            vmr = fill(FT(4.11e-4), dims)
            qv = fill(FT(0.02), dims)
            for basis in (DryBasis(), MoistBasis())
                packed = pack_initial_tracer_mass(grid, air, vmr; mass_basis = basis, qv)
                expected = basis isa DryBasis ? vmr .* air : vmr .* air .* (1 .- qv)
                @test packed == expected
            end
            @test pack_initial_tracer_mass(grid, air, vmr; mass_basis = DryBasis(), qv) ==
                  pack_initial_tracer_mass(grid, air, vmr; mass_basis = DryBasis())
            @test_throws ArgumentError pack_initial_tracer_mass(grid, air, vmr; mass_basis = MoistBasis())
            @test_throws DimensionMismatch pack_initial_tracer_mass(grid, air, vmr;
                mass_basis = MoistBasis(), qv = zeros(FT, 1))
        end
    end

    @testset "build_initial_mixing_ratio — CubedSphereMesh uniform" begin
        Nc = 4
        Hp = 1
        Nz = 3
        mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        # Halo-padded 6-panel air_mass tuple (matches TransportBinaryDriver layout).
        air_mass = ntuple(_ -> fill(FT(1e10), Nc + 2 * Hp, Nc + 2 * Hp, Nz), 6)

        vmr = build_initial_mixing_ratio(air_mass, grid,
                                         Dict("kind" => "uniform",
                                              "background" => 4.11e-4))
        @test vmr isa NTuple{6, Array{FT, 3}}
        @test all(size(vmr[p]) == (Nc, Nc, Nz) for p in 1:6)   # interior only
        @test all(all(vmr[p] .== FT(4.11e-4)) for p in 1:6)

        vmr_step = build_initial_mixing_ratio(air_mass, grid,
                                              Dict("kind" => "latitude_step",
                                                   "south_value" => 4.0e-4,
                                                   "north_value" => 4.4e-4))
        @test vmr_step isa NTuple{6, Array{FT, 3}}
        @test all(size(vmr_step[p]) == (Nc, Nc, Nz) for p in 1:6)
        step_vals = vcat((vec(vmr_step[p]) for p in 1:6)...)
        @test all(v -> v == FT(4.0e-4) || v == FT(4.4e-4), step_vals)
        @test minimum(step_vals) == FT(4.0e-4)
        @test maximum(step_vals) == FT(4.4e-4)

        vmr_blob = build_initial_mixing_ratio(air_mass, grid,
                                              Dict("kind" => "gaussian_blob",
                                                   "lon0_deg" => 0.0,
                                                   "lat0_deg" => 35.0,
                                                   "sigma_lon_deg" => 30.0,
                                                   "sigma_lat_deg" => 20.0,
                                                   "amplitude" => 8.0e-5,
                                                   "background" => 4.0e-4))
        @test vmr_blob isa NTuple{6, Array{FT, 3}}
        @test all(size(vmr_blob[p]) == (Nc, Nc, Nz) for p in 1:6)
        blob_vals = vcat((vec(vmr_blob[p]) for p in 1:6)...)
        @test minimum(blob_vals) ≥ FT(4.0e-4)
        @test maximum(blob_vals) > FT(4.0e-4)
        @test maximum(blob_vals) < FT(4.8e-4)

        # Unsupported kind errors with a helpful message
        @test_throws ArgumentError build_initial_mixing_ratio(air_mass, grid,
                                                              Dict("kind" => "bl_enhanced"))
    end

    @testset "cs_native preserves signed values by default" begin
        mktempdir() do dir
            Nc, Hp, Nz = 2, 1, 2
            path = joinpath(dir, "signed_cs_native.nc")
            _write_synthetic_cs_native_ic_file(path; Nc, Nz)

            mesh = CubedSphereMesh(; FT, Nc, Hp)
            vertical = HybridSigmaPressure(zeros(FT, Nz + 1),
                                           collect(FT, range(1.0, 0.0; length = Nz + 1)))
            grid = AtmosGrid(mesh, vertical, CPU(); FT)
            air_mass = ntuple(_ -> fill(FT(1e10), Nc + 2Hp, Nc + 2Hp, Nz), 6)
            base_cfg = Dict{String, Any}(
                "kind" => "cs_native",
                "file" => path,
                "variable" => "SIGNED",
                "vertical_order" => "surface_first",
            )

            signed = build_initial_mixing_ratio(air_mass, grid, base_cfg)
            @test minimum(vcat((vec(signed[p]) for p in 1:6)...)) < 0
            @test maximum(vcat((vec(signed[p]) for p in 1:6)...)) > 0
            # Equal level counts retain the pre-E7 cell-for-cell copy exactly.
            for k in 1:Nz, p in 1:6, j in 1:Nc, i in 1:Nc
                expected = (isodd(i + j + p + k) ? -1.0 : 1.0) *
                           (1e-6 + 1e-8 * (i + 2j + 3p + 4k))
                @test signed[p][i, j, k] === FT(expected)
            end

            clamped_cfg = copy(base_cfg)
            clamped_cfg["clamp_negative"] = true
            clamped = build_initial_mixing_ratio(air_mass, grid, clamped_cfg)
            @test minimum(vcat((vec(clamped[p]) for p in 1:6)...)) == 0
            @test maximum(vcat((vec(clamped[p]) for p in 1:6)...)) > 0
        end
    end

    @testset "cs_native hybrid coefficients: orientation, agreement and units" begin
        mktempdir() do dir
            Nc, Hp, Nz = 2, 1, 3
            mesh = CubedSphereMesh(; FT, Nc, Hp)
            # A normal top-first target: interface 1 is the model top.
            A_top = [100.0, 3000.0, 6000.0, 0.0]
            B_top = [0.0, 0.2, 0.6, 1.0]
            grid = AtmosGrid(mesh, HybridSigmaPressure(FT.(A_top), FT.(B_top)), CPU(); FT)
            air_mass = ntuple(_ -> fill(FT(1e10), Nc + 2Hp, Nc + 2Hp, Nz), 6)
            # Source levels surface-first: level 1 is the surface layer.
            surface_first_values = [1e-6, 2e-6, 3e-6]
            cfg_for(path) = Dict{String, Any}("kind" => "cs_native", "file" => path,
                                              "variable" => "TRACER")
            build(path) = build_initial_mixing_ratio(air_mass, grid, cfg_for(path))
            message(f) = try
                f()
                ""
            catch err
                err isa ArgumentError || err isa DimensionMismatch || rethrow()
                sprint(showerror, err)
            end

            # Opposite orientation, equal level counts: the source column is
            # reversed onto the target, so the surface value lands at k = Nz.
            # ap of 6000 Pa without a units attribute is unambiguously Pa.
            reversed = _write_hybrid_cs_native_ic_file(joinpath(dir, "surface_first.nc");
                Nc, level_values = surface_first_values,
                ap = reverse(A_top), bp = reverse(B_top), ap_units = nothing)
            vmr = build(reversed)
            for p in 1:6, k in 1:Nz
                @test all(vmr[p][:, :, k] .== FT(surface_first_values[Nz - k + 1]))
            end
            # A consistent vertical_order is accepted; a contradicting one is not.
            consistent = merge(cfg_for(reversed), Dict{String, Any}("vertical_order" => "surface_first"))
            @test build_initial_mixing_ratio(air_mass, grid, consistent) == vmr
            contradicting = merge(cfg_for(reversed), Dict{String, Any}("vertical_order" => "toa_first"))
            @test occursin("contradicts the hybrid coefficients",
                           message(() -> build_initial_mixing_ratio(air_mass, grid, contradicting)))

            # Same orientation as the target: copied level for level.
            same = _write_hybrid_cs_native_ic_file(joinpath(dir, "top_first.nc");
                Nc, level_values = reverse(surface_first_values), ap = A_top, bp = B_top)
            @test build(same) == vmr

            # Equal level counts with other coefficients are another vertical
            # grid: rejected, never copied cell for cell.
            other_bp = _write_hybrid_cs_native_ic_file(joinpath(dir, "other_bp.nc");
                Nc, level_values = surface_first_values,
                ap = reverse(A_top), bp = [1.0, 0.5, 0.2, 0.0])
            @test occursin("bp coefficients differ", message(() -> build(other_bp)))
            other_ap = _write_hybrid_cs_native_ic_file(joinpath(dir, "other_ap.nc");
                Nc, level_values = surface_first_values,
                ap = [0.0, 7000.0, 3000.0, 100.0], bp = reverse(B_top))
            @test occursin("ap coefficients differ", message(() -> build(other_ap)))

            # Only one of ap and bp is an error, whatever the level counts.
            only_ap = _write_hybrid_cs_native_ic_file(joinpath(dir, "only_ap.nc");
                Nc, level_values = surface_first_values, ap = reverse(A_top), bp = nothing)
            only_bp = _write_hybrid_cs_native_ic_file(joinpath(dir, "only_bp.nc");
                Nc, level_values = surface_first_values, ap = nothing, bp = reverse(B_top))
            for path in (only_ap, only_bp)
                @test occursin("must provide both ap and bp, or neither", message(() -> build(path)))
            end

            # ap in hPa, or unit-less with hPa-sized values, is rejected, not guessed.
            hpa = _write_hybrid_cs_native_ic_file(joinpath(dir, "hpa.nc");
                Nc, level_values = surface_first_values,
                ap = reverse(A_top) ./ 100, bp = reverse(B_top), ap_units = "hPa")
            @test occursin("must use Pa; got units = \"hPa\"", message(() -> build(hpa)))
            ambiguous = _write_hybrid_cs_native_ic_file(joinpath(dir, "ambiguous.nc");
                Nc, level_values = surface_first_values,
                ap = reverse(A_top) ./ 100, bp = reverse(B_top), ap_units = nothing)
            @test occursin("ambiguous with hPa", message(() -> build(ambiguous)))
            # Explicit Pa spellings and a unit-less pure-sigma (all-zero) ap pass.
            pascal = _write_hybrid_cs_native_ic_file(joinpath(dir, "pascal.nc");
                Nc, level_values = surface_first_values,
                ap = reverse(A_top), bp = reverse(B_top), ap_units = "Pascal")
            @test build(pascal) == vmr
            sigma_grid = AtmosGrid(mesh, HybridSigmaPressure(zeros(FT, Nz + 1), FT.(B_top)), CPU(); FT)
            sigma = _write_hybrid_cs_native_ic_file(joinpath(dir, "sigma.nc");
                Nc, level_values = surface_first_values,
                ap = zeros(Nz + 1), bp = reverse(B_top), ap_units = nothing)
            sigma_vmr = build_initial_mixing_ratio(air_mass, sigma_grid, cfg_for(sigma))
            @test sigma_vmr == vmr
        end
    end

    @testset "cs_native conservatively remaps mismatched hybrid levels" begin
        mktempdir() do dir
            Nc, Hp, Nz = 2, 1, 2
            path = joinpath(dir, "mismatched_cs_native.nc")
            _write_mismatched_cs_native_ic_file(path; Nc)
            mesh = CubedSphereMesh(; FT, Nc, Hp)
            vertical = HybridSigmaPressure(FT[0, 0, 0], FT[0, 0.5, 1])
            grid = AtmosGrid(mesh, vertical, CPU(); FT)
            air_mass = ntuple(_ -> fill(FT(1e10), Nc + 2Hp, Nc + 2Hp, Nz), 6)
            cfg = Dict{String, Any}(
                "kind" => "cs_native", "file" => path, "variable" => "TRACER",
                "humidity_variable" => "QV")

            ps_total = ntuple(_ -> fill(FT(90000), Nc, Nc), 6)
            total = build_initial_mixing_ratio(
                air_mass, grid, cfg; surface_pressure = ps_total,
                pressure_basis = TotalSurfacePressure())
            for p in 1:6
                @test total[p][:, :, 1] == fill(FT(0.6), Nc, Nc)
                @test total[p][:, :, 2] == fill(FT(5.4), Nc, Nc)
                # Dry-carrier pressure integral is conserved in every column.
                @test sum(Float64.(total[p][1, 1, :])) * 37500 == 225000
            end

            # The typed dry-pressure path recovers the same 90 kPa total ps
            # from target qv before placing source and target interfaces.
            target_qv = ntuple(_ -> fill(FT(0.1), Nc, Nc, Nz), 6)
            ps_dry = ntuple(_ -> fill(FT(81000), Nc, Nc), 6)
            dry = build_initial_mixing_ratio(
                air_mass, grid, cfg; surface_pressure = ps_dry,
                pressure_basis = DrySurfacePressure(), qv = target_qv)
            @test dry == total

            @test_throws ArgumentError build_initial_mixing_ratio(air_mass, grid, cfg)
        end
    end

    @testset "surface-anchored conservative hybrid remap ($T)" for T in (Float32, Float64)
        function remap!(dest, src_q, src_qv, ap, bp, A_tgt, B_tgt, ps,
                        target_dp_dry, src_p = zeros(Float64, length(src_q) + 1),
                        cumulative = similar(src_p), dry_cumulative = similar(src_p))
            ICIO._remap_hybrid_profile!(dest, src_q, src_qv, ap, bp, A_tgt,
                                        B_tgt, ps, src_p, cumulative, dry_cumulative)
        end

        # Different source and target partitions of the same pressure column,
        # including a strongly humid source bottom layer. The tracer integral
        # is conserved on dry pressure, not total pressure.
        src_q = T[9.0, 2.0, 5.0]                         # surface -> TOA
        src_qv = T[0.35, 0.08, 0.001]                    # strong humidity
        ap = zeros(T, 4)
        bp = T[1.0, 0.7, 0.2, 0.0]                      # surface -> TOA
        A_tgt = zeros(T, 5)
        B_tgt = T[0.0, 0.1, 0.4, 0.8, 1.0]              # TOA -> surface
        ps = T(100_000)
        src_interfaces = Float64.(ap) .+ Float64.(bp) .* Float64(ps)
        tgt_p = Float64.(A_tgt) .+ Float64.(B_tgt) .* Float64(ps)
        target_dp_dry = zeros(Float64, 4)
        for k in eachindex(target_dp_dry), s in eachindex(src_qv)
            overlap = max(0.0, min(tgt_p[k + 1], src_interfaces[s]) -
                               max(tgt_p[k], src_interfaces[s + 1]))
            target_dp_dry[k] += (1 - Float64(src_qv[s])) * overlap
        end
        dest = zeros(T, 4)
        scratch_p = zeros(Float64, 4)
        scratch_cumulative = similar(scratch_p)
        scratch_dry_cumulative = similar(scratch_p)
        remap!(dest, src_q, src_qv, ap, bp, A_tgt, B_tgt, ps, target_dp_dry,
               scratch_p, scratch_cumulative, scratch_dry_cumulative)
        source_integral = sum(Float64.(src_q) .* (1 .- Float64.(src_qv)) .*
                              (-diff(src_interfaces)))
        target_integral = sum(Float64.(dest) .* target_dp_dry)
        @test target_integral ≈ source_integral rtol = 64eps(T)
        @test !isapprox(sum(Float64.(dest) .* diff(tgt_p)),
                        sum(Float64.(src_q) .* (-diff(src_interfaces))); rtol = 1e-3)
        @test @allocated(remap!(dest, src_q, src_qv, ap, bp, A_tgt, B_tgt,
                                ps, target_dp_dry, scratch_p,
                                scratch_cumulative, scratch_dry_cumulative)) == 0

        uniform = similar(dest)
        remap!(uniform, fill(T(400e-6), length(src_q)), src_qv, ap, bp,
               A_tgt, B_tgt, ps, fill(1.0, length(dest)), scratch_p,
               scratch_cumulative, scratch_dry_cumulative)
        @test uniform ≈ fill(T(400e-6), length(uniform)) rtol = 8eps(T)

        # Without source-file humidity the target dry factor cancels exactly,
        # even for a target layer with 40% humidity.
        target_qv = T[0.0, 0.4, 0.12, 0.02]
        binary_dp_dry = (1 .- Float64.(target_qv)) .* diff(tgt_p)
        fallback = zeros(T, 4)
        remap!(fallback, src_q, nothing, ap, bp, A_tgt, B_tgt, ps,
               binary_dp_dry, scratch_p, scratch_cumulative, scratch_dry_cumulative)
        expected = zeros(Float64, 4)
        for k in eachindex(expected), s in eachindex(src_q)
            overlap = max(0.0, min(tgt_p[k + 1], src_interfaces[s]) -
                               max(tgt_p[k], src_interfaces[s + 1]))
            expected[k] += Float64(src_q[s]) * overlap / (tgt_p[k + 1] - tgt_p[k])
        end
        @test Float64.(fallback) ≈ expected rtol = 32eps(T)

        # Caller-owned scratch makes the production kernel allocation-free.
        remap!(fallback, src_q, nothing, ap, bp, A_tgt, B_tgt, ps,
               binary_dp_dry, scratch_p, scratch_cumulative)
        @test @allocated(remap!(fallback, src_q, nothing, ap, bp, A_tgt, B_tgt,
                                ps, binary_dp_dry, scratch_p,
                                scratch_cumulative, scratch_dry_cumulative)) == 0

        # A near-surface peak stays on the same sigma layers when every column
        # receives a uniform surface-pressure shift.
        peaked_q = T[50.0, 7.0, 3.0, 1.0]
        peaked_bp = T[1.0, 0.95, 0.8, 0.4, 0.0]
        shifted_1 = zeros(T, 5)
        shifted_2 = similar(shifted_1)
        target_bp = T[0.0, 0.1, 0.3, 0.6, 0.85, 1.0]
        remap!(shifted_1, peaked_q, nothing, zeros(T, 5), peaked_bp,
               zeros(T, 6), target_bp, T(98_325),
               diff(Float64.(target_bp) .* 98_325))
        remap!(shifted_2, peaked_q, nothing, zeros(T, 5), peaked_bp,
               zeros(T, 6), target_bp, T(98_725),
               diff(Float64.(target_bp) .* 98_725))
        @test shifted_1 ≈ shifted_2 rtol = 32eps(T)

        # A pure pressure-level source keeps its absolute levels. Its surface
        # end layer extends when the source bottom is above the target surface,
        # and is truncated when it is below the target surface.
        pressure_q = T[10.0, 2.0]
        pressure_ap = T[800.0, 600.0, 300.0]
        pressure_bp = zeros(T, 3)
        target_A = T[0.0, 300.0, 0.0]
        target_B = T[0.0, 0.0, 1.0]
        shallower = zeros(T, 2)
        deeper = similar(shallower)
        remap!(shallower, pressure_q, nothing, pressure_ap, pressure_bp,
               target_A, target_B, T(700),
               diff(Float64.(target_A) .+ Float64.(target_B) .* 700))
        remap!(deeper, pressure_q, nothing, pressure_ap, pressure_bp,
               target_A, target_B, T(1_000),
               diff(Float64.(target_A) .+ Float64.(target_B) .* 1_000))
        @test shallower ≈ T[2.0, 4.0] rtol = 16eps(T)
        @test deeper ≈ T[2.0, 4600 / 700] rtol = 16eps(T)

        @test_throws DimensionMismatch remap!(zeros(T, 3), src_q, nothing,
                                              ap[1:3], bp[1:3], A_tgt,
                                              B_tgt, ps, ones(3))
    end

    @testset "LL, RG, and CS binary surface-pressure basis traits ($T)" for T in (Float32, Float64)
        vertical = HybridSigmaPressure(zeros(T, 3), T[0.0, 0.5, 1.0])
        ps_total = T(100_000)
        dp_dry = T[30_000.0, 45_000.0] # qv = [0.40, 0.10]

        ll_mesh = LatLonMesh(; FT = T, Nx = 4, Ny = 2)
        ll_grid = AtmosGrid(ll_mesh, vertical, CPU(); FT = T)
        ll_air = Array{T}(undef, 4, 2, 2)
        for j in 1:2, i in 1:4, k in 1:2
            ll_air[i, j, k] = dp_dry[k] * cell_area(ll_mesh, i, j) /
                              AtmosTransport.Grids.gravity(ll_grid)
        end
        @test AtmosTransport.MetDrivers.infer_surface_pressure_basis(
            ll_grid, vec(ll_air), vec(fill(ps_total, 4, 2)),
            AtmosTransport.MetDrivers.LatLonBinaryGeometry(4, 2, [], [], [], [])) isa ICIO.TotalSurfacePressure

        rg_mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 4]; FT = T)
        rg_grid = AtmosGrid(rg_mesh, vertical, CPU(); FT = T)
        rg_air = Array{T}(undef, ncells(rg_mesh), 2)
        for c in 1:ncells(rg_mesh), k in 1:2
            rg_air[c, k] = dp_dry[k] * cell_area(rg_mesh, c) /
                           AtmosTransport.Grids.gravity(rg_grid)
        end
        @test AtmosTransport.MetDrivers.infer_surface_pressure_basis(
            rg_grid, vec(rg_air), fill(ps_total, ncells(rg_mesh)),
            AtmosTransport.MetDrivers.ReducedGaussianBinaryGeometry([], [])) isa
            ICIO.TotalSurfacePressure

        Nc, Hp = 2, 1
        cs_mesh = CubedSphereMesh(; FT = T, Nc, Hp)
        cs_grid = AtmosGrid(cs_mesh, vertical, CPU(); FT = T)
        cs_air = ntuple(_ -> zeros(T, Nc + 2Hp, Nc + 2Hp, 2), 6)
        for p in 1:6, j in 1:Nc, i in 1:Nc, k in 1:2
            cs_air[p][i + Hp, j + Hp, k] =
                dp_dry[k] * cell_area(cs_mesh, i, j) /
                AtmosTransport.Grids.gravity(cs_grid)
        end
        cs_ps_total = ntuple(_ -> fill(ps_total, Nc, Nc), 6)
        cs_ps_dry = ntuple(_ -> fill(sum(dp_dry), Nc, Nc), 6)
        cs_geometry = AtmosTransport.MetDrivers.CubedSphereBinaryGeometry(
            Nc, 6, :gnomonic, :equiangular, :equiangular, :normalized_vertex_mean, 0.0)
        cs_flat = reduce(vcat, vec(Array(@view panel[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :])) for panel in cs_air)
        @test AtmosTransport.MetDrivers.infer_surface_pressure_basis(
            cs_grid, cs_flat, reduce(vcat, vec.(cs_ps_total)), cs_geometry) isa ICIO.TotalSurfacePressure
        @test AtmosTransport.MetDrivers.infer_surface_pressure_basis(
            cs_grid, cs_flat, reduce(vcat, vec.(cs_ps_dry)), cs_geometry) isa ICIO.DrySurfacePressure
        recovered = ICIO._total_surface_pressure(
            ICIO.DrySurfacePressure(), sum(dp_dry), vertical.A, vertical.B, T[0.4, 0.1])
        @test recovered ≈ ps_total rtol = 16eps(T)
    end

    @testset "missing-humidity pressure fallback and initial qv selection" begin
        @test ICIO._total_surface_pressure(
            ICIO.DrySurfacePressure(), 98_000.0, zeros(3), [0.0, 0.5, 1.0], nothing) == 98_000.0

        runner = AtmosTransport.Models.DrivenRunner
        @test runner._initial_condition_qv((qv_start = :endpoint,
                                            vdiff = (qv = :vdiff,))) === :endpoint
        @test runner._initial_condition_qv((qv_start = nothing,
                                            vdiff = (qv = :vdiff,))) === :vdiff
        @test runner._initial_condition_qv((qv_start = nothing,
                                            vdiff = nothing)) === nothing

    end

    @testset "validated humidity config reaches LL/RG/CS builders ($T)" for T in (Float32, Float64)
        mktempdir() do dir
            path = joinpath(dir, "q_source.nc")
            _write_synthetic_ic_file(path; with_humidity = true,
                                     humidity_name = "q",
                                     humidity_standard_name = false)
            init_cfg = Dict{String, Any}(
                "kind" => "file",
                "file" => path,
                "variable" => "CO2",
                "humidity_variable" => "q",
            )
            cfg = Dict{String, Any}(
                "input" => Dict("binary_paths" => [path]),
                "numerics" => Dict("float_type" => string(T)),
                "tracers" => Dict(
                    "nested" => Dict("init" => init_cfg),
                    "flat" => copy(init_cfg),
                ),
            )
            @test validate_config(cfg) == (true, String[])
            specs = AtmosTransport.Models.DrivenRunner._parse_tracer_specs(cfg)
            @test length(specs) == 2
            @test all(spec -> spec.init_cfg["humidity_variable"] == "q", specs)

            # Humidity is opt-in: a conventional `q` name does not silently
            # change the conservative carrier.
            autodetect_cfg = copy(init_cfg)
            delete!(autodetect_cfg, "humidity_variable")
            source = ICIO._load_file_initial_condition_source(autodetect_cfg, T, 2)
            @test source.qv === nothing

            vertical = HybridSigmaPressure(T[0, 0, 0], T[0, 0.5, 1])
            ps_total = T(100_000)
            src_bp = [1.0, 2 / 3, 1 / 3, 0.0] # file order: surface -> TOA
            src_qv = [0.35, 0.08, 0.001]
            tgt_p = [0.0, 50_000.0, 100_000.0]
            src_p = src_bp .* Float64(ps_total)
            dp_dry = zeros(Float64, 2)
            for k in 1:2, s in 1:3
                overlap = max(0.0, min(tgt_p[k + 1], src_p[s]) -
                                   max(tgt_p[k], src_p[s + 1]))
                dp_dry[k] += (1 - src_qv[s]) * overlap
            end
            ps_dry = T(sum(dp_dry))
            target_qv = T.(1 .- dp_dry ./ diff(tgt_p))
            cases = (
                (ICIO.TotalSurfacePressure(), ps_total, nothing),
                (ICIO.DrySurfacePressure(), ps_dry, target_qv),
                (ICIO.DrySurfacePressure(), ps_dry, nothing),
            )

            ll_mesh = LatLonMesh(; FT = T, Nx = 4, Ny = 2)
            ll_grid = AtmosGrid(ll_mesh, vertical, CPU(); FT = T)
            ll_air = Array{T}(undef, 4, 2, 2)
            for j in 1:2, i in 1:4, k in 1:2
                ll_air[i, j, k] = T(dp_dry[k] * cell_area(ll_mesh, i, j) /
                                      AtmosTransport.Grids.gravity(ll_grid))
            end
            ll_qv = repeat(reshape(target_qv, 1, 1, 2), 4, 2, 1)
            ll_results = map(cases) do (basis, ps, qv_profile)
                qv = qv_profile === nothing ? nothing : ll_qv
                build_initial_mixing_ratio(ll_air, ll_grid, init_cfg;
                    surface_pressure = fill(ps, 4, 2), pressure_basis = basis,
                    qv)
            end
            @test ll_results[1] ≈ ll_results[2] rtol = 128eps(T)
            @test all(isfinite, ll_results[3])

            rg_mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 4]; FT = T)
            rg_grid = AtmosGrid(rg_mesh, vertical, CPU(); FT = T)
            rg_air = Array{T}(undef, ncells(rg_mesh), 2)
            for c in 1:ncells(rg_mesh), k in 1:2
                rg_air[c, k] = T(dp_dry[k] * cell_area(rg_mesh, c) /
                                   AtmosTransport.Grids.gravity(rg_grid))
            end
            rg_qv = repeat(reshape(target_qv, 1, 2), ncells(rg_mesh), 1)
            rg_results = map(cases) do (basis, ps, qv_profile)
                qv = qv_profile === nothing ? nothing : rg_qv
                build_initial_mixing_ratio(rg_air, rg_grid, init_cfg;
                    surface_pressure = fill(ps, ncells(rg_mesh)),
                    pressure_basis = basis, qv)
            end
            @test rg_results[1] ≈ rg_results[2] rtol = 128eps(T)
            @test all(isfinite, rg_results[3])

            Nc, Hp = 2, 1
            cs_mesh = CubedSphereMesh(; FT = T, Nc, Hp)
            cs_grid = AtmosGrid(cs_mesh, vertical, CPU(); FT = T)
            cs_air = ntuple(_ -> zeros(T, Nc + 2Hp, Nc + 2Hp, 2), 6)
            for p in 1:6, j in 1:Nc, i in 1:Nc, k in 1:2
                cs_air[p][i + Hp, j + Hp, k] =
                    T(dp_dry[k] * cell_area(cs_mesh, i, j) /
                      AtmosTransport.Grids.gravity(cs_grid))
            end
            cs_qv = ntuple(_ -> repeat(reshape(target_qv, 1, 1, 2), Nc, Nc, 1), 6)
            cs_results = map(cases) do (basis, ps, qv_profile)
                qv = qv_profile === nothing ? nothing : cs_qv
                build_initial_mixing_ratio(cs_air, cs_grid, init_cfg;
                    surface_pressure = ntuple(_ -> fill(ps, Nc, Nc), 6),
                    pressure_basis = basis, qv)
            end
            for p in 1:6
                @test cs_results[1][p] ≈ cs_results[2][p] rtol = 128eps(T)
                @test all(isfinite, cs_results[3][p])
            end
        end
    end


end

@testset "catrine_co2 default follows the data root" begin
    ICIO = AtmosTransport.Models.InitialConditionIO
    withenv("ATMOSTRANSPORT_DATA_ROOT" => "/tmp/bf2_data") do
        file, variable, time_index = ICIO._resolve_file_init(Dict(), :catrine_co2)
        @test file == "/tmp/bf2_data/catrine/InitialConditions/startCO2_202112010000.nc"
        @test (variable, time_index) == ("CO2", 1)
        @test first(ICIO._resolve_file_init(Dict("file" => "/tmp/explicit.nc"), :catrine_co2)) ==
              "/tmp/explicit.nc"
    end
end
