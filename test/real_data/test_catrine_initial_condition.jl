# Conservative C90 near-surface IC invariant, requiring the CATRINE dataset.
using Test, AtmosTransport, TOML
import NCDatasets: NCDataset, nomissing
const ICIO = AtmosTransport.Models.InitialConditionIO
const FT = Float64

@testset "CATRINE C90 near-surface band is preserved" begin
    catrine_ic_path =
        expanduser("~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc")
    if !isfile(catrine_ic_path)
        @info "Skipping CATRINE C90 conservative-remap regression; file not present at $catrine_ic_path"
        @test_skip false
    else
        # Independent bottom-fraction mean for one column. Source files
        # are surface first; model columns are top first.
        function surface_band(q, dp, fraction; surface_first)
            total = sum(dp)
            remaining = fraction * total
            weighted = 0.0
            indices = surface_first ? eachindex(q) : Iterators.reverse(eachindex(q))
            for k in indices
                take = min(remaining, dp[k])
                weighted += Float64(q[k]) * take
                remaining -= take
                remaining <= 16eps(Float64) * total && break
            end
            return weighted / (fraction * total)
        end

        coeff_path = normpath(joinpath(@__DIR__, "..", "..", "config",
                                       "geos_L72_coefficients.toml"))
        coeff = TOML.parsefile(coeff_path)["coefficients"]
        A = Float64.(coeff["a"])
        B = Float64.(coeff["b"])
        Nz = length(A) - 1

        mesh = CubedSphereMesh(; FT = Float64, Nc = 90, Hp = 1)
        grid = AtmosGrid(mesh, HybridSigmaPressure(A, B), CPU(); FT = Float64)
        air = ntuple(_ -> ones(Float64, 92, 92, Nz), 6)
        cfg = Dict{String, Any}("kind" => "file", "file" => catrine_ic_path,
                                "variable" => "CO2")

        # Put the source topography on C90, then shift every column by the
        # same 400 Pa. The correct surface-anchored remap is invariant to
        # this pin-sized shift; the replaced absolute-pressure mapping was
        # not. `Psurf` is read only to construct this independent test
        # perturbation and is not part of FileInitialConditionSource.
        source = ICIO._load_file_initial_condition_source(cfg, Float64, Nz)
        source_ps = NCDataset(catrine_ic_path) do ds
            lon = Float64.(ds["longitude"][:])
            lat = Float64.(ds["latitude"][:])
            raw_ps = Float64.(nomissing(ds["Psurf"][:, :], 100_000.0))
            orientation = AtmosTransport.GriddedIO.latlon_orientation(lon, lat)
            AtmosTransport.GriddedIO.orient(raw_ps, orientation)
        end
        source_mesh = AtmosTransport.GriddedIO.source_latlon_mesh(
            source.lon, source.lat; radius = mesh.radius)
        ps_regridder = AtmosTransport.Regridding.build_regridder(source_mesh, mesh)
        ps_flat = Vector{Float64}(undef, 6 * 90 * 90)
        AtmosTransport.Regridding.apply_regridder!(ps_flat, ps_regridder,
                                                   reshape(source_ps, :))
        ps_base = ntuple(_ -> Matrix{Float64}(undef, 90, 90), 6)
        AtmosTransport.Preprocessing.unpack_flat_to_panels_2d!(ps_base, ps_flat, 90)
        ps = ntuple(p -> ps_base[p] .+ 400.0, 6)
        mapped = build_initial_mixing_ratio(air, grid, cfg; surface_pressure = ps)

        target_weighted = 0.0
        target_area = 0.0
        for p in 1:6, j in 1:90, i in 1:90
            area = Float64(cell_area(mesh, i, j))
            target_dp = diff(A .+ B .* ps[p][i, j])
            target_weighted += area * surface_band(@view(mapped[p][i, j, :]),
                                                   target_dp, 0.1;
                                                   surface_first = false)
            target_area += area
        end
        target_mean = target_weighted / target_area

        source_mean = NCDataset(catrine_ic_path) do ds
            q = Float64.(nomissing(ds["CO2"][:, :, :], 0.0))
            source_ps = Float64.(nomissing(ds["Psurf"][:, :], 100_000.0))
            ap = Float64.(ds["ap"][:])
            bp = Float64.(ds["bp"][:])
            lat = Float64.(ds["latitude"][:])
            weighted = 0.0
            area_sum = 0.0
            for j in eachindex(lat), i in axes(q, 1)
                # Exact spherical strip weight for the regular 1-degree
                # source grid; the longitude-width factor cancels.
                south = max(-90.0, lat[j] - 0.5)
                north = min(90.0, lat[j] + 0.5)
                area = sind(north) - sind(south)
                interfaces = ap .+ bp .* source_ps[i, j]
                dp = -diff(interfaces)
                weighted += area * surface_band(@view(q[i, j, :]), dp, 0.1;
                                                surface_first = true)
                area_sum += area
            end
            weighted / area_sum
        end

        difference_ppm = (target_mean - source_mean) * 1e6
        @info "CATRINE C90 conservative IC regression" target_ppm=target_mean * 1e6 source_ppm=source_mean * 1e6 difference_ppm
        @test abs(difference_ppm) <= 1e-3
    end
end
