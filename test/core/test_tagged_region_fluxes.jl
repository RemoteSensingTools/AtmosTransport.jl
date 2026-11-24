#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Region-tagged surface fluxes (`[tagged_fluxes.<group>]`): one flux and an
# integer region mask on the flux's own grid become one tracer per selected
# mask value.
#
#   - the expansion into `[tracers]`: default values and names, `exclude`,
#     every validation error;
#   - lat-lon, reduced-Gaussian and cubed-sphere targets, static and
#     time-varying (`lmdz_co2`, `cs_native`): each tagged tracer is the flux
#     over its cells, the tagged sum is the flux over the selected values, and
#     with every value selected (no fill) it equals the untagged source cell by
#     cell; excluded values and fill cells emit nothing;
#   - the flux is read once per group;
#   - a mask on another grid is refused.
# ---------------------------------------------------------------------------

using Test
using Dates
using Logging
using Printf: @sprintf
import NCDatasets: NCDataset, defDim, defVar

using AtmosTransport
using AtmosTransport: LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, HybridSigmaPressure,
                      AtmosGrid, CPU
using AtmosTransport.Models.InitialConditionIO: build_surface_flux_source, build_surface_flux_sources
using AtmosTransport.Parameters: DRY_AIR_MOLAR_MASS, SPECIES_MOLAR_MASS
using AtmosTransport.GriddedIO: read_region_mask, region_mask_values, LatLonRegionMask,
                                CubedSphereRegionMask

const ICIO = AtmosTransport.Models.InitialConditionIO
const RUNNER = AtmosTransport.Models.DrivenRunner

quietly(f) = with_logger(f, SimpleLogger(devnull, Logging.Warn))
storage_scale_co2() = DRY_AIR_MOLAR_MASS / SPECIES_MOLAR_MASS.co2

# 10° flux grid in the runtime orientation.
const LON = collect(5.0:10:355)
const LAT = collect(-85.0:10:85)
const NX, NY = length(LON), length(LAT)

vert() = HybridSigmaPressure(Float64[0, 100, 300], Float64[0, 0, 1])
ll_grid() = AtmosGrid(LatLonMesh(; FT = Float64, Nx = 24, Ny = 12), vert(), CPU(); FT = Float64)
rg_grid() = AtmosGrid(ReducedGaussianMesh(Float64[-60, -20, 20, 60], [8, 12, 12, 8]; FT = Float64),
                      vert(), CPU(); FT = Float64)
const NC = 6
cs_grid(FT = Float64) = AtmosGrid(CubedSphereMesh(; Nc = NC, Hp = 3, FT), vert(), CPU(); FT)

# A mixed-sign flux pattern (kg CO2 m⁻² s⁻¹) on the 10° grid.
flux_pattern(i, j, t = 0) = 1e-8 * (sind(LAT[j] + 7) + 0.3 * cosd(LON[i]) + 0.05 * t)

# Combined region × PFT codes (100 × region + pft, plus small and negative codes).
const CODES = (1203, 7, -4, 0, 2101)
code_at(i, j) = CODES[mod1(i + 2j, length(CODES))]

# Static flux file (no time axis).
function static_flux_file(path)
    NCDataset(path, "c") do ds
        defDim(ds, "lon", NX); defDim(ds, "lat", NY)
        defVar(ds, "lon", Float64, ("lon",))[:] = LON
        defVar(ds, "lat", Float64, ("lat",))[:] = LAT
        defVar(ds, "F", Float64, ("lon", "lat"), attrib = Dict("units" => "kg CO2 m-2 s-1"))[:, :] =
            [flux_pattern(i, j) for i in 1:NX, j in 1:NY]
    end
    return path
end

# LMDZ-like series: 3-hourly stamps over two days of January 2022, kgC.
function lmdz_flux_file(path; keep = (i, j) -> true)
    n = 16
    NCDataset(path, "c") do ds
        defDim(ds, "lon", NX); defDim(ds, "lat", NY); defDim(ds, "time", n)
        defVar(ds, "lon", Float64, ("lon",))[:] = LON
        defVar(ds, "lat", Float64, ("lat",))[:] = LAT
        defVar(ds, "time", Float64, ("time",),
               attrib = Dict("units" => "hours since 2022-01-01 00:00:00"))[:] = 3.0 .* (0:n-1)
        defVar(ds, "flux_apos", Float64, ("lon", "lat", "time"),
               attrib = Dict("units" => "kgC m-2 s-1"))[:, :, :] =
            [keep(i, j) ? flux_pattern(i, j, t) : 0.0 for i in 1:NX, j in 1:NY, t in 1:n]
    end
    return path
end

# Lat-lon Int32 mask stored in another orientation than the flux (longitudes
# -175…175, latitude descending), with `fill_cells` set to `_FillValue`.
function ll_mask_file(path; fill_cells = (), lon = LON, lat = LAT, code = code_at)
    nx, ny = length(lon), length(lat)
    codes = Int32[code(i, j) for i in 1:nx, j in 1:ny]
    for (i, j) in fill_cells
        codes[i, j] = Int32(-9999)
    end
    # File order: longitudes from -180, latitude north to south.
    shift = findfirst(>=(180), lon)
    order = shift === nothing ? (1:nx) : vcat(shift:nx, 1:(shift - 1))
    file_lon = [x >= 180 ? x - 360 : x for x in lon[order]]
    NCDataset(path, "c") do ds
        defDim(ds, "lon", nx); defDim(ds, "lat", ny)
        defVar(ds, "lon", Float64, ("lon",))[:] = file_lon
        defVar(ds, "lat", Float64, ("lat",))[:] = reverse(lat)
        defVar(ds, "region", Int32, ("lon", "lat"), attrib = Dict("_FillValue" => Int32(-9999)))[:, :] =
            codes[order, end:-1:1]
    end
    return path
end

# Cubed-sphere Int32 mask `(Xdim, Ydim, nf)`.
# Cubed-sphere Int32 mask `(Xdim, Ydim, nf)` with the cube's `lons`/`lats`
# (`centres = false` omits them; `lon_shift` moves them).
function cs_mask_file(path; Nc = NC, centres = true, lon_shift = 0.0)
    mesh = CubedSphereMesh(; Nc, FT = Float64)
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", Nc); defDim(ds, "Ydim", Nc); defDim(ds, "nf", 6)
        defVar(ds, "region", Int32, ("Xdim", "Ydim", "nf"))[:, :, :] =
            Int32[CODES[mod1(i + 2j + 3p, length(CODES))] for i in 1:Nc, j in 1:Nc, p in 1:6]
        if centres
            lonlat = [AtmosTransport.Grids.panel_cell_center_lonlat(mesh, p) for p in 1:6]
            defVar(ds, "lons", Float64, ("Xdim", "Ydim", "nf"))[:, :, :] =
                cat((first(c) .+ lon_shift for c in lonlat)...; dims = 3)
            defVar(ds, "lats", Float64, ("Xdim", "Ydim", "nf"))[:, :, :] = cat((last(c) for c in lonlat)...; dims = 3)
        end
    end
    return path
end

# cs_native series: daily stamps over three days of January 2022.
function cs_native_file(path; Nc = NC)
    NCDataset(path, "c") do ds
        for (name, n) in (("Xdim", Nc), ("Ydim", Nc), ("nf", 6), ("time", 3))
            defDim(ds, name, n)
        end
        defVar(ds, "time", Float64, ("time",),
               attrib = Dict("units" => "hours since 2022-01-01 00:00:00"))[:] = [0.0, 24.0, 48.0]
        defVar(ds, "F", Float32, ("Xdim", "Ydim", "nf", "time"),
               attrib = Dict("units" => "kg CO2 m-2 s-1"))[:, :, :, :] =
            Float32[1f-9 * (i + 10j + 100p + 1000t) * (iseven(i + j) ? 1 : -1)
                    for i in 1:Nc, j in 1:Nc, p in 1:6, t in 1:3]
    end
    return path
end

tagged_cfg(flux, mask; extra...) = Dict{String, Any}("tagged_fluxes" => Dict{String, Any}(
    "nee" => Dict{String, Any}("flux" => flux, "mask_file" => mask, "mask_variable" => "region",
                               (String(k) => v for (k, v) in extra)...)))

tracer_names(cfg) = Set(keys(RUNNER._expand_tagged_fluxes(cfg)["tracers"]))
message(f) = try f(); "" catch err; err isa ArgumentError || rethrow(); err.msg end

cells(src::AtmosTransport.SurfaceFluxSource) =
    src.cell_mass_rate isa Tuple ? reduce(vcat, vec.(src.cell_mass_rate)) : vec(src.cell_mass_rate)
cells(src::AtmosTransport.TimeVaryingSurfaceFluxSource) =
    reduce(vcat, vec.(src.cell_mass_rate_series))
# Sum over tracers of the per-cell rates.
cell_sum(sources) = reduce(+, cells.(sources))
by_name(sources) = Dict(s.tracer_name => s for s in sources)

# Flux over the cells of `values` on the source grid, kg CO2/s, with the source
# mesh's areas (the static loader's areas for files without `cell_area`).
function selected_flux_total(mask, values; radius)
    area = ICIO._lonlat_cell_areas_m2(LON, LAT; radius)
    F = [flux_pattern(i, j) for i in 1:NX, j in 1:NY]
    return sum(F[c] * area[c] for c in eachindex(F) if mask.valid[c] && mask.codes[c] in values)
end

@testset "region masks: codes, orientation and fill" begin
    mktempdir() do dir
        mask = read_region_mask(ll_mask_file(joinpath(dir, "m.nc"); fill_cells = ((1, 1), (36, 18))),
                                "region")
        @test mask isa LatLonRegionMask
        @test mask.lon == LON && mask.lat == LAT                 # runtime orientation
        @test mask.codes[2, 3] == code_at(2, 3)
        @test !mask.valid[1, 1] && !mask.valid[36, 18]
        @test region_mask_values(mask) == sort!(collect(Int64, CODES))
        cs = read_region_mask(cs_mask_file(joinpath(dir, "cs.nc")), "region")
        @test cs isa CubedSphereRegionMask && size(cs.codes) == (NC, NC, 6)
        # Non-integer values are refused.
        bad = joinpath(dir, "bad.nc")
        NCDataset(bad, "c") do ds
            defDim(ds, "lon", NX); defDim(ds, "lat", NY)
            defVar(ds, "lon", Float64, ("lon",))[:] = LON
            defVar(ds, "lat", Float64, ("lat",))[:] = LAT
            defVar(ds, "region", Float64, ("lon", "lat"))[:, :] = fill(1.5, NX, NY)
        end
        @test occursin("non-integer", message(() -> read_region_mask(bad, "region")))
        # A float mask: NaN and missing_value cells hold no code.
        fmask = joinpath(dir, "float.nc")
        NCDataset(fmask, "c") do ds
            defDim(ds, "lon", NX); defDim(ds, "lat", NY)
            defVar(ds, "lon", Float64, ("lon",))[:] = LON
            defVar(ds, "lat", Float64, ("lat",))[:] = LAT
            values = [Float64(code_at(i, j)) for i in 1:NX, j in 1:NY]
            values[1, 1] = NaN
            values[2, 1] = -999.0
            defVar(ds, "region", Float64, ("lon", "lat"), attrib = Dict("missing_value" => -999.0))[:, :] = values
        end
        f = read_region_mask(fmask, "region")
        @test !f.valid[1, 1] && !f.valid[2, 1] && f.valid[3, 1] && f.codes[3, 1] == code_at(3, 1)
        # Int64 codes beyond 2^53 are kept exactly (no Float64 round trip).
        big = joinpath(dir, "big.nc")
        NCDataset(big, "c") do ds
            defDim(ds, "lon", NX); defDim(ds, "lat", NY)
            defVar(ds, "lon", Float64, ("lon",))[:] = LON
            defVar(ds, "lat", Float64, ("lat",))[:] = LAT
            defVar(ds, "region", Int64, ("lon", "lat"))[:, :] =
                [iseven(i) ? 2^53 + 1 : 2^53 for i in 1:NX, j in 1:NY]
        end
        @test region_mask_values(read_region_mask(big, "region")) == [2^53, 2^53 + 1]
        # A square mask stored (lat, lon) is matched to its axes by the
        # coordinates' dimensions and transposed.
        sq = joinpath(dir, "square.nc")
        n = 18
        sq_lon = collect(10.0:20:350); sq_lat = collect(-85.0:10:85)
        NCDataset(sq, "c") do ds
            defDim(ds, "lon", n); defDim(ds, "lat", n)
            defVar(ds, "lon", Float64, ("lon",))[:] = sq_lon
            defVar(ds, "lat", Float64, ("lat",))[:] = sq_lat
            defVar(ds, "region", Int32, ("lat", "lon"))[:, :] = Int32[100i + j for j in 1:n, i in 1:n]
        end
        m = read_region_mask(sq, "region")
        @test m.codes[3, 5] == 305 && m.codes[5, 3] == 503
        # Other dimension names are refused.
        odd = joinpath(dir, "odd.nc")
        NCDataset(odd, "c") do ds
            defDim(ds, "lon", NX); defDim(ds, "lat", NY); defDim(ds, "x", NX); defDim(ds, "y", NY)
            defVar(ds, "lon", Float64, ("lon",))[:] = LON
            defVar(ds, "lat", Float64, ("lat",))[:] = LAT
            defVar(ds, "region", Int32, ("x", "y"))[:, :] = ones(Int32, NX, NY)
        end
        @test occursin("expected the coordinate dimensions", message(() -> read_region_mask(odd, "region")))
    end
end

@testset "expansion: defaults, names, exclude and validation errors" begin
    mktempdir() do dir
        mask = ll_mask_file(joinpath(dir, "m.nc"))
        flux = Dict{String, Any}("kind" => "file", "file" => static_flux_file(joinpath(dir, "f.nc")),
                                 "variable" => "F", "species" => "co2")
        # Default: every value of the mask; default names <group>_<value>.
        @test tracer_names(tagged_cfg(flux, mask)) ==
              Set(["nee_1203", "nee_7", "nee_m4", "nee_0", "nee_2101"])
        @test tracer_names(tagged_cfg(flux, mask; exclude = [0, -4])) ==
              Set(["nee_1203", "nee_7", "nee_2101"])
        @test tracer_names(tagged_cfg(flux, mask; values = [7, 1203], names = ["a", "b"])) ==
              Set(["a", "b"])
        # Explicit tracers stay; the tagged tracer carries the group's init and flux.
        cfg = tagged_cfg(flux, mask; values = [7], init = Dict("kind" => "uniform", "background" => 4e-4))
        cfg["tracers"] = Dict{String, Any}("co2" => Dict{String, Any}("init" => Dict("kind" => "uniform")))
        expanded = RUNNER._expand_tagged_fluxes(cfg)
        @test !haskey(expanded, "tagged_fluxes") && haskey(expanded["tracers"], "co2")
        @test expanded["tracers"]["nee_7"]["init"]["background"] == 4e-4
        region = ICIO.tagged_flux_region(expanded["tracers"]["nee_7"]["surface_flux"])
        @test region == ICIO.TaggedFluxRegion("nee", mask, "region", 7)
        @test RUNNER._expand_tagged_fluxes(expanded) === expanded          # idempotent
        @test !haskey(cfg["tracers"], "nee_7")                              # input untouched
        specs = RUNNER._parse_tracer_specs(cfg)                             # readers see them
        @test [s.name for s in specs] == [:co2, :nee_7]

        err(cfg) = message(() -> RUNNER._expand_tagged_fluxes(cfg))
        @test occursin("names has 1 entries but values has 2",
                       err(tagged_cfg(flux, mask; values = [7, 0], names = ["a"])))
        @test occursin("names needs `values`", err(tagged_cfg(flux, mask; names = ["a"])))
        @test occursin("already defined by [tagged_fluxes.nee] value 7",
                       err(tagged_cfg(flux, mask; values = [7, 0], names = ["a", "a"])))
        clash = tagged_cfg(flux, mask; values = [7])
        clash["tracers"] = Dict{String, Any}("nee_7" => Dict{String, Any}())
        @test occursin("already defined by [tracers.nee_7]", err(clash))
        @test occursin("99 is not in mask 'region'", err(tagged_cfg(flux, mask; values = [99, 7])))
        @test occursin("it holds -4, 0, 7, 1203, 2101", err(tagged_cfg(flux, mask; values = [99])))
        @test occursin("7 is both in `values` and in `exclude`",
                       err(tagged_cfg(flux, mask; values = [7], exclude = [7])))
        @test occursin("no mask values left",
                       err(tagged_cfg(flux, mask; exclude = collect(CODES))))
        @test occursin("lists a value twice", err(tagged_cfg(flux, mask; values = [7, 7])))
        @test occursin("must be an array of integers", err(tagged_cfg(flux, mask; values = [true])))
        @test occursin("unknown option(s) value (did you mean `values`?)",
                       err(tagged_cfg(flux, mask; value = [7])))
        @test occursin("requires `mask_variable`",
                       err(Dict("tagged_fluxes" => Dict("nee" => Dict("flux" => flux, "mask_file" => mask)))))
        @test occursin("kind other than \"none\"", err(tagged_cfg(Dict("kind" => "none"), mask)))
        # Tagged names imply no species: the flux must name it.
        @test occursin("set flux.species (tagged tracer names imply no species",
                       err(tagged_cfg(delete!(copy(flux), "species"), mask)))
        @test occursin("resolve to different species", err(tagged_cfg(delete!(copy(flux), "species"), mask;
                                                                       values = [7, 0], names = ["co2_natural", "sf6"])))
        @test occursin("need conservative regridding",
                       err(tagged_cfg(merge(flux, Dict("regridding" => "bilinear")), mask)))
        # A valid group has no key findings.
        @test isempty(RUNNER._config_key_findings(RUNNER._expand_tagged_fluxes(
            tagged_cfg(flux, mask; values = [7], names = ["a"]))).errors)
        # The singular builder refuses a tagged tracer's table (it holds the whole flux).
        tagged_sf = RUNNER._expand_tagged_fluxes(tagged_cfg(flux, mask; values = [7]))["tracers"]["nee_7"]["surface_flux"]
        @test occursin("is a [tagged_fluxes] tracer",
                       message(() -> build_surface_flux_source(ll_grid(), :nee_7, tagged_sf, Float64)))
        @test occursin("region mask file not found", err(tagged_cfg(flux, joinpath(dir, "none.nc"))))
        @test occursin("an internal key", err(tagged_cfg(merge(flux, Dict(ICIO.TAGGED_REGION_KEY => 1)), mask)))
        # An empty [tagged_fluxes] adds nothing (no empty [tracers]).
        @test !haskey(RUNNER._expand_tagged_fluxes(Dict{String, Any}("tagged_fluxes" => Dict{String, Any}())),
                      "tracers")
        # validate_config reports an expansion error like any other.
        ok, errors = RUNNER.validate_config(tagged_cfg(flux, mask; values = [99]))
        @test !ok && any(e -> occursin("99 is not in mask", e), errors)
        # Unknown flux keys of a group are reported once, under the group's label.
        findings = RUNNER._config_key_findings(RUNNER._expand_tagged_fluxes(
            tagged_cfg(merge(flux, Dict("scal" => 2.0)), mask)))
        flux_errors = filter(e -> occursin("scal", e), findings.errors)
        @test length(flux_errors) == 1 && occursin("[tagged_fluxes.nee.flux]", only(flux_errors))
    end
end

@testset "static sources on lat-lon, reduced-Gaussian and cubed-sphere targets" begin
    mktempdir() do dir
        file = static_flux_file(joinpath(dir, "f.nc"))
        flux = Dict{String, Any}("kind" => "file", "file" => file, "variable" => "F", "species" => "co2")
        full_mask = ll_mask_file(joinpath(dir, "full.nc"))
        holed_mask = ll_mask_file(joinpath(dir, "holed.nc"); fill_cells = ((3, 4), (20, 9)))
        for grid in (ll_grid(), rg_grid(), cs_grid())
            radius = grid.horizontal.radius
            untagged = quietly(() -> build_surface_flux_source(grid, :ref, flux, Float64))
            # Every value of a fill-free mask: the tagged sum is the untagged
            # source cell by cell.
            all_sources = quietly(() -> build_surface_flux_sources(
                grid, RUNNER._parse_tracer_specs(tagged_cfg(flux, full_mask)), Float64))
            @test length(all_sources) == length(CODES)
            ref = cells(untagged)
            @test maximum(abs.(cell_sum(all_sources) .- ref)) <= 1e-12 * maximum(abs.(ref))
            # Each tracer is the flux over its cells; excluded values and fill
            # cells emit nothing.
            selected = quietly(() -> build_surface_flux_sources(
                grid, RUNNER._parse_tracer_specs(tagged_cfg(flux, holed_mask; exclude = [0, 7])),
                Float64))
            mask = read_region_mask(holed_mask, "region")
            @test Set(s.tracer_name for s in selected) == Set([:nee_1203, :nee_m4, :nee_2101])
            for s in selected
                v = parse(Int, replace(String(s.tracer_name)[5:end], "m" => "-"))
                @test sum(cells(s)) / storage_scale_co2() ≈ selected_flux_total(mask, (v,); radius) rtol = 1e-12
            end
            expected = selected_flux_total(mask, (1203, -4, 2101); radius)
            @test sum(cell_sum(selected)) / storage_scale_co2() ≈ expected rtol = 1e-12
        end
    end
end

@testset "the flux is read once per group" begin
    mktempdir() do dir
        flux = Dict{String, Any}("kind" => "file", "file" => static_flux_file(joinpath(dir, "f.nc")),
                                 "variable" => "F", "species" => "co2")
        specs = RUNNER._parse_tracer_specs(tagged_cfg(flux, ll_mask_file(joinpath(dir, "m.nc"))))
        logger = Test.TestLogger(min_level = Logging.Debug)
        with_logger(() -> build_surface_flux_sources(ll_grid(), specs, Float64), logger)
        loads = count(r -> r.message == "Loaded static surface flux", logger.logs)
        @test loads == 1
        @test count(r -> startswith(string(r.message), "Tagged flux nee / "), logger.logs) == length(CODES)

        lmdz = Dict{String, Any}("kind" => "lmdz_co2", "file" => lmdz_flux_file(joinpath(dir, "l.nc")),
                                 "time_varying" => true, "species" => "co2")
        specs = RUNNER._parse_tracer_specs(tagged_cfg(lmdz, ll_mask_file(joinpath(dir, "m.nc"))))
        logger = Test.TestLogger(min_level = Logging.Info)
        with_logger(logger) do
            build_surface_flux_sources(cs_grid(), specs, Float64;
                                       reference_time = DateTime(2022, 1, 1), run_span = (0.0, 86400.0))
        end
        @test count(r -> r.message == "Loaded time-varying surface flux", logger.logs) == 1
    end
end

@testset "time-varying sources: lmdz_co2 and cs_native on the cube" begin
    mktempdir() do dir
        reference_time = DateTime(2022, 1, 1)
        run_span = (0.0, 86400.0)
        for FT in (Float64, Float32)
            grid = cs_grid(FT)
            lmdz = Dict{String, Any}("kind" => "lmdz_co2", "file" => lmdz_flux_file(joinpath(dir, "l$FT.nc")),
                                     "time_varying" => true, "species" => "co2")
            untagged = quietly(() -> build_surface_flux_source(grid, :ref, lmdz, FT; reference_time, run_span))
            mask = ll_mask_file(joinpath(dir, "m$FT.nc"))
            tagged = quietly(() -> build_surface_flux_sources(
                grid, RUNNER._parse_tracer_specs(tagged_cfg(lmdz, mask)), FT; reference_time, run_span))
            @test all(s -> s.times == untagged.times && s.scheme == untagged.scheme, tagged)
            ref = Float64.(cells(untagged))
            tol = FT == Float64 ? 1e-12 : 1e-5
            @test maximum(abs.(Float64.(cell_sum(tagged)) .- ref)) <= tol * maximum(abs.(ref))
            # One value: the flux over its cells, slice by slice.
            one = only(quietly(() -> build_surface_flux_sources(
                grid, RUNNER._parse_tracer_specs(tagged_cfg(lmdz, mask; values = [1203])), FT;
                reference_time, run_span)))
            @test one.tracer_name == :nee_1203
            @test maximum(abs.(Float64.(cells(one)) .- Float64.(cells(by_name(tagged)[:nee_1203])))) == 0
            # Fill cells and excluded values emit nothing: the tagged sum is
            # the untagged source of a file with those cells zeroed.
            holes = ((3, 4), (20, 9))
            holed = ll_mask_file(joinpath(dir, "h$FT.nc"); fill_cells = holes)
            kept(i, j) = !((i, j) in holes) && !(code_at(i, j) in (0, 7))
            premasked = merge(lmdz, Dict{String, Any}("file" => lmdz_flux_file(joinpath(dir, "p$FT.nc"); keep = kept)))
            expect = Float64.(cells(quietly(() -> build_surface_flux_source(grid, :ref, premasked, FT;
                                                                            reference_time, run_span))))
            excl = quietly(() -> build_surface_flux_sources(
                grid, RUNNER._parse_tracer_specs(tagged_cfg(lmdz, holed; exclude = [0, 7])), FT;
                reference_time, run_span))
            @test length(excl) == 3
            @test maximum(abs.(Float64.(cell_sum(excl)) .- expect)) <= tol * maximum(abs.(expect))

            native = Dict{String, Any}("kind" => "cs_native", "file" => cs_native_file(joinpath(dir, "n$FT.nc")),
                                       "variable" => "F", "time_varying" => true, "species" => "co2")
            base = build_surface_flux_source(grid, :ref, native, FT; reference_time, run_span)
            cs_mask = cs_mask_file(joinpath(dir, "csm$FT.nc"))
            all_native = quietly(() -> build_surface_flux_sources(
                grid, RUNNER._parse_tracer_specs(tagged_cfg(native, cs_mask)), FT; reference_time, run_span))
            @test cell_sum(all_native) == cells(base)                          # exact 0/1 split
            m = read_region_mask(cs_mask, "region")
            excl = quietly(() -> build_surface_flux_sources(
                grid, RUNNER._parse_tracer_specs(tagged_cfg(native, cs_mask; exclude = [7])), FT;
                reference_time, run_span))
            # Excluded cells emit nothing; the others keep the flux exactly.
            @test all(1:6) do p
                summed = reduce(+, (src.cell_mass_rate_series[p] for src in excl))
                summed == base.cell_mass_rate_series[p] .* (m.codes[:, :, p] .!= 7)
            end
        end
    end
end

@testset "a mask on another grid is refused" begin
    mktempdir() do dir
        static = Dict{String, Any}("kind" => "file", "file" => static_flux_file(joinpath(dir, "f.nc")),
                                   "variable" => "F", "species" => "co2")
        coarse = ll_mask_file(joinpath(dir, "coarse.nc"); lon = collect(10.0:20:350),
                              lat = collect(-80.0:20:80), code = (i, j) -> 1)
        msg = message(() -> quietly(() -> build_surface_flux_sources(
            ll_grid(), RUNNER._parse_tracer_specs(tagged_cfg(static, coarse)), Float64)))
        @test occursin("lat-lon grid 18×9", msg) && occursin("36×18", msg) &&
              occursin("the mask must be on the flux's own grid", msg)
        reference_time, run_span = DateTime(2022, 1, 1), (0.0, 86400.0)
        native = Dict{String, Any}("kind" => "cs_native", "file" => cs_native_file(joinpath(dir, "n.nc")),
                                   "variable" => "F", "time_varying" => true, "species" => "co2")
        msg = message(() -> build_surface_flux_sources(
            cs_grid(), RUNNER._parse_tracer_specs(tagged_cfg(native, ll_mask_file(joinpath(dir, "m.nc")))),
            Float64; reference_time, run_span))
        @test occursin("must be a cubed-sphere mask", msg)
        msg = message(() -> build_surface_flux_sources(
            cs_grid(), RUNNER._parse_tracer_specs(tagged_cfg(native, cs_mask_file(joinpath(dir, "c4.nc"); Nc = 4))),
            Float64; reference_time, run_span))
        @test occursin("C4 cubed-sphere mask", msg) && occursin("C$(NC) cube", msg)
        # Same shape, other panel layout: the stored centres are checked.
        for (file, needle) in ((cs_mask_file(joinpath(dir, "shift.nc"); lon_shift = 15.0), "from the runtime mesh centres"),
                               (cs_mask_file(joinpath(dir, "bare.nc"); centres = false), "no 'lons'/'lats'"))
            msg = message(() -> build_surface_flux_sources(
                cs_grid(), RUNNER._parse_tracer_specs(tagged_cfg(native, file)), Float64; reference_time, run_span))
            @test occursin(needle, msg)
        end
        lmdz = Dict{String, Any}("kind" => "lmdz_co2", "file" => lmdz_flux_file(joinpath(dir, "l.nc")),
                                 "time_varying" => true, "species" => "co2")
        msg = message(() -> build_surface_flux_sources(
            cs_grid(), RUNNER._parse_tracer_specs(tagged_cfg(lmdz, cs_mask_file(joinpath(dir, "c.nc")))),
            Float64; reference_time, run_span))
        @test occursin("cubed-sphere mask, but", msg)
        # Every listed file of a series is checked, also one the run does not read.
        other = joinpath(dir, "other.nc")
        NCDataset(other, "c") do ds
            defDim(ds, "lon", 18); defDim(ds, "lat", 9); defDim(ds, "time", 1)
            defVar(ds, "lon", Float64, ("lon",))[:] = collect(10.0:20:350)
            defVar(ds, "lat", Float64, ("lat",))[:] = collect(-80.0:20:80)
            defVar(ds, "time", Float64, ("time",), attrib = Dict("units" => "hours since 2022-02-01 00:00:00"))[:] = [0.0]
            defVar(ds, "flux_apos", Float64, ("lon", "lat", "time"), attrib = Dict("units" => "kgC m-2 s-1"))[:, :, :] =
                zeros(18, 9, 1)
        end
        series = Dict{String, Any}("kind" => "lmdz_co2", "files" => [lmdz["file"], other],
                                   "time_varying" => true, "species" => "co2")
        msg = message(() -> build_surface_flux_sources(
            cs_grid(), RUNNER._parse_tracer_specs(tagged_cfg(series, ll_mask_file(joinpath(dir, "m2.nc")))),
            Float64; reference_time, run_span))
        @test occursin("other.nc", msg) && occursin("18×9", msg)
    end
end
