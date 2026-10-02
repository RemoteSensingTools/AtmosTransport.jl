# Shared helpers for the observation-sampling tests (test/core/test_observation_*.jl).
module ObservationTestFixtures

using AtmosTransport, Dates, Logging
using AtmosTransport.Output: observation_output_spec

export ORIGIN, GRAVITY, unix, fake_ll_model, sampler_spec, table_source, quiet

const ORIGIN = DateTime(2021, 12, 2)
const GRAVITY = 9.80665
unix(dt) = datetime2unix(dt)

"""
A 4×3 lat-lon column model with 3 levels (p_top = 100 Pa) holding tracers
`co2` (`q`) and `ch4`; air mass varies slightly by column.
"""
function fake_ll_model(; q = 400e-6, FT = Float64)
    mesh = LatLonMesh(; FT = FT, Nx = 4, Ny = 3)
    A = FT[100.0, 0.0, 0.0, 0.0]
    B = FT[0.0, 0.1, 0.5, 1.0]
    grid = AtmosGrid(mesh, HybridSigmaPressure(A, B), CPU(); FT = FT)
    ps = 98_000.0
    air = Array{FT}(undef, 4, 3, 3)
    for k in 1:3, j in 1:3, i in 1:4
        dp = (A[k + 1] + B[k + 1] * ps) - (A[k] + B[k] * ps)
        air[i, j, k] = FT(dp * cell_area(mesh, i, j) / GRAVITY * (1 + 0.01 * i))
    end
    state = CellState(DryBasis, air; co2 = air .* FT(q), ch4 = air .* FT(1.9e-6))
    return (; state, grid), mesh
end

"Parse an `[output.observations]` spec rooted in `dir` (`obs.nc` or `obs_{YYYYMMDD}.nc`)."
function sampler_spec(dir; split = "single", sources, kwargs...)
    obs_path = split == "daily" ? joinpath(dir, "obs_{YYYYMMDD}.nc") : joinpath(dir, "obs.nc")
    output_cfg = Dict{String, Any}("path" => joinpath(dir, "snap.nc"), "hours" => [0.0], "split" => split,
        "observations" => Dict{String, Any}("path" => obs_path, "sources" => sources,
                                            Dict(String(k) => v for (k, v) in kwargs)...))
    return observation_output_spec(output_cfg)
end

table_source(path; mode = "soundings") = Dict{String, Any}("kind" => "table", "mode" => mode, "path" => path)

"Run `f` with logging and standard streams silenced."
function quiet(f)
    with_logger(NullLogger()) do
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                f()
            end
        end
    end
end

end
