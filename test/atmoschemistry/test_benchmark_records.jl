using Test
using TOML

@testset "AtmosChemistry adapter benchmark records" begin
    results = normpath(joinpath(
        @__DIR__, "..", "..", "scripts", "benchmarks", "results"))
    records = filter(readdir(results; join = true)) do path
        endswith(path, ".toml")
    end
    @test !isempty(records)

    lifecycle_records = filter(records) do path
        record = TOML.parsefile(path)
        get(record, "schema_version", 0) >= 3 &&
            get(record, "benchmark", "") ==
                "AtmosTransport AtmosChemistry adapter"
    end
    @test !isempty(lifecycle_records)

    required_timings = (
        "state_staging_seconds",
        "workspace_allocation_seconds",
        "first_coupling_seconds",
        "adapter_state_to_first_result_seconds",
        "coupling_median_seconds",
    )
    for path in lifecycle_records
        record = TOML.parsefile(path)
        @test all(haskey(record, key) for key in required_timings)
        @test !isempty(record["gpu_uuid"])
        @test record["planned_backend_bytes"] +
              record["planned_host_bytes"] ==
              record["measured_incremental_logical_owned_bytes"]
        @test record["matched_state_checks_passed"] === true
        if record["schema_version"] >= 4
            @test record["comparison_order"] ==
                  "alternating_adapter_and_core"
        end
        @test all(!isempty(value) for value in values(record["provenance"]))
        if record["has_host_forcing_stage"] === false
            @test record["planned_host_forcing_stage_bytes"] == 0
        end
    end
end
