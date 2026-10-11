# Dashboard records built from the harness's raw results. Base + JSON3 only,
# so the test suite can load it by path without the benchmark environment.
# Include `labels.jl` first.

using JSON3

"""
    timed_steps_per_repeat(operator, steps) -> Int

Steps one timed repeat covers. For the model operators a repeat runs `steps`
model steps, and for `adjoint` one forward and reverse run over `steps`
steps. For `io` a repeat is one write-and-read round trip of the payload,
which is the io case's step. A case's `time_per_step_seconds` is a repeat's
wall time divided by this, and the raw record stores it as
`steps_per_repeat`.
"""
timed_steps_per_repeat(operator, steps::Integer) = Symbol(operator) === :io ? 1 : Int(steps)

# Phases that are not part of a synthetic benchmark step.
const _SKIPPED_PHASES = ("window_advance", "forcing_refresh", "chemistry")

# github-action-benchmark stores `extra` as a string; a JSON object would be
# published as "[object Object]". The grouped dashboard parses this JSON.
_extra_string(d::AbstractDict) = JSON3.write(d)

"""
    dashboard_records(results) -> Vector

github-action-benchmark records (`name`, `unit = "s/step"`, `value`, `extra`)
of the raw results that succeeded: one record per case with its median time
per step, and one `$(BREAKDOWN_SERIES_PREFIX)` record per phase. A raw
record's `phase_times_seconds` are totals accumulated over all timed repeats,
so a phase's time per step is its total divided by
`steps_per_repeat * repeats`. Every `extra` carries the commit and the
environment fingerprint (`julia_version`, `manifest_sha256` of the resolved
benchmark Manifest), so the dashboard shows where the environment changed.
"""
function dashboard_records(results)
    records = Any[]
    for r in results
        get(r, "status", "ok") == "ok" || continue
        metadata = r["metadata"]
        environment = Dict{String, Any}("commit" => metadata["commit"],
                                        "julia_version" => metadata["julia_version"],
                                        "manifest_sha256" => metadata["manifest_sha256"])
        extra = merge(environment, Dict{String, Any}(
            "device" => r["device_name"],
            "updates_per_second" => r["cell_tracer_updates_per_second"],
        ))
        for key in ("bytes_per_second", "adjoint_runs_per_second")
            haskey(r, key) && (extra[key] = r[key])
        end
        push!(records, Dict(
            "name" => r["name"],
            "unit" => "s/step",
            "value" => r["time_per_step_seconds"],
            "extra" => _extra_string(extra),
        ))
        timed_steps = r["steps_per_repeat"] * r["repeats"]
        for (phase, total_s) in r["phase_times_seconds"]
            phase in _SKIPPED_PHASES && continue
            per_step = total_s / timed_steps
            per_step <= 0 && continue
            push!(records, Dict(
                "name" => breakdown_series_name(r["name"], phase),
                "unit" => "s/step",
                "value" => per_step,
                "extra" => _extra_string(environment),
            ))
        end
    end
    return records
end
