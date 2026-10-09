# ---------------------------------------------------------------------------
# Footprint result + tape-byte estimate result types.
#
# `CSFootprintResult` is the user-facing return value of the CS reverse-mode
# surface-emission footprint API; `CSTapeByteEstimate` is the diagnostic
# returned by `cs_tape_byte_estimate` (TapeRecording.jl) to size
# in-memory or on-disk tape allocations.
#
# Loaded into the `Adjoints` module via an `include` from
# `src/Adjoints/Adjoints.jl`.
# ---------------------------------------------------------------------------

"""
    CSFootprintResult

Reverse-mode footprint result for one scalar objective. `footprints[t]`
is an `NTuple{6}` of `(Nc, Nc)` arrays containing `dJ / dE`, where `E` is
the per-cell model-storage emission rate applied at the midpoint of model
step `t`. Storage is dry mixing ratio × carrier-air mass; these rates are
not physical kg-species per second and are not divided by cell area.
The derivative includes the timestep factor. `lag_steps[t] == nsteps - t`.
"""
struct CSFootprintResult{FT, O <: AbstractCSFootprintObjective, A2 <: AbstractArray{FT, 2}}
    objective::O
    footprints::Vector{NTuple{6, A2}}
    lag_steps::Vector{Int}
    dt::FT
    # Compatibility field from the earlier finite-difference prototype;
    # reverse-mode results set it to zero.
    epsilon::FT
    # Not evaluated by the reverse pass. Current built-in objectives only
    # need dJ/drm at final time, independent of final tracer mass.
    base_value::FT
end

"""
    CSTapeByteEstimate

Counts and byte estimate for the CS adjoint tape. Each `*_records` field
is an **op count** — the number of records of that type the forward
pass will push onto the tape — except `state_records` which is a
**payload-staging count** (full panel tuples written, doubled for
nonlinear schemes that stage both `panels_m` and `panels_rm` per
sweep).

`state_bytes = state_records * bytes_per_state` is the raw panel-data
cost of the tape; halo, midpoint, and the diffusion-palindrome's two
op records contribute scalar metadata only and are not counted in
`state_bytes`.

`total_records` is the **op count**:
`sweep + halo + midpoint + diffusion + convection`. It is not in
general equal to `state_records + halo + midpoint`, because (a)
nonlinear schemes have `state_records = 2 * sweep_records` and (b)
the diffusion palindrome contributes two op records per step but
stages only one panel tuple.
"""
struct CSTapeByteEstimate
    nsteps::Int
    sweep_records::Int
    halo_records::Int
    midpoint_records::Int
    diffusion_records::Int
    convection_records::Int
    state_records::Int
    total_records::Int
    bytes_per_state::Int
    state_bytes::Int
end
