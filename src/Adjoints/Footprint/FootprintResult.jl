# ---------------------------------------------------------------------------
# Footprint result + tape-byte estimate result types.
#
# `CSFootprintResult` is the user-facing return value of the CS reverse-mode
# surface-emission footprint API; `CSTapeByteEstimate` is the diagnostic
# returned by `cs_tape_byte_estimate` (TapeRecording.jl) to size
# in-memory or on-disk tape allocations.
#
# Loaded by `Adjoints.Footprint`.
# ---------------------------------------------------------------------------

"""
    CSFootprintResult

Reverse-mode result for one scalar objective. `footprints[t]` is an
`NTuple{6}` of `(Nc, Nc)` arrays containing `dJ / dE`, where `E` is the
per-cell model-storage emission rate the surface-flux operator applies at the
palindrome center of model step `t`. Storage is dry mixing ratio × carrier-air
mass; these rates are not physical kg-species per second and are not divided
by cell area. The derivative includes the timestep factor.
`lag_steps[t] == nsteps - t`. A step plan without a surface-flux operator
(`surface_flux = false`) has no emission entry point and an empty
`footprints`.

`initial_adjoint` is `dJ/drm` at the initial state: an `NTuple{6}` of
halo-padded panels like the initial tracer mass (the halo cells are zero:
halos are a function of the interiors).
"""
struct CSFootprintResult{FT, O <: AbstractCSFootprintObjective, A2 <: AbstractArray{FT, 2},
                         A3 <: AbstractArray{FT, 3}}
    objective::O
    footprints::Vector{NTuple{6, A2}}
    lag_steps::Vector{Int}
    dt::FT
    initial_adjoint::NTuple{6, A3}
end

"""
    CSTapeByteEstimate

Counts and byte estimate for the CS adjoint tape. Each `*_records` field
is an **op count** — the number of records of that type the forward
pass will push onto the tape — except `state_records` which is a
**payload-staging count**: the six-panel tuples staged (air masses, the
limiter codes of monotone and CW84 PPM sweeps, the tracer masses of FV3
vertical sweeps), each once however many records read it.

`state_bytes` is the raw panel-data cost of the tape: `bytes_per_state` (one
air-mass state) per staged air or tracer mass and `bytes_per_state /
sizeof(FT)` per staged limiter-code tuple; Lin-Rood additionally retains
face-shaped tuples whose exact byte cost is included in `state_bytes`.
Halo, midpoint and end-of-step records and the diffusion records contribute
scalar metadata only and are not counted in `state_bytes`. On a split-sweep
tape the palindrome center's diffusion records and the reverse half's first
Z sweep share one staged air mass, and a convection record shares the next
step's first sweep's (or a `:preserve_vmr` reset's), staging its own only
before a `:preserve_tracer_mass` reset or at the end of the run.
`step_records` counts the end-of-step markers (one per step, after
convection), where the reverse pass injects observation forcing.

`total_records` is the **op count**:
`sweep + halo + midpoint + diffusion + convection + reset + step`; it is not
in general related to `state_records`, since records share staged tuples and
a limited PPM sweep stages two.
"""
struct CSTapeByteEstimate
    nsteps::Int
    sweep_records::Int
    halo_records::Int
    midpoint_records::Int
    diffusion_records::Int
    convection_records::Int
    step_records::Int
    state_records::Int
    total_records::Int
    bytes_per_state::Int
    state_bytes::Int
end
