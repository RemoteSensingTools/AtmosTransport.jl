# Lat-lon transport-binary to cubed-sphere transport-binary regrid path.

# Per-window contract gates (replay + per-substep positivity) live in
# `cubed_sphere_contracts.jl` — all three CS-producing preprocessors call the
# same `verify_cs_window_contract!` so no path can silently skip a check.

# The LL → CS regrid has no winds or OMEGA of its own (payload only), but it
# runs a cubed-sphere column balance whose closure is recorded per window.
_regrid_window_evidence(::Type{FT}, closure::AbstractColumnClosureRecord) where FT =
    FluxFidelityEvidenceBundle(
        NoFluxFidelityEvidence{CubedSphereTargetGeometry,FT,SourcePayloadOnly}(SourcePayloadOnly()),
        NoSourceOmegaEvidence{CubedSphereTargetGeometry,FT,SourceHasNoOmega}(SourceHasNoOmega()),
        closure)

# The balance and the `cm` diagnosis run for `base_steps` substeps; an adaptive
# output schedule then rescales `am`, `bm`, `cm` to `steps`. The record states
# the closure of the stored fluxes (`rescaled_column_closure_record`); call it
# before `m_next` is converted to the stored `dm`.
_regrid_stored_closure_record(bal_diag, cm_diag, am, bm, m, m_next, cs_grid,
                              base_steps::Int, steps::Int) =
    rescaled_column_closure_record(column_closure_record(bal_diag, cm_diag),
                                   am, bm, m, m_next, cs_grid.face_table,
                                   base_steps, steps, cs_grid.poisson_scratch)

# The pole rows of a lat-lon source are ordinary cells: the wind recovery takes
# their zonal wind from `am` like every other row. ERA5 spectral lat-lon
# binaries (the only writer of `spectral_half_dt_seconds` is `build_v4_header`)
# written before the fix of issue #58 zeroed the pole-row zonal fluxes before
# the column balance, which then put its correction there: the stored pole-row
# `am` is neither zero nor the wind, and only the header tells such a binary
# apart. Returns the refusal message, or `nothing` for an acceptable source.
function _pole_row_zonal_flux_refusal(raw::AbstractDict, path)
    haskey(raw, "spectral_half_dt_seconds") || return nothing
    get(raw, POLE_ROW_ZONAL_FLUX_KEY, nothing) == "open" && return nothing
    return "regrid_ll_binary_to_cs: $(basename(path)) is an ERA5 spectral lat-lon " *
           "binary without `$(POLE_ROW_ZONAL_FLUX_KEY) = \"open\"` in its header. " *
           "Binaries written before 2026-10-14 closed the zonal faces of the pole " *
           "rows (issue #58), so their pole-row wind is not in the file. Regenerate " *
           "the source with the current preprocessor."
end

# Output substep count of one CS window. The fluxes arrive scaled for `steps`
# substeps; an adaptive policy raises the count until the per-substep Courant
# ratio of the CS window meets its target and rescales the amounts (all of them
# are linear in 1/steps, the balance and `cm` included). A fixed policy keeps
# the requested count; the write-time gate then checks it.
function _select_regrid_output_steps!(policy::SubstepSchedulePolicy, steps::Int,
                                      m, am, bm, cm, m_next)
    policy.adaptive_substeps || return steps
    for _ in 1:ADAPTIVE_SUBSTEP_MAX_REFINEMENTS
        ratio = verify_substep_positivity_cs!(m, am, bm, cm;
                                              cfl_limit = policy.substep_cfl_target,
                                              m_next = m_next).ratio
        new_steps = next_substeps(policy, steps, ratio)
        new_steps == steps && break
        rescale_substep_amounts!((am..., bm..., cm...), steps, new_steps)
        steps = new_steps
    end
    return steps
end

function _ll_binary_to_cs_regridder_cache_key(grid::CubedSphereTargetGeometry,
                                              Nx_ll::Integer,
                                              Ny_ll::Integer)
    return Symbol("cs_binary_ll_to_cs_regridder_C", grid.Nc, "_",
                  Int(Nx_ll), "x", Int(Ny_ll))
end

function _get_or_build_ll_binary_to_cs_regridder!(cache,
                                                  ll_mesh,
                                                  grid::CubedSphereTargetGeometry,
                                                  Nx_ll::Integer,
                                                  Ny_ll::Integer)
    cache_key = _ll_binary_to_cs_regridder_cache_key(grid, Nx_ll, Ny_ll)
    t_reg = time()
    regridder, reused = cached_or_build!(cache, cache_key) do
        build_regridder(ll_mesh, grid.mesh; normalize=false, cache_dir=grid.cache_dir)
    end
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)
    @info @sprintf("  Regridder: %s%d→%d  nnz=%d (%.1fs)",
                   reused ? "reused " : "", n_src, n_dst,
                   length(regridder.intersections.nzval), time() - t_reg)
    return regridder
end

"""
    regrid_transport_binary(input_path, target_grid, out_path; kwargs...)

Generic transport-binary regrid extension point. Concrete target/source
combinations implement methods here instead of adding new topology-specific
scripts. The currently implemented production pair is LL transport binary to
cubed sphere.
"""
function regrid_transport_binary(input_path::String,
                                 target_grid::AbstractTargetGeometry,
                                 out_path::String; kwargs...)
    throw(ArgumentError(
        "No transport-binary regrid implementation for target=" *
        "$(nameof(typeof(target_grid))). Implement `regrid_transport_binary` " *
        "for this target/source pair. Existing production support is " *
        "lat-lon transport binary -> cubed_sphere; LL -> lower-resolution LL " *
        "is still an explicit missing contract."))
end

function regrid_transport_binary(input_path::String,
                                 target_grid::CubedSphereTargetGeometry,
                                 out_path::String; kwargs...)
    return regrid_ll_binary_to_cs(input_path, target_grid, out_path; kwargs...)
end

"""
    carried_tm5_convection_record(source_header, source_has_tm5) -> Dict{String, Any}

Header record of TM5 convection sections that a binary-to-binary tool copies
from its source: the source's `TM5_CONVECTION_SOURCE_KEY` (`"unrecorded"` when
it has none) and its `TM5_EC2TM_CONVENTION_KEY`, if present. Empty when the
source has no TM5 sections. Readers reject a lat-lon → cubed-sphere regrid with
TM5 sections and no source record: it predates this record and copied
sections of the retired lat-lon route.
"""
function carried_tm5_convection_record(source_header::AbstractDict, source_has_tm5::Bool)
    source_has_tm5 || return Dict{String, Any}()
    record = Dict{String, Any}(
        TM5_CONVECTION_SOURCE_KEY => get(source_header, TM5_CONVECTION_SOURCE_KEY, "unrecorded"))
    haskey(source_header, TM5_EC2TM_CONVENTION_KEY) &&
        (record[TM5_EC2TM_CONVENTION_KEY] = source_header[TM5_EC2TM_CONVENTION_KEY])
    return record
end

"""
    regrid_ll_binary_to_cs(ll_binary_path, cs_grid, out_path; FT=Float64, mass_basis=nothing)

Regrid an existing LL transport binary to a cubed-sphere binary.

Reads each window from the LL binary, recovers cell-center winds from am/bm,
conservatively regrids m/ps/winds to CS panels, rotates winds to panel-local
coordinates, reconstructs CS face fluxes, closes continuity against forward
mass endpoints, and stream-writes the CS binary.

This reuses the entire CS regrid/continuity/write infrastructure from the
spectral→CS path — the only difference is the data source (binary reader
instead of spectral synthesis).

The met interval comes from the source header. The LL winds of window `w` are
recovered with that window's own source scaling,
`flux_application_seconds(dt_met, steps_w, flux_kind)` with `steps_w` from
`steps_per_window_by_window` (the stored `am`, `bm` are amounts of each
window's own substep count). The CS output substep count is a property of the
CS grid, not of the source, whose count the narrow lat-lon pole-row cells set:
by default every CS window gets the smallest count whose per-substep Courant
ratio meets `positivity_cfl_limit` (the fluxes are reconstructed for one
substep, balanced, and rescaled), and the output carries that per-window
schedule. Passing `steps_per_window` fixes one count for all windows instead.

The pole rows of the source are ordinary cells whose zonal faces carry flux,
so the pole-row `u` is recovered from `am` like every other row. ERA5 spectral
lat-lon binaries written before the fix of issue #58 (2026-10-14) closed those
faces before the column balance; the stored pole-row `am` is then the balance
correction, not the wind. They are recognised by their header (an ERA5
spectral lat-lon binary without `pole_row_zonal_flux = "open"`) and refused
before any output is opened. Regenerate them with the current preprocessor.

## Keyword arguments
- `FT::Type = Float64` — on-disk float type for the output CS binary.
- `mass_basis::Union{Nothing, Symbol} = nothing` — output mass-basis label.
  `nothing` (default) = match the source. Setting this to a value that
  differs from the source's `mass_basis` currently errors: actual
  dry↔moist conversion requires loading the source's `qv` and applying
  `apply_dry_basis_native!`, which this function does not do. Invariant
  14 mandates `:dry` end-to-end; use a dry source.
- `steps_per_window::Union{Nothing, Integer} = nothing` — output substep
  count. `nothing` = per window from the CS Courant number (see above); an
  integer fixes the count for every window (stored per-substep fluxes scale
  with `1/steps_per_window`, the window-integrated transport does not).
- `allow_terminal_zero_tendency::Bool = false` — diagnostic-only escape hatch
  for legacy LL sources that do not carry `dm`. Production-safe regrids should
  leave this at `false` so the final CS window is closed against an explicit
  endpoint target instead of an inferred zero-tendency fallback.
- `column_balance_weights::Symbol = :hybrid_mass` — how the column Poisson balance
  spreads its correction over the levels (`:mass`, `:hybrid_b` or
  `:hybrid_mass`; see `AbstractColumnWeights`), from the source binary's hybrid
  `B` interfaces. Other weights than `:mass` are recorded in the header.
- `write_replay_check::Bool = true` — run the write-time replay-continuity
  gate. `false` is a diagnostic escape hatch; the output header then records
  `write_replay_check = false`.
- `run_cache = nothing` — optional `PreprocessorRunCache` used to reuse the
  LL→CS conservative regridder across calls in the same preprocessing run.
"""
function regrid_ll_binary_to_cs(ll_binary_path::String,
                                cs_grid::CubedSphereTargetGeometry,
                                out_path::String;
                                FT::Type{<:AbstractFloat} = Float64,
                                mass_basis::Union{Nothing, Symbol} = nothing,
                                allow_terminal_zero_tendency::Bool = false,
                                positivity_cfl_limit::Real = DEFAULT_POSITIVITY_CFL_LIMIT,
                                require_substep_positivity::Bool = true,
                                steps_per_window::Union{Nothing, Integer} = nothing,
                                cs_balance_tol::Real = DEFAULT_POISSON_BALANCE_TOL,
                                cs_balance_project_every::Integer = DEFAULT_POISSON_PROJECT_EVERY,
                                column_balance_weights::Symbol = :hybrid_mass,
                                write_replay_check::Bool = true,
                                run_cache = nothing)
    t_start = time()
    Nc = cs_grid.Nc
    validate_column_balance_weights(column_balance_weights, "regrid_ll_binary_to_cs")

    # --- Open LL binary reader ---
    reader = TransportBinaryReader(ll_binary_path; FT=FT)
    h = reader.header
    geometry = binary_geometry(reader)
    geometry isa LatLonBinaryGeometry || throw(ArgumentError(
        "LL-to-CS regridding requires a lat-lon binary; got $(grid_type(reader))"))
    # Decided from the header, before any output file is opened.
    pole_refusal = _pole_row_zonal_flux_refusal(h.raw_header, ll_binary_path)
    pole_refusal === nothing || (close(reader); throw(ArgumentError(pole_refusal)))
    if surface_pressure_basis(reader) isa DrySurfacePressure
        close(reader)
        throw(ArgumentError(
            "regrid_ll_binary_to_cs cannot transform a dry-surface-pressure input: " *
            "wind recovery and hybrid flux reconstruction require total pressure, " *
            "and the regrid path does not load humidity. Regenerate the LL source " *
            "with surface_pressure_basis=TotalSurfacePressure()."))
    end
    Nx_ll = geometry.Nx
    Ny_ll = geometry.Ny
    Nz = h.nlevel
    Nt = h.nwindow
    A_ifc = Float64.(h.A_ifc)
    B_ifc = Float64.(h.B_ifc)
    # The regrid keeps the source levels, so these are the output interfaces.
    balance_weights = column_weights(column_balance_weights, B_ifc)

    # Refuse silent basis relabeling. The function reads raw `m/am/bm/ps`
    # from the source and never touches `qv`, so mismatched basis produces
    # a mislabeled binary. Matching or unset is OK.
    source_basis = Symbol(h.mass_basis)
    output_basis = mass_basis === nothing ? source_basis : Symbol(mass_basis)
    output_basis === source_basis || throw(ArgumentError(
        "regrid_ll_binary_to_cs: requested mass_basis=$(output_basis) " *
        "differs from source header mass_basis=$(source_basis). This " *
        "function does not perform dry↔moist conversion (it would need " *
        "to load `qv` from the source and apply `apply_dry_basis_native!` " *
        "to m/am/bm). Regenerate the source on the desired basis, or omit " *
        "the `mass_basis` kwarg to match the source."))
    source_has_tm5 = has_tm5_convection(reader)
    source_has_surface = all(s in h.payload_sections for s in (:pblh, :ustar, :pbl_hflux, :t2m))
    source_surface_partial = any(s in h.payload_sections for s in (:pblh, :ustar, :pbl_hflux, :t2m)) &&
                             !source_has_surface
    source_surface_partial && throw(ArgumentError(
        "regrid_ll_binary_to_cs: source has a partial PBL surface payload; " *
        "expected all of pblh, ustar, pbl_hflux, t2m."))
    source_has_dm = :dm in h.payload_sections
    source_has_dm || allow_terminal_zero_tendency || throw(ArgumentError(
        "regrid_ll_binary_to_cs requires source `dm` payloads to close the final " *
        "CS window safely. Source $(basename(ll_binary_path)) lacks `dm`; " *
        "regenerate the LL binary with flux deltas or pass " *
        "`allow_terminal_zero_tendency=true` for diagnostic-only regrids."
    ))
    source_has_dm && delta_semantics(reader) === :forward_window_endpoint_difference ||
        !source_has_dm || throw(ArgumentError(
            "regrid_ll_binary_to_cs requires source delta_semantics = " *
            ":forward_window_endpoint_difference when `dm` is present, got " *
            "$(delta_semantics(reader))."
        ))
    source_mass_fix = Dict{String, Any}()
    for key in ("mass_fix_enabled", "mass_fix_target_ps_dry_pa")
        haskey(h.raw_header, key) && (source_mass_fix[key] = h.raw_header[key])
    end

    # Timestep metadata comes from the source header. The stored `am/bm` are
    # amounts over `flux_application_seconds` of each window's own source
    # substep count (`steps_per_window_by_window`; an adaptive LL schedule
    # varies by window), so winds are recovered with the per-window source
    # scaling.
    #
    # The output substep count belongs to the CS grid, not to the source: the
    # LL count is set by the LL Courant number, which the narrow pole-row cells
    # dominate (issue #58). By default each CS window gets the count its own
    # per-substep Courant ratio needs (fluxes reconstructed for one substep,
    # then rescaled; `_select_regrid_output_steps!`). An explicit
    # `steps_per_window` fixes the count for every window; the positivity gate
    # checks it.
    met_interval = Float64(h.dt_met_seconds)
    src_steps = Int.(h.steps_per_window_by_window)
    src_flux_kind = flux_kind(reader)
    steps_per_window === nothing || steps_per_window >= 1 || throw(ArgumentError(
        "steps_per_window must be ≥ 1, got $(steps_per_window)"))
    out_policy = SubstepSchedulePolicy(
        adaptive_substeps = steps_per_window === nothing,
        substep_cfl_target = positivity_cfl_limit,
        max_steps_per_window = DEFAULT_MAX_STEPS_PER_WINDOW)
    base_steps = steps_per_window === nothing ? 1 : Int(steps_per_window)
    out_schedule = fill(base_steps, Nt)
    @info steps_per_window === nothing ?
        @sprintf("  Output substeps: per window from the CS Courant number (target %.2f); source %d–%d",
                 positivity_cfl_limit, minimum(src_steps), maximum(src_steps)) :
        @sprintf("  Output substeps: %d per window (explicit); source %d–%d",
                 base_steps, minimum(src_steps), maximum(src_steps))
    src_dt_factor(win) = FT(flux_application_seconds(met_interval, src_steps[win], src_flux_kind))
    out_dt_factor = FT(flux_application_seconds(met_interval, base_steps, :substep_mass_amount))
    gravity = FT(STANDARD_GRAVITY)

    @info @sprintf("  LL source: %s (%d×%d×%d, %d windows)",
                   basename(ll_binary_path), Nx_ll, Ny_ll, Nz, Nt)
    @info @sprintf("  CS target: C%d (%d panels, %d levels)", Nc, CS_PANEL_COUNT, Nz)

    # --- Build LL source mesh for regridder ---
    # Reconstruct the LL mesh from the binary header metadata. Binaries written
    # before `planet_radius_m` was recorded were all preprocessed on the IFS
    # sphere. Source and target must share one sphere.
    src_radius = haskey(h.raw_header, "planet_radius_m") ? h.planet_radius_m : IFS_EARTH_RADIUS
    FT(src_radius) == cs_grid.mesh.radius || (close(reader); throw(ArgumentError(
        "LL source radius $(src_radius) m differs from the CS target radius $(cs_grid.mesh.radius) m")))
    ll_mesh = LatLonMesh(; FT=FT,
                          size=(Nx_ll, Ny_ll),
                          longitude=(-180, 180),
                          latitude=(-90, 90),
                          radius=FT(src_radius))
    ll_lats = FT.(ll_mesh.φᶜ)
    Δy_ll = FT(ll_mesh.radius * deg2rad(ll_mesh.Δφ))
    Δlon_ll = FT(deg2rad(ll_mesh.Δλ))

    # --- Build conservative regridder (LL → CS) ---
    regridder = _get_or_build_ll_binary_to_cs_regridder!(
        run_cache, ll_mesh, cs_grid, Nx_ll, Ny_ll)
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)

    # --- Allocate workspaces ---
    cs_ws = allocate_cs_preprocess_workspace(Nc, Nx_ll, Ny_ll, Nz, n_src, n_dst, FT)
    Δx = cs_grid.mesh.Δx
    Δy = cs_grid.mesh.Δy

    # Pre-allocate LL read buffers
    m_ll  = Array{FT}(undef, Nx_ll, Ny_ll, Nz)
    am_ll = Array{FT}(undef, Nx_ll + 1, Ny_ll, Nz)
    bm_ll = Array{FT}(undef, Nx_ll, Ny_ll + 1, Nz)
    cm_ll = Array{FT}(undef, Nx_ll, Ny_ll, Nz + 1)
    ps_ll = Array{FT}(undef, Nx_ll, Ny_ll)
    dm_ll = source_has_dm ? Array{FT}(undef, Nx_ll, Ny_ll, Nz) : nothing

    # Raw PBL surface fields are per-area/intensive diagnostics. They are
    # area-averaged to CS cell centers and carried verbatim in the output
    # binary; runtime derives Kz from them for the active met window.
    pblh_ll  = source_has_surface ? Array{FT}(undef, Nx_ll, Ny_ll) : nothing
    ustar_ll = source_has_surface ? Array{FT}(undef, Nx_ll, Ny_ll) : nothing
    hflux_ll = source_has_surface ? Array{FT}(undef, Nx_ll, Ny_ll) : nothing
    t2m_ll   = source_has_surface ? Array{FT}(undef, Nx_ll, Ny_ll) : nothing

    # TM5 convection sections are layer-center kg/m²/s.  Conservative
    # LL→CS regrid is the right path (per
    # `feedback_conservative_regrid_for_mass_fluxes.md`); no rotation,
    # no Poisson balance — they're purely vertical-flux carriers, the
    # horizontal advection sees zero divergence from them.
    entu_ll = source_has_tm5 ? Array{FT}(undef, Nx_ll, Ny_ll, Nz) : nothing
    detu_ll = source_has_tm5 ? Array{FT}(undef, Nx_ll, Ny_ll, Nz) : nothing
    entd_ll = source_has_tm5 ? Array{FT}(undef, Nx_ll, Ny_ll, Nz) : nothing
    detd_ll = source_has_tm5 ? Array{FT}(undef, Nx_ll, Ny_ll, Nz) : nothing

    # --- Build vertical coordinate from binary header ---
    vc_merged = HybridSigmaPressure(A_ifc, B_ifc)

    # --- Open streaming CS binary writer ---
    # Stage to a unique sibling file so the positivity gate (final step) can
    # error without leaving a bad binary at `out_path`. Renamed to the
    # final name on success.
    mkpath(dirname(out_path))
    tmp_path = open_preprocessing_staging(out_path)
    inner_writer = open_streaming_cs_transport_binary(
        tmp_path, Nc, CS_PANEL_COUNT, Nz, Nt, vc_merged;
        FT=FT,
        dt_met_seconds=met_interval,
        half_dt_seconds=met_interval / 2,
        steps_per_window=base_steps,
        include_flux_delta=true,
        include_surface=source_has_surface,
        include_tm5conv=source_has_tm5,
        mass_basis=output_basis,
        surface_pressure_basis=surface_pressure_basis(reader),
        # The regrid copies the source's TM5 sections and their basis record;
        # explicit `nothing` preserves a genuinely legacy keyless source.
        tm5_convection_mass_basis=tm5_convection_mass_basis(reader),
        cs_geometry_tags(cs_grid.mesh)...,
        planet_radius=cs_grid.mesh.radius,
        extra_header=with_replay_record(with_column_weights_record(merge(Dict{String, Any}(
            "preprocessor"      => "regrid_ll_binary_to_cs",
            "source_type"       => "ll_transport_binary",
            "source_path"       => ll_binary_path,
            "target_type"       => "cubed_sphere",
            "regrid_method"     => "conservative",
            "poisson_balanced"  => true,
            "horizontal_balance" => HORIZONTAL_BALANCE_TAG,
            "adaptive_substeps" => out_policy.adaptive_substeps,
        ), source_mass_fix, carried_tm5_convection_record(h.raw_header, source_has_tm5)),
           column_balance_weights), write_replay_check))
    writer = CubedSphereBinaryWriter(inner_writer, State.mass_basis(output_basis);
                                     Nc = Nc,
                                     npanel = CS_PANEL_COUNT,
                                     final_path = out_path)

    bytes_per_window = writer.inner.elems_per_window * sizeof(FT)
    expected_total = writer.inner.header_bytes + Nt * bytes_per_window
    @info @sprintf("  Output: %s (%.2f GB, %d windows)", basename(out_path),
                   expected_total / 1e9, Nt)
    @info "  Streaming: LL binary → CS regrid → balance → write..."
    write_replay_on = write_replay_check
    write_replay_on || @info "  Write-time CS replay gate SKIPPED (disabled for this run)"
    replay_tol = replay_tolerance(FT)

    # --- Helper: read one LL window and regrid to CS ---
    function _read_and_regrid_to_cs!(win_idx, m_out, ps_out, am_out, bm_out;
                                     pblh_out = nothing, ustar_out = nothing,
                                     hflux_out = nothing, t2m_out = nothing,
                                     entu_out = nothing, detu_out = nothing,
                                     entd_out = nothing, detd_out = nothing)
        load_window!(reader, win_idx; m=m_ll, ps=ps_ll, am=am_ll, bm=bm_ll, cm=cm_ll)

        # Conservative regrid:
        #   `m` (kg/cell, extensive) → density convert → regrid → ×dst_area
        #   `ps` (Pa, intensive) → straight area-averaged regrid
        # Without the `ExtensiveCellField()` tag, `m` would be treated as
        # intensive and the LL pole-row's small cells would distort the CS
        # polar mass by ~12×. See `Quantities.jl` for the full taxonomy.
        regrid_3d_to_cs_panels!(m_out, regridder, m_ll, cs_ws, Nc, ExtensiveCellField())
        regrid_2d_to_cs_panels!(ps_out, regridder, ps_ll, cs_ws, Nc, IntensiveCellField())

        # Recover LL cell-center winds from binary's am/bm
        recover_ll_cell_center_winds!(cs_ws.u_cc, cs_ws.v_cc,
            am_ll, bm_ll, ps_ll,
            A_ifc, B_ifc, ll_lats, Δy_ll, Δlon_ll,
            FT(ll_mesh.radius), gravity, src_dt_factor(win_idx))

        # Regrid geographic winds to CS + rotate to panel-local
        regrid_3d_to_cs_panels!(cs_ws.u_cs_panels, regridder, cs_ws.u_cc, cs_ws, Nc)
        regrid_3d_to_cs_panels!(cs_ws.v_cs_panels, regridder, cs_ws.v_cc, cs_ws, Nc)
        rotate_winds_to_panel_local!(cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                      cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                      cs_grid.tangent_basis, Nc, Nz)

        # Reconstruct CS face fluxes
        reconstruct_cs_fluxes!(am_out, bm_out, cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                               cs_ws.dp_panels, ps_out,
                               A_ifc, B_ifc, Δx, Δy, gravity, out_dt_factor, Nc, Nz)

        if source_has_surface
            load_surface_window!(reader, win_idx;
                                 pblh = pblh_ll, ustar = ustar_ll,
                                 hflux = hflux_ll, t2m = t2m_ll)
            regrid_2d_to_cs_panels!(pblh_out, regridder, pblh_ll, cs_ws, Nc, IntensiveCellField())
            regrid_2d_to_cs_panels!(ustar_out, regridder, ustar_ll, cs_ws, Nc, IntensiveCellField())
            regrid_2d_to_cs_panels!(hflux_out, regridder, hflux_ll, cs_ws, Nc, IntensiveCellField())
            regrid_2d_to_cs_panels!(t2m_out, regridder, t2m_ll, cs_ws, Nc, IntensiveCellField())
        end

        # TM5 sections (layer-center kg/m²/s): conservative regrid to
        # CS panels.  No rotation — entu/detu/entd/detd are scalar
        # vertical-flux carriers, not horizontal vectors.
        if source_has_tm5
            load_tm5_convection_window!(reader, win_idx;
                                         entu = entu_ll, detu = detu_ll,
                                         entd = entd_ll, detd = detd_ll)
            regrid_3d_to_cs_panels!(entu_out, regridder, entu_ll, cs_ws, Nc)
            regrid_3d_to_cs_panels!(detu_out, regridder, detu_ll, cs_ws, Nc)
            regrid_3d_to_cs_panels!(entd_out, regridder, entd_ll, cs_ws, Nc)
            regrid_3d_to_cs_panels!(detd_out, regridder, detd_ll, cs_ws, Nc)
        end
    end

    # --- Pre-allocate sliding buffer ---
    cur_m  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT)
    cur_ps = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
    cur_am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), CS_PANEL_COUNT)
    cur_bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), CS_PANEL_COUNT)
    cur_cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), CS_PANEL_COUNT)
    cur_pblh  = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    cur_ustar = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    cur_hflux = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    cur_t2m   = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    nxt_pblh  = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    nxt_ustar = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    nxt_hflux = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    nxt_t2m   = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    # TM5 sliding buffers: 2 sets — `cur_*` holds the window we're
    # writing, `nxt_*` holds the next window's read so we can swap
    # without re-allocating.  Both sets only allocate when the source
    # actually carries TM5 sections.
    cur_entu = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    cur_detu = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    cur_entd = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    cur_detd = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    nxt_entu = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    nxt_detu = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    nxt_entd = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    nxt_detd = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing



    worst_pre = 0.0; worst_post = 0.0; worst_iter = 0
    worst_replay_rel = 0.0
    worst_replay_abs = 0.0
    worst_replay_win = 0
    worst_replay_idx = (0, 0, 0, 0)
    worst_positivity = init_cs_positivity_accumulator()
    @info "  Applying column CS mass-balance correction; diagnosing cm from endpoint mass tendency..."

    function _emit_regridded_window!(win_idx)
        t_bal = time()
        bal_diag = balance_cs_column_mass_fluxes!(
            cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
            cs_grid.face_table, cs_grid.cell_degree, base_steps,
            cs_grid.poisson_scratch; tol=Float64(cs_balance_tol),
            max_iter=DEFAULT_POISSON_BALANCE_MAX_ITER,
            project_every=Int(cs_balance_project_every),
            weights=balance_weights,
            fixed_global_mass=output_basis == :dry &&
                (get(h.raw_header, "mass_fix_enabled", false) ||
                 get(h.raw_header, "global_mass_pin_enabled", false)))
        t_bal = time() - t_bal

        worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
        worst_post = max(worst_post, bal_diag.max_post_residual)
        worst_iter = max(worst_iter, bal_diag.max_cg_iter)


        fill_cs_window_mass_tendency!(cs_ws.dm_panels, cur_m, cs_ws.m_next_panels, base_steps)

        cm_diag = diagnose_balanced_cs_cm!(cur_cm, cur_am, cur_bm, cs_ws.dm_panels,
                                            cur_m, Nc, Nz, bal_diag,
                                            "LL-to-CS window $(win_idx)", balance_weights)
        out_schedule[win_idx] = _select_regrid_output_steps!(
            out_policy, base_steps, cur_m, cur_am, cur_bm, cur_cm, cs_ws.m_next_panels)
        closure = _regrid_stored_closure_record(bal_diag, cm_diag, cur_am, cur_bm, cur_m,
                                                cs_ws.m_next_panels, cs_grid, base_steps,
                                                out_schedule[win_idx])
        pos_diag = if write_replay_on
            contract = verify_cs_window_contract!(cur_m, cur_am, cur_bm, cur_cm,
                                                  cs_ws.m_next_panels,
                                                  out_schedule[win_idx], win_idx;
                                                  replay_tol = replay_tol,
                                                  positivity_cfl_limit = positivity_cfl_limit)
            if worst_replay_win == 0 || contract.replay.max_rel_err > worst_replay_rel
                worst_replay_rel = contract.replay.max_rel_err
                worst_replay_abs = contract.replay.max_abs_err
                worst_replay_win = win_idx
                worst_replay_idx = contract.replay.worst_idx
            end
            contract.positivity
        else
            verify_substep_positivity_cs!(cur_m, cur_am, cur_bm, cur_cm;
                                         cfl_limit = positivity_cfl_limit,
                                         m_next = cs_ws.m_next_panels)
        end
        worst_positivity = update_cs_positivity_accumulator(worst_positivity, pos_diag, win_idx)
        convert_cs_mass_target_to_delta!(cs_ws.m_next_panels, cur_m)

        window_nt = (m=cur_m, am=cur_am, bm=cur_bm, cm=cur_cm, ps=cur_ps,
                     dm=cs_ws.m_next_panels)
        if source_has_surface
            window_nt = merge(window_nt,
                              (surface=(pblh=cur_pblh, ustar=cur_ustar,
                                        hflux=cur_hflux, t2m=cur_t2m),))
        end
        if source_has_tm5
            window_nt = merge(window_nt,
                              (tm5_fields=(entu=cur_entu, detu=cur_detu,
                                           entd=cur_entd, detd=cur_detd),))
        end
        write_window!(writer, ReadyWindow{CubedSphereTargetGeometry, FT}(
            win_idx, window_nt, _regrid_window_evidence(FT, closure)))
        return (t_bal, bal_diag)
    end

    # --- Process first window ---
    t0 = time()
    _read_and_regrid_to_cs!(1, cs_ws.m_panels, cs_ws.ps_panels, cs_ws.am_panels, cs_ws.bm_panels;
                             pblh_out = cur_pblh, ustar_out = cur_ustar,
                             hflux_out = cur_hflux, t2m_out = cur_t2m,
                             entu_out = cur_entu, detu_out = cur_detu,
                             entd_out = cur_entd, detd_out = cur_detd)
    @info @sprintf("    Window  1/%d: read+regrid %.2fs", Nt, time() - t0)

    copy_cs_panels!(cur_m,  cs_ws.m_panels)
    copy_cs_panels!(cur_ps, cs_ws.ps_panels)
    copy_cs_panels!(cur_am, cs_ws.am_panels)
    copy_cs_panels!(cur_bm, cs_ws.bm_panels)
    # cur_entu/detu/entd/detd were filled directly by `_read_and_regrid_to_cs!`
    # (TM5 sections need no rebalancing — they're vertical-flux carriers).

    # --- Sliding-window loop: windows 2..Nt ---
    for win in 2:Nt
        t0 = time()
        _read_and_regrid_to_cs!(win, cs_ws.m_panels, cs_ws.ps_panels, cs_ws.am_panels, cs_ws.bm_panels;
                                 pblh_out = nxt_pblh, ustar_out = nxt_ustar,
                                 hflux_out = nxt_hflux, t2m_out = nxt_t2m,
                                 entu_out = nxt_entu, detu_out = nxt_detu,
                                 entd_out = nxt_entd, detd_out = nxt_detd)
        t_read = time() - t0

        copy_cs_panels!(cs_ws.m_next_panels, cs_ws.m_panels)

        t_bal, bal_diag = _emit_regridded_window!(win - 1)

        if should_log_window(win - 1, Nt)
            @info @sprintf("    Window %2d/%d: wrote (colbal %.2fs pre=%.2e post=%.2e iter=%d) | read %2d (%.2fs)",
                           win - 1, Nt, t_bal, bal_diag.max_pre_residual,
                           bal_diag.max_post_residual, bal_diag.max_cg_iter, win, t_read)
        end

        copy_cs_panels!(cur_m,  cs_ws.m_panels)
        copy_cs_panels!(cur_ps, cs_ws.ps_panels)
        copy_cs_panels!(cur_am, cs_ws.am_panels)
        copy_cs_panels!(cur_bm, cs_ws.bm_panels)
        if source_has_surface
            copy_cs_panels!(cur_pblh, nxt_pblh)
            copy_cs_panels!(cur_ustar, nxt_ustar)
            copy_cs_panels!(cur_hflux, nxt_hflux)
            copy_cs_panels!(cur_t2m, nxt_t2m)
        end
        if source_has_tm5
            copy_cs_panels!(cur_entu, nxt_entu)
            copy_cs_panels!(cur_detu, nxt_detu)
            copy_cs_panels!(cur_entd, nxt_entd)
            copy_cs_panels!(cur_detd, nxt_detd)
        end
    end

    # --- Balance & write LAST window ---
    if source_has_dm
        deltas = load_flux_delta_window!(reader, Nt; dm=dm_ll)
        (deltas !== nothing && haskey(deltas, :dm)) || throw(ArgumentError(
            "regrid_ll_binary_to_cs: source header for $(basename(ll_binary_path)) " *
            "declares `dm`, but window $(Nt) could not be loaded."
        ))
        @. cs_ws.u_cc = m_ll + dm_ll
        # `m + dm` is extensive (kg/cell): density-convert via the dispatcher.
        regrid_3d_to_cs_panels!(cs_ws.m_next_panels, regridder, cs_ws.u_cc,
                                cs_ws, Nc, ExtensiveCellField())
    else
        @warn "regrid_ll_binary_to_cs: using zero-tendency fallback for the final CS window because source `dm` is unavailable."
        copy_cs_panels!(cs_ws.m_next_panels, cur_m)
    end
    t_bal, bal_diag = _emit_regridded_window!(Nt)

    @info @sprintf("    Window %2d/%d (last): colbal %.2fs  pre=%.2e post=%.2e iter=%d",
                   Nt, Nt, t_bal, bal_diag.max_pre_residual,
                   bal_diag.max_post_residual, bal_diag.max_cg_iter)

    set_streaming_steps_per_window_schedule!(writer.inner, out_schedule)
    close_streaming_binary!(writer)
    close(reader)
    @info @sprintf("  Output substep schedule: min=%d max=%d (%s)",
                   minimum(out_schedule), maximum(out_schedule),
                   out_policy.adaptive_substeps ? "CS Courant number" : "explicit")

    @info @sprintf("  Column balance summary: pre=%.3e  post=%.3e  max_iter=%d",
                   worst_pre, worst_post, worst_iter)
    if write_replay_on
        replay_msg = worst_replay_win > 0 ?
            @sprintf("max rel=%.3e abs=%.3e kg win=%d cell=%s",
                     worst_replay_rel, worst_replay_abs, worst_replay_win, worst_replay_idx) :
            "no windows checked"
        @info "  Write-time replay gate: $replay_msg"
    end

    # Single contract summary — errors (and deletes the staging file) on
    # require_substep_positivity=true if the gate failed, otherwise warns.
    summarize_cs_positivity_status(worst_positivity;
                                   cfl_limit = positivity_cfl_limit,
                                   steps_per_window = maximum(out_schedule),
                                   require_substep_positivity = require_substep_positivity,
                                   quarantine_path = writer_staging_path(writer))
    # Reached only when positivity passed, or when require_substep_positivity=false
    # turned a violation into a warning. Promote the staged file either way.
    promote_streaming_binary!(writer)

    actual = filesize(out_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(out_path),
                   actual / 1e9, time() - t_start)
    actual == expected_total ||
        @warn @sprintf("File size mismatch: expected %d bytes, got %d", expected_total, actual)

    return out_path
end
