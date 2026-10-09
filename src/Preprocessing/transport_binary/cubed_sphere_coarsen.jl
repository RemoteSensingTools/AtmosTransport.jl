# Experimental nested cubed-sphere transport-binary coarsener.
#
# IMPORTANT: this path is intended for testing and has not yet been validated
# as a scientifically interchangeable replacement for direct preprocessing at
# the target resolution. It conservatively restricts the *existing transport
# operator*. In particular, block-summed `dkg` is the exact restriction of the
# C90 diffusion operator for block-uniform tracer concentrations; it is not a
# fresh nonlinear TM5 bldiff calculation on C30 meteorology.

const _EXPERIMENTAL_CS_COARSEN_SECTIONS = Set((
    :m, :am, :bm, :cm, :ps, :dm,
    :pblh, :ustar, :pbl_hflux, :t2m, :dkg,
))

_open_cs_coarsen_reader(path::AbstractString, ::Type{FT}) where FT =
    TransportBinaryReader(String(path); FT)
_load_cs_coarsen_window(reader, win::Int) = load_window!(reader, win)

@inline _cs_coarsen_geometry(header) = binary_geometry(header)
@inline _cs_coarsen_Nc(header) = _cs_coarsen_geometry(header).Nc
@inline _cs_coarsen_definition(geometry::CubedSphereBinaryGeometry) = geometry.definition

@inline function _coarsen_sum_cells3!(dst::AbstractArray{FT, 3},
                                      src::AbstractArray{FT, 3},
                                      ratio::Int) where FT
    Nc, Nz = size(dst, 1), size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        total = zero(FT)
        i0 = (i - 1) * ratio
        j0 = (j - 1) * ratio
        for jj in 1:ratio, ii in 1:ratio
            total += src[i0 + ii, j0 + jj, k]
        end
        dst[i, j, k] = total
    end
    return dst
end

@inline function _coarsen_weighted_cells2!(dst::AbstractMatrix{FT},
                                            src::AbstractMatrix{FT},
                                            source_area::AbstractMatrix,
                                            ratio::Int) where FT
    Nc = size(dst, 1)
    @inbounds for j in 1:Nc, i in 1:Nc
        numerator = 0.0
        denominator = 0.0
        i0 = (i - 1) * ratio
        j0 = (j - 1) * ratio
        for jj in 1:ratio, ii in 1:ratio
            area = Float64(source_area[i0 + ii, j0 + jj])
            numerator += Float64(src[i0 + ii, j0 + jj]) * area
            denominator += area
        end
        dst[i, j] = FT(numerator / denominator)
    end
    return dst
end

@inline function _coarsen_xfaces!(dst::AbstractArray{FT, 3},
                                  src::AbstractArray{FT, 3},
                                  ratio::Int) where FT
    Nc, Nz = size(dst, 2), size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
        fine_i = (i - 1) * ratio + 1
        j0 = (j - 1) * ratio
        total = zero(FT)
        for jj in 1:ratio
            total += src[fine_i, j0 + jj, k]
        end
        dst[i, j, k] = total
    end
    return dst
end

@inline function _coarsen_yfaces!(dst::AbstractArray{FT, 3},
                                  src::AbstractArray{FT, 3},
                                  ratio::Int) where FT
    Nc, Nz = size(dst, 1), size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
        fine_j = (j - 1) * ratio + 1
        i0 = (i - 1) * ratio
        total = zero(FT)
        for ii in 1:ratio
            total += src[i0 + ii, fine_j, k]
        end
        dst[i, j, k] = total
    end
    return dst
end

function _allocate_nested_cs_coarsen_buffers(::Type{FT}, Nc::Int, Nz::Int,
                                             npanel::Int;
                                             include_surface::Bool,
                                             include_dkg::Bool) where FT
    panels3(nz) = ntuple(_ -> Array{FT}(undef, Nc, Nc, nz), npanel)
    panels2() = ntuple(_ -> Array{FT}(undef, Nc, Nc), npanel)
    return (
        m = panels3(Nz),
        am = ntuple(_ -> Array{FT}(undef, Nc + 1, Nc, Nz), npanel),
        bm = ntuple(_ -> Array{FT}(undef, Nc, Nc + 1, Nz), npanel),
        cm = panels3(Nz + 1),
        ps = panels2(),
        dm = panels3(Nz),
        m_next = panels3(Nz),
        surface = include_surface ? (
            pblh = panels2(), ustar = panels2(),
            hflux = panels2(), t2m = panels2(),
        ) : nothing,
        dkg = include_dkg ? panels3(Nz) : nothing,
    )
end

function _coarsen_nested_cs_window!(dst, src, source_dm,
                                    source_area::AbstractMatrix,
                                    ratio::Int, source_steps::Int)
    npanel = length(src.m)
    Threads.@threads for p in 1:npanel
        _coarsen_sum_cells3!(dst.m[p], src.m[p], ratio)
        _coarsen_sum_cells3!(dst.dm[p], source_dm[p], ratio)
        _coarsen_xfaces!(dst.am[p], src.am[p], ratio)
        _coarsen_yfaces!(dst.bm[p], src.bm[p], ratio)
        _coarsen_sum_cells3!(dst.cm[p], src.cm[p], ratio)
        _coarsen_weighted_cells2!(dst.ps[p], src.ps[p], source_area, ratio)
        if dst.surface !== nothing
            _coarsen_weighted_cells2!(dst.surface.pblh[p], src.surface.pblh[p], source_area, ratio)
            _coarsen_weighted_cells2!(dst.surface.ustar[p], src.surface.ustar[p], source_area, ratio)
            _coarsen_weighted_cells2!(dst.surface.hflux[p], src.surface.hflux[p], source_area, ratio)
            _coarsen_weighted_cells2!(dst.surface.t2m[p], src.surface.t2m[p], source_area, ratio)
        end
        dst.dkg === nothing || _coarsen_sum_cells3!(dst.dkg[p], src.dkg[p], ratio)
        @inbounds for idx in eachindex(dst.m[p])
            dst.m_next[p][idx] = dst.m[p][idx] + dst.dm[p][idx]
        end
        # Source am/bm/cm are mass amounts per source substep. Multiplying by
        # the source schedule recovers the one-substep/full-window candidate;
        # the chosen target schedule is applied after the positivity scan.
        scale = eltype(dst.am[p])(source_steps)
        dst.am[p] .*= scale
        dst.bm[p] .*= scale
        dst.cm[p] .*= scale
    end
    return dst
end

function _source_header_symbol(header::AbstractDict, key::AbstractString,
                               default::Symbol)
    return Symbol(replace(lowercase(String(get(header, key, String(default)))),
                          '-' => '_', ' ' => '_'))
end

function _experimental_coarsen_metadata(reader, output_Nc::Int, ratio::Int)
    raw = reader.header.raw_header
    metadata = Dict{String, Any}(
        "preprocessor" => "experimental_nested_cs_binary_coarsener",
        "preprocessor_contract" => "experimental_nested_cs_operator_restriction_v1",
        "adaptive_substeps" => true,
        "experimental" => true,
        "validation_status" => "testing_only_not_yet_scientifically_validated",
        "experimental_caveat" =>
            "Nested CS operator restriction has write-time replay and positivity gates but still requires direct-C30 comparison and tracer validation.",
        "coarsening_method" => "nested_$(ratio)x$(ratio)_block_operator_restriction",
        "coarsening_source_Nc" => _cs_coarsen_Nc(reader.header),
        "coarsening_target_Nc" => output_Nc,
        "coarsening_ratio" => ratio,
        "coarsening_flux_semantics" =>
            "sum coarse-boundary fine faces; remove internal faces; recompute adaptive target schedule",
        "coarsening_dkg_semantics" =>
            "sum fine-column interface exchange kg_s; exact for block-uniform tracer, not fresh target-grid bldiff",
        "source_binary" => abspath(reader.path),
        "source_binary_bytes" => filesize(reader.path),
        "source_format_version" => reader.header.format_version,
        "source_steps_per_window_by_window" => copy(reader.header.steps_per_window_by_window),
        "source_preprocessor_contract" => get(raw, "preprocessor_contract", "unknown"),
        "source_type" => get(raw, "source_type", "transport_binary"),
        "target_type" => "cubed_sphere",
        "regrid_method" => "nested_block_operator_restriction",
        "poisson_balanced" => true,
    )
    for key in ("date", "mass_fix_enabled", "mass_fix_target_ps_dry_pa",
                "global_mass_pin_enabled", "global_mass_pin_target_kg",
                "vertical_mapping_method", "target_vertical_name",
                "target_coefficients", "merge_map", "merge_min_thickness_Pa")
        haskey(raw, key) && (metadata[key] = raw[key])
    end
    return metadata
end

"""
    coarsen_nested_cs_transport_binary(input_path, output_path; target_Nc=30, ...)

Experimentally restrict a nested cubed-sphere transport binary (for example
C90 to C30) one met window at a time. Cell-extensive quantities and `dkg` are
block-summed, coarse boundary faces are summed, intensive surface fields are
area weighted, and a new adaptive per-window transport schedule is selected.

This is a testing-only operator coarsener, not yet a validated substitute for
direct preprocessing at `target_Nc`. The output header records that caveat.
Only the payload used by the ERA5 N320 C90 L66 no-convection campaign is
currently accepted; unfamiliar physics sections are rejected explicitly.
"""
function coarsen_nested_cs_transport_binary(
        input_path::AbstractString,
        output_path::AbstractString;
        target_Nc::Integer = 30,
        substep_cfl_target::Real = 0.90,
        positivity_cfl_limit::Real = 0.95,
        max_steps_per_window::Integer = 4096,
        force::Bool = false,
        mark_gated::Bool = false)
    input_abs = abspath(input_path)
    output_abs = abspath(output_path)
    input_abs == output_abs && throw(ArgumentError("input and output paths must differ"))
    isfile(input_abs) || throw(ArgumentError("input binary does not exist: $(input_abs)"))
    isfile(output_abs) && !force && throw(ArgumentError(
        "output already exists: $(output_abs); pass force=true to replace it"))
    target = Int(target_Nc)
    target > 0 || throw(ArgumentError("target_Nc must be positive; got $(target)"))
    target_cfl = Float64(substep_cfl_target)
    hard_cfl = Float64(positivity_cfl_limit)
    0 < target_cfl < hard_cfl <= 1 || throw(ArgumentError(
        "require 0 < substep_cfl_target < positivity_cfl_limit <= 1; got " *
        "$(target_cfl), $(hard_cfl)"))
    max_steps = Int(max_steps_per_window)
    max_steps >= 1 || throw(ArgumentError("max_steps_per_window must be positive"))

    probe = _open_cs_coarsen_reader(input_abs, Float32)
    DiskFT = probe.header.float_bytes == 4 ? Float32 : Float64
    close(probe)
    reader = _open_cs_coarsen_reader(input_abs, DiskFT)
    writer = nothing
    started = time()
    try
        h = reader.header
        geometry = _cs_coarsen_geometry(h)
        source_Nc = geometry.Nc
        npanel = geometry.npanel
        npanel == 6 || throw(ArgumentError("only six-panel CS binaries are supported"))
        source_Nc > target && source_Nc % target == 0 || throw(ArgumentError(
            "nested coarsening requires source Nc > target Nc and exact divisibility; " *
            "got C$(source_Nc) -> C$(target)"))
        ratio = source_Nc ÷ target
        :dm in h.payload_sections || throw(ArgumentError(
            "source binary must contain forward endpoint :dm payloads"))
        delta_semantics(reader) === :forward_window_endpoint_difference ||
            throw(ArgumentError("source delta_semantics must be forward_window_endpoint_difference"))
        flux_kind = _source_header_symbol(h.raw_header, "flux_kind", :substep_mass_amount)
        flux_kind === :substep_mass_amount || throw(ArgumentError(
            "source flux_kind must be substep_mass_amount; got $(flux_kind)"))
        unsupported = setdiff(Set(h.payload_sections), _EXPERIMENTAL_CS_COARSEN_SECTIONS)
        isempty(unsupported) || throw(ArgumentError(
            "experimental coarsener does not yet define safe semantics for sections: " *
            join(sort!(String.(collect(unsupported))), ", ")))

        include_surface = all(s in h.payload_sections for s in
                              (:pblh, :ustar, :pbl_hflux, :t2m))
        any(s in h.payload_sections for s in (:pblh, :ustar, :pbl_hflux, :t2m)) == include_surface ||
            throw(ArgumentError("source contains an incomplete surface payload"))
        include_dkg = :dkg in h.payload_sections
        mesh = CubedSphereMesh(; Nc = source_Nc, Hp = 0, FT = DiskFT,
                               definition = mesh_definition(reader), radius = DiskFT(h.planet_radius_m))
        source_area = mesh.cell_areas
        buffers = _allocate_nested_cs_coarsen_buffers(
            DiskFT, target, h.nlevel, npanel;
            include_surface, include_dkg)

        mkpath(dirname(output_abs))
        metadata = _experimental_coarsen_metadata(reader, target, ratio)
        vc = HybridSigmaPressure(DiskFT.(h.A_ifc), DiskFT.(h.B_ifc))
        writer = open_streaming_cs_transport_binary(
            output_abs, target, npanel, h.nlevel, h.nwindow, vc;
            FT = DiskFT,
            dt_met_seconds = h.dt_met_seconds,
            half_dt_seconds = h.dt_met_seconds / 2,
            steps_per_window = 1,
            source_flux_sampling = _source_header_symbol(h.raw_header, "source_flux_sampling", :window_start_endpoint),
            air_mass_sampling = _source_header_symbol(h.raw_header, "air_mass_sampling", :window_start_endpoint),
            flux_sampling = _source_header_symbol(h.raw_header, "flux_sampling", :window_constant),
            flux_kind = :substep_mass_amount,
            include_flux_delta = true,
            mass_basis = h.mass_basis,
            include_surface,
            include_precomputed_dkg = include_dkg,
            panel_convention = String(geometry.panel_convention),
            cs_definition = String(_cs_coarsen_definition(geometry)),
            cs_coordinate_law = String(geometry.coordinate_law),
            cs_center_law = String(geometry.center_law),
            longitude_offset_deg = geometry.longitude_offset_deg,
            planet_radius = h.planet_radius_m,
            extra_header = metadata)

        schedule = Vector{Int}(undef, h.nwindow)
        worst_replay = 0.0
        worst_ratio = 0.0
        for win in 1:h.nwindow
            source = _load_cs_coarsen_window(reader, win)
            source_dm = load_flux_delta_window!(reader, win).dm
            source_steps = h.steps_per_window_by_window[win]
            _coarsen_nested_cs_window!(buffers, source, source_dm,
                                       source_area, ratio, source_steps)

            one_step = verify_substep_positivity_cs!(
                buffers.m, buffers.am, buffers.bm, buffers.cm;
                cfl_limit = target_cfl, m_next = buffers.m_next)
            isfinite(one_step.ratio) || error(
                "non-finite C$(target) positivity ratio in window $(win) at $(one_step.location)")
            steps = max(1, ceil(Int, one_step.ratio / target_cfl))
            steps <= max_steps || error(
                "window $(win) requires $(steps) substeps, exceeding max $(max_steps)")
            rescale_substep_amounts!((buffers.am..., buffers.bm..., buffers.cm...), 1, steps)

            gate = verify_cs_window_contract!(
                buffers.m, buffers.am, buffers.bm, buffers.cm, buffers.m_next,
                steps, win;
                replay_tol = replay_tolerance(DiskFT),
                positivity_cfl_limit = hard_cfl)
            schedule[win] = steps
            worst_replay = max(worst_replay, gate.replay.max_rel_err)
            worst_ratio = max(worst_ratio, gate.positivity.ratio)

            payload = (m = buffers.m, am = buffers.am, bm = buffers.bm,
                       cm = buffers.cm, ps = buffers.ps, dm = buffers.dm,
                       surface = buffers.surface, dkg = buffers.dkg)
            write_streaming_cs_window!(writer, payload, target, npanel)
            @info "experimental CS coarsen window" window=win source_steps target_steps=steps positivity_ratio=gate.positivity.ratio replay_rel=gate.replay.max_rel_err
        end
        set_streaming_steps_per_window_schedule!(writer, schedule)
        close_streaming_transport_binary!(writer)
        writer = nothing

        # Reopen to exercise the complete reader-side structural contract.
        check = _open_cs_coarsen_reader(output_abs, DiskFT)
        close(check)
        if mark_gated
            touch(output_abs * ".coarsen-gated")
        end
        return (
            input = input_abs,
            output = output_abs,
            source_Nc,
            target_Nc = target,
            ratio,
            schedule,
            worst_replay_rel = worst_replay,
            worst_positivity_ratio = worst_ratio,
            input_bytes = filesize(input_abs),
            output_bytes = filesize(output_abs),
            elapsed_seconds = time() - started,
            experimental = true,
        )
    catch
        if writer !== nothing
            isopen(writer.io) && close(writer.io)
            isfile(writer.staging_path) && rm(writer.staging_path; force = true)
        end
        rethrow()
    finally
        close(reader)
    end
end

export coarsen_nested_cs_transport_binary
