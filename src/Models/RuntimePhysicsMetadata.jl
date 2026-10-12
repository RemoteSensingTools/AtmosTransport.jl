# Stable runtime-output labels. Physics families dispatch on the same typed specs
# that materialize the operators, so metadata cannot drift into a second string
# switch over TOML values.

_physics_limiter(::NoLimiter) = "none"
_physics_limiter(::MonotoneLimiter) = "monotone"
_physics_limiter(::CW84Limiter) = "cw84"

_physics_vertical(::SameAsHorizontal) = "same_as_horizontal"
_physics_vertical(::FV3ScalarProfile{true}) = "fv3_kord8"
_physics_vertical(::FV3ScalarProfile{false}) = "fv3_kord8_signed"
_physics_linrood_vertical(::UpwindScheme) = "upwind"
_physics_linrood_vertical(s::PPMScheme) = _physics_vertical(s.vertical)

_advection_attributes(::UpwindAdvectionSpec) = Dict(
    "physics_advection_scheme" => "upwind",
    "physics_advection_limiter" => "not_applicable",
    "physics_vertical_reconstruction" => "upwind",
)
_advection_attributes(::SlopesAdvectionSpec) = Dict(
    "physics_advection_scheme" => "slopes",
    "physics_advection_limiter" => "monotone",
    "physics_vertical_reconstruction" => "same_as_horizontal",
)
_advection_attributes(::NoAdvectionSpec) = Dict(
    "physics_advection_scheme" => "none",
    "physics_advection_limiter" => "not_applicable",
    "physics_vertical_reconstruction" => "none",
)
_advection_attributes(spec::PPMAdvectionSpec) = Dict(
    "physics_advection_scheme" => "ppm",
    "physics_advection_limiter" => _physics_limiter(spec.limiter),
    "physics_vertical_reconstruction" => _physics_vertical(spec.vertical),
)
_advection_attributes(spec::LinRoodAdvectionSpec) = Dict(
    "physics_advection_scheme" => "linrood_ppm",
    "physics_advection_order" => string(spec.order),
    "physics_advection_limiter" => "not_applicable",
    "physics_vertical_reconstruction" => _physics_linrood_vertical(spec.vertical),
)

_cloud_base_label(::CMFMCEdgeCloudBase) = "cmfmc"
_cloud_base_label(::ArchivedCloudBase) = "dqrcu"

_convection_attributes(::NoConvectionSpec) = Dict("physics_convection" => "none")
_convection_attributes(spec::CMFMCConvectionSpec) = Dict(
    "physics_convection" => "cmfmc",
    "physics_convection_clamp" => string(spec.clamp),
    "physics_convection_cloud_base" => _cloud_base_label(spec.cloud_base),
)
function _convection_attributes(spec::TM5ConvectionSpec)
    return _collab_convection_attributes("tm5", spec)
end
function _convection_attributes(spec::CMFMCMatrixConvectionSpec)
    return _collab_convection_attributes("cmfmc_matrix", spec)
end
function _collab_convection_attributes(kind, spec)
    return Dict(
        "physics_convection" => kind,
        "physics_convection_solver" => spec.use_collab_lu ? "collaborative_lu" : "column_lu",
        "physics_convection_lmax" => string(spec.lmax_conv),
        "physics_convection_level_merge" => string(spec.n_merge),
    )
end

_surface_coupling_label(coupled::Bool) = coupled ? "diffusive_boundary" : "split"
_diffusion_attributes(::NoDiffusionSpec) = Dict("physics_diffusion" => "none")
_diffusion_attributes(spec::ConstantDiffusionSpec) = Dict(
    "physics_diffusion" => "constant",
    "physics_diffusion_kz_m2_s" => string(spec.value),
    "physics_diffusion_surface_flux_coupling" => _surface_coupling_label(spec.surface_flux_boundary),
)
_diffusion_attributes(spec::WindowPBLKzDiffusionSpec) = _diffusion_attributes(
    "tm5_beljaars_viterbo_local_kz", spec.surface_flux_boundary)
_diffusion_attributes(spec::HoltslagBovilleVdiffDiffusionSpec) = _diffusion_attributes(
    "geoschem_holtslag_boville_vdiff", spec.surface_flux_boundary)
_diffusion_attributes(spec::TM5DkgDiffusionSpec) = _diffusion_attributes(
    "tm5_dkg", spec.surface_flux_boundary)
_diffusion_attributes(::GCHPNonlocalVdiffDiffusionSpec) = Dict(
    "physics_diffusion" => "geoschem_nonlocal_vdiff",
    "physics_diffusion_surface_flux_coupling" => "diffusive_boundary",
)
_diffusion_attributes(kind, coupling) = Dict(
    "physics_diffusion" => kind,
    "physics_diffusion_surface_flux_coupling" => _surface_coupling_label(coupling),
)

_chemistry_attributes(::NoChemistrySpec) = Dict(
    "physics_chemistry" => "none",
    "physics_decay_half_lives_seconds" => "none",
)
function _chemistry_attributes(spec::DecayChemistrySpec)
    entries = sort!(["$(name)=$(getproperty(spec.half_lives, name))"
                     for name in keys(spec.half_lives)])
    return Dict(
        "physics_chemistry" => "exponential_decay",
        "physics_decay_half_lives_seconds" => join(entries, ","),
    )
end

_surface_flux_label(source::SurfaceFluxSource) = "$(source.tracer_name)=static"
_surface_flux_label(source::TimeVaryingSurfaceFluxSource) =
    "$(source.tracer_name)=time_varying:$(_flux_scheme_label(source.scheme))"
_flux_scheme_label(::StepwiseFlux) = "stepwise"
_flux_scheme_label(::LinearInterpFlux) = "linear"
_flux_scheme_label(::ConservativeMeanFlux) = "conservative_mean"

function _surface_flux_attributes(sources)
    labels = sort!([_surface_flux_label(source) for source in sources])
    return Dict("physics_surface_fluxes" => isempty(labels) ? "none" : join(labels, ","))
end

function _binary_provenance_attributes(driver, binary_paths)
    raw = driver.reader.header.raw_header
    value(key, default = "unknown") = string(get(raw, key, default))
    provenance_keys = ("preprocessor", "git_commit", "git_dirty", "generation_fingerprint")
    provenance = String["format_version=$(driver.reader.header.format_version)"]
    for key in provenance_keys
        haskey(raw, key) && push!(provenance, "$(key)=$(value(key))")
    end
    return Dict(
        "physics_met_source" => value("source_type"),
        "physics_binary_provenance" => join(provenance, ";"),
        "physics_binary_inputs" => join(basename.(binary_paths), ","),
    )
end

"Build the complete output metadata block from typed runtime specifications."
function runtime_physics_attributes(cfg, driver, surface_sources, mass_basis::Symbol,
                                    binary_paths)
    attributes = Dict{String, String}()
    merge!(attributes, _advection_attributes(advection_spec(_advection_section(cfg))))
    merge!(attributes, _diffusion_attributes(diffusion_spec(_diffusion_section(cfg))))
    merge!(attributes, _convection_attributes(convection_spec(_convection_section(cfg))))
    merge!(attributes, _chemistry_attributes(chemistry_spec(_chemistry_section(cfg))))
    merge!(attributes, _surface_flux_attributes(surface_sources))
    merge!(attributes, _binary_provenance_attributes(driver, binary_paths))
    attributes["physics_mass_basis"] = String(mass_basis)
    run_cfg = get(cfg, "run", Dict{String, Any}())
    attributes["physics_cadence"] = String(_resolve_physics_cadence(
        get(run_cfg, "physics_cadence", nothing)))
    attributes["physics_air_mass_reset_mode"] = String(_normalize_air_mass_reset_mode(
        get(run_cfg, "air_mass_reset_mode", "preserve_tracer_mass")))
    return attributes
end

