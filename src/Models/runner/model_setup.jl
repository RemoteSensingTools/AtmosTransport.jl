"""
    _assert_gpu_residency!(state, arch)

When a GPU backend is selected, assert that `state.air_mass` lives on that
backend. Called once after model construction, before the run loop; a backend
mismatch raises an error that identifies the state storage.
"""
function _assert_gpu_residency!(state, arch)
    is_gpu(arch) || return nothing
    backing = assert_residency!(state.air_mass, arch; label = "state.air_mass")
    wrapper = Base.typename(typeof(backing)).wrapper
    @info @sprintf("[gpu verified] backend=%s backing=%s device=%s",
                   String(backend_name(arch)),
                   String(nameof(wrapper)),
                   device_name(arch))
    return nothing
end

# ===========================================================================
# Model construction
#
# Initial conditions are dry mixing ratios. The basis-aware packer converts
# them to conservative storage; moist carrier mass requires humidity (`qv`).
# ===========================================================================

function _allocate_structured_runner_fluxes(mesh::LatLonMesh, Nz::Int, FT, basis)
    return allocate_face_fluxes(mesh, Nz; FT = FT, basis = basis)
end

function _allocate_structured_runner_fluxes(mesh::ReducedGaussianMesh, Nz::Int, FT, basis)
    return allocate_face_fluxes(mesh, Nz; FT = FT, basis = basis)
end

function _allocate_structured_runner_fluxes(mesh, _Nz::Int, _FT, _basis)
    throw(ArgumentError(
        "TransportBinaryDriver model construction requires a lat-lon or " *
        "reduced-Gaussian grid; got $(typeof(mesh))"))
end

function _allocate_cs_runner_fluxes(mesh::CubedSphereMesh, Nz::Int, FT, basis)
    return allocate_face_fluxes(mesh, Nz; FT = FT, basis = basis)
end

function _allocate_cs_runner_fluxes(mesh, _Nz::Int, _FT, _basis)
    throw(ArgumentError(
        "TransportBinaryDriver returned incompatible horizontal grid " *
        "$(typeof(mesh)); expected CubedSphereMesh"))
end

# Initial humidity is an endpoint field when the binary carries one. Native CS
# physics binaries instead carry the same window-start qv in the VDIFF state.
_initial_condition_qv(window) = window.qv_start !== nothing ? window.qv_start :
                                window.vdiff === nothing ? nothing : window.vdiff.qv

# Initialize one tracer at a time in its final packed slot. `tracer_init`
# iterates `name => init_cfg` pairs, and the packed tracer order is that
# iteration order: the runner passes the specs of `_parse_tracer_specs`, sorted
# by name as on lat-lon and reduced-Gaussian grids.
function _initialize_cs_dry_state(grid::AtmosGrid{<:CubedSphereMesh},
                                  air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                  tracer_init; surface_pressure,
                                  pressure_basis = TotalSurfacePressure(),
                                  qv = nothing) where FT
    isempty(tracer_init) && throw(ArgumentError("at least one tracer must be configured"))
    names = Tuple(name for (name, _) in tracer_init)
    allunique(names) || throw(ArgumentError("duplicate tracer names in $(names)"))
    raw = ntuple(p -> similar(air_mass[p], size(air_mass[p])..., length(names)), 6)
    # Reuse one interior VMR buffer across analytic initializers. Each tracer
    # is copied into its independent packed slot before the buffer is reused.
    vmr = nothing
    for (index, (_name, init_cfg)) in enumerate(tracer_init)
        vmr = _build_cs_initial_mixing_ratio(
            air_mass, grid, init_cfg, vmr;
            surface_pressure, pressure_basis, mass_basis = DryBasis(), qv)
        destination = ntuple(p -> selectdim(raw[p], 4, index), 6)
        _cs_pack_interior_into_halo!(destination, grid, air_mass, vmr, nothing)
    end
    return CubedSphereState(DryBasis, air_mass, raw, names;
                           halo_width = grid.horizontal.Hp)
end

function _make_structured_model(driver::TransportBinaryDriver;
                                FT::Type{<:AbstractFloat},
                                recipe,
                                tracer_specs,
                                arch,
                                start_window::Integer = 1)
    grid = driver_grid(driver)
    mesh = grid.horizontal
    _require_global_longitude_span(mesh)
    window = load_transport_window(driver, Int(start_window))
    air_mass = copy(window.air_mass)

    tracer_specs_tuple = Tuple(tracer_specs)
    isempty(tracer_specs_tuple) && throw(ArgumentError("at least one tracer must be configured"))

    basis_type = air_mass_basis(driver) == :dry ? DryBasis : MoistBasis
    basis = basis_type()
    qv = _initial_condition_qv(window)
    pressure_basis = surface_pressure_basis(driver)
    tracer_names_tup = Tuple(spec.name for spec in tracer_specs_tuple)
    rm_arrays = map(tracer_specs_tuple) do spec
        vmr = build_initial_mixing_ratio(air_mass, grid, spec.init_cfg;
                                         surface_pressure = window.surface_pressure,
                                         pressure_basis, mass_basis = basis, qv)
        return pack_initial_tracer_mass(grid, air_mass, vmr;
                                        mass_basis = basis, qv)
    end

    tracer_tuple = NamedTuple{tracer_names_tup}(Tuple(rm_arrays))
    state = CellState(basis_type, air_mass; tracer_tuple...)
    fluxes = _allocate_structured_runner_fluxes(
        mesh, nlevels(grid), FT, basis_type)
    model = TransportModel(state, fluxes, grid, recipe.advection;
                           diffusion = recipe.diffusion,
                           convection = recipe.convection,
                           chemistry = recipe.chemistry)
    adaptor = array_adapter(arch)
    return adaptor === Array ? model : Base.invokelatest(Adapt.adapt, adaptor, model)
end

# Structured transport closes the zonal boundary periodically. Regional
# LatLonMesh values remain useful for source/regridding grids, but they cannot
# enter the transport runner without silently joining their west/east edges.
_require_global_longitude_span(::ReducedGaussianMesh) = nothing
function _require_global_longitude_span(mesh::LatLonMesh)
    span = Float64(last(mesh.λᶠ) - first(mesh.λᶠ))
    tolerance = max(64eps(Float64), 4eps(eltype(mesh))) * 360.0
    isapprox(span, 360.0; rtol = 0.0, atol = tolerance) ||
        throw(ArgumentError(
            "lat-lon transport requires a global 360-degree longitude span " *
            "because the zonal boundary is periodic; got $(span) degrees."))
    return nothing
end

# Snapshot capture and NetCDF writing live in `AtmosTransport.Output`. The
# runner only decides when to sample; the output module owns topology-specific
# diagnostics and file layout.

# ===========================================================================
# Capability validation
#
# Validate TOML physics against binary capabilities BEFORE constructing the
# model, so users get a precise error up front instead of silently failing
# partway through. Runs after `build_runtime_physics_recipe` (which already
# validates kind strings against recipe types) but before model construction
# (which discovers problems at the first load).
# ===========================================================================

function _validate_capability_match(driver, recipe)
    _validate_convection_capability(recipe.convection,
                                     binary_capabilities(driver.reader))
    return nothing
end

# Dispatch on the concrete convection-operator type so a new operator is a
# new method (compile-time coverage), not a new branch in an if-chain. The
# raw `cfg` no longer participates — the recipe has already been built and
# its convection field is authoritative.
_validate_convection_capability(::NoConvection, _caps) = nothing

function _validate_convection_capability(::TM5Convection, caps)
    caps.tm5_convection || throw(ArgumentError(
        "[convection] kind = \"tm5\" requires the binary to carry " *
        "entu, detu, entd, detd; this binary's payload_sections are " *
        "$(caps.payload_sections). Regenerate with a TM5-enabled " *
        "preprocessor or set convection.kind = \"none\"."))
    return nothing
end

function _validate_convection_capability(::CMFMCConvection, caps)
    caps.cmfmc_convection || throw(ArgumentError(
        "[convection] kind = \"cmfmc\" requires the binary to carry " *
        "the cmfmc section; this binary's payload_sections are " *
        "$(caps.payload_sections)."))
    return nothing
end

# The matrix variant has NO Tiedtke fallback — `dtrain` is the explicit
# detrainment rate that closes the continuity derivation
# `entu - detu = cmfmc[k] - cmfmc[k+1]`. A binary with cmfmc but no dtrain is
# hard-rejected up front so the failure mode is actionable at recipe-validation
# time (not at the first window load several seconds later).
function _validate_convection_capability(::CMFMCMatrixConvection, caps)
    (caps.cmfmc_convection && :dtrain in caps.payload_sections) ||
        throw(ArgumentError(
            "[convection] kind = \"cmfmc_matrix\" requires the binary " *
            "to carry both cmfmc AND dtrain payloads (no Tiedtke fallback); " *
            "this binary's payload_sections are $(caps.payload_sections). " *
            "Regenerate the binary with a preprocessor that emits :dtrain, " *
            "or fall back to kind=\"cmfmc\" which has a Tiedtke path."))
    return nothing
end

# Catch-all for any future convection operator. Forces a method to be added
# here when a new operator type appears, which is the whole point of the
# dispatch refactor.
function _validate_convection_capability(op::AbstractConvection, _caps)
    throw(ArgumentError(
        "no _validate_convection_capability method for $(typeof(op)); " *
        "add a method in src/Models/runner/model_setup.jl when introducing a new convection " *
        "operator type."))
end
