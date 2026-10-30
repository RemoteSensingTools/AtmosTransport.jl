@inline _ansi_enabled() =
    get(ENV, "NO_COLOR", "") == "" && get(ENV, "TERM", "dumb") != "dumb"

@inline function _ansi_style(text::AbstractString, code::AbstractString)
    return _ansi_enabled() ? string("\e[", code, "m", text, "\e[0m") : String(text)
end

@inline _bold(text::AbstractString) = _ansi_style(text, "1")
@inline _cyan(text::AbstractString) = _ansi_style(text, "1;36")

# Labels dispatch on the `vertical` field itself: methods specialised on a
# type parameter whose bound differs from the declared one are ambiguous or
# silently lose to the generic method.
_advection_label(scheme) = String(nameof(typeof(scheme)))
_advection_label(s::LinRoodPPMScheme{ORD}) where ORD = "Lin-Rood PPM$(ORD)" * _vertical_suffix(s.vertical)
_advection_label(s::PPMScheme) = "PPM" * _limiter_suffix(s.limiter) * _vertical_suffix(s.vertical)
_advection_label(::SlopesScheme) = "Slopes"
_advection_label(::UpwindScheme) = "Upwind"
_limiter_suffix(::Any) = ""
_limiter_suffix(::CW84Limiter) = ", CW84"
_vertical_suffix(::Any) = ""
_vertical_suffix(v::PPMScheme) = _vertical_suffix(v.vertical)          # Lin-Rood's vertical sweeps
_vertical_suffix(::FV3ScalarProfile{true}) = ", FV3 kord 8 vertical"
_vertical_suffix(::FV3ScalarProfile{false}) = ", FV3 kord 8 vertical (signed)"

_diffusion_label(op) = String(nameof(typeof(op)))

function _diffusion_label(op::ImplicitVerticalDiffusion)
    coupling = uses_diffusive_surface_flux_boundary(op) ? ", surface_flux=before_full_solve" :
               ", surface_flux=midpoint_split"
    return string(nameof(typeof(op)), coupling)
end

function _schedule_label(driver)
    schedule = steps_per_window_schedule(driver)
    if isempty(schedule)
        return "n/a"
    end
    lo, hi = extrema(schedule)
    if lo == hi
        return string(first(schedule))
    end
    return @sprintf("%d..%d, max=%d", lo, hi, steps_per_window(driver))
end

function _physics_summary_lines(; topology, mesh_label, levels, halo_width,
                                  backend, FT, recipe, driver, tracers,
                                  binary_count, snapshot_file)
    scheme = _cyan(_advection_label(recipe.advection))
    return (
        @sprintf("%s", _bold(String(topology))),
        @sprintf("|-- grid:      %s, levels=%d, Hp=%d",
                 mesh_label, levels, halo_width),
        @sprintf("|-- numerics:  scheme=%s, FT=%s, backend=%s",
                 scheme, FT, backend),
        @sprintf("|-- physics:   diffusion=%s, convection=%s",
                 _diffusion_label(recipe.diffusion),
                 nameof(typeof(recipe.convection))),
        @sprintf("|-- schedule:  window_dt=%.0fs, steps/window=%s, binaries=%d",
                 Float64(window_dt(driver)), _schedule_label(driver),
                 binary_count),
        @sprintf("|-- tracers:   %s", join(String.(tracers), ", ")),
        @sprintf("`-- output:    %s", snapshot_file),
    )
end

function _log_runtime_summary(; topology, mesh_label, levels, halo_width,
                                backend, FT, recipe, driver, tracers,
                                binary_count, snapshot_file)
    lines = _physics_summary_lines(; topology, mesh_label, levels, halo_width,
                                   backend, FT, recipe, driver, tracers,
                                   binary_count, snapshot_file)
    @info "Driven runtime\n" * join(lines, "\n")
end
