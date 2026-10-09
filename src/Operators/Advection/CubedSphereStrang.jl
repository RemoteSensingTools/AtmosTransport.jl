# ---------------------------------------------------------------------------
# Cubed-sphere Strang splitting orchestrator for src
#
# Performs X → Y → Z → Z → Y → X dimensionally-split advection on 6
# gnomonic panels with halo exchange between horizontal sweeps. The panel
# sweeps are in cs_sweep_{common,x,y,z}.jl, the workspace in cs_workspace.jl
# and the subcycle count in cs_subcycling.jl.
#
# Panel-interior kernels reuse the SAME reconstruction functions
# (_xface_tracer_flux, _yface_tracer_flux, _zface_tracer_flux) as the
# LatLon path. The only CS-specific logic is:
#   1. Halo exchange after each horizontal sweep (fill_panel_halos!)
#   2. Kernel launch on interior indices with Hp offset
#   3. Per-panel loop over 6 panels
#   4. Paired physical-seam transfers within each horizontal group
#
# The panel arrays have layout (Nc+2Hp, Nc+2Hp, Nz) with interior at
# [Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]. The reconstruction stencil reads into
# the halo region naturally.
#
# Each global X/Y group includes its panel-interior faces plus physical seams
# owned by the lower-numbered panel's corresponding local axis. CubedSphereSeams
# caches those transfers before any panel update and applies them with opposite
# signs to both neighbors, including rotated X/Y contacts. Each group therefore
# conserves air and tracer mass for mirrored input fluxes. Vertical sweeps are
# panel-local and closed. Temporal accuracy at seams needs separate validation.
#
# References:
#   Strang (1968) — symmetric splitting for second-order accuracy
#   Putman & Lin (2007) — FV3 cubed-sphere transport
# ---------------------------------------------------------------------------

# =========================================================================
# Public API: strang_split_cs!
# =========================================================================

"""
    strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                     mesh, scheme, workspace; flux_scale=1, cfl_limit=0.95)

Perform one Strang-split advection step on a 6-panel cubed-sphere field
with automatic CFL-based subcycling per direction.

## Splitting sequence

    X sweep (n_x subcycles)
    → fill_panel_halos!(dir=1)     ← exchange halos between panels (X direction)
    → Y sweep (n_y subcycles)
    → fill_panel_halos!(dir=2)     ← exchange halos between panels (Y direction)
    → Z sweep (n_z subcycles)      ← first Z half-step
    → Z sweep (n_z subcycles)      ← second Z half-step (palindrome)
    → fill_panel_halos!(dir=2)
    → Y sweep (n_y subcycles)
    → fill_panel_halos!(dir=1)
    → X sweep (n_x subcycles)

This sequence follows Strang's X-Y-Z-Z-Y-X ordering. Second-order splitting
requires sufficiently accurate subflows; the palindrome alone does not prove
the transport's temporal order. Horizontal groups pair physical seam transfers
before halo exchange supplies neighboring values for the next reconstruction.

## Subcycling

All six legs use one count from the initial-mass palindrome budget
`2 * (out_x + out_y + out_z) / m_start`, with flux divided by that count.
A supplied `subcycle_count` uses the binary's schedule; setting
`ATMOSTR_ASSERT_CS_BINARY_CFL=1` checks it against the runtime budget.

## Panel array layout

Each panel's rm and m arrays are `(Nc+2Hp, Nc+2Hp, Nz)` with Hp-wide halos.
Interior cells are at indices `[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]`. The sweep
kernels only update interior cells; halo regions are filled by
`fill_panel_halos!` from adjacent panels.

## Arguments

- `panels_rm`, `panels_m`: NTuple{6} of 3D arrays `(Nc+2Hp, Nc+2Hp, Nz)` —
  tracer mass and air mass. Modified in-place.
- `panels_am`, `panels_bm`, `panels_cm`: NTuple{6} of flux arrays.
  `am[Nc+2Hp+1, Nc+2Hp, Nz]`, `bm[Nc+2Hp, Nc+2Hp+1, Nz]`,
  `cm[Nc+2Hp, Nc+2Hp, Nz+1]`. Read-only.
- `mesh`: `CubedSphereMesh` with Nc, Hp, and panel connectivity.
- `scheme`: advection scheme — `UpwindScheme()` uses gamma-clamped upwind
  for tracer face transfers. `SlopesScheme()` and `PPMScheme()` use the generic
  KA kernels with `_xface_tracer_flux` dispatch. Higher-order schemes
  require `mesh.Hp ≥ 2` (Slopes) or `mesh.Hp ≥ 3` (PPM).
- `workspace`: pre-allocated `CSAdvectionWorkspace` buffers.
- `flux_scale`: overall scaling applied to all fluxes (default 1.0).
- `cfl_limit`: maximum CFL per subcycle pass (default 0.95).
"""
function strang_split_cs!(panels_rm::NTuple{6},
                          panels_m::NTuple{6},
                          panels_am::NTuple{6},
                          panels_bm::NTuple{6},
                          panels_cm::NTuple{6},
                          mesh::CubedSphereMesh,
                          scheme,
                          workspace::CSAdvectionWorkspace;
                          flux_scale = one(eltype(panels_m[1])),
                          cfl_limit::Real = 0.95,
                          subcycle_count::Union{Nothing, Integer} = nothing,
                          midpoint! = nothing)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_rm[1], 3)
    FT = eltype(panels_m[1])
    fs = convert(FT, flux_scale)
    cfl_ft = convert(FT, cfl_limit)

    n_pal = if subcycle_count === nothing
        # Budget all six legs against initial carrier mass. Face-local tracer
        # clamping does not replace a safe total-outflow budget.
        SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
            panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
            flux_scale = fs)
    else
        n = Int(subcycle_count)
        n >= 1 || throw(ArgumentError("strang_split_cs!: subcycle_count must be ≥ 1, got $(subcycle_count)"))
        if get(ENV, "ATMOSTR_ASSERT_CS_BINARY_CFL", "0") == "1"
            required = SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
                panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
                flux_scale = fs)
            required <= n || throw(ArgumentError(
                "strang_split_cs!: binary substep contract requested " *
                "subcycle_count=$n, but runtime CFL assertion requires " *
                "$required. Regenerate the binary or disable " *
                "ATMOSTR_ASSERT_CS_BINARY_CFL for diagnostic runs."))
        end
        n
    end
    n_x = n_pal
    n_y = n_pal
    n_z = n_pal

    _record_cs_subcycle_growth!(workspace, n_x, n_y, n_z)

    fs_x = fs / FT(n_x)
    fs_y = fs / FT(n_y)
    fs_z = fs / FT(n_z)

    # ---- X sweep (subcycled) ----
    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_cs_horizontal!(panels_rm, panels_m, panels_am, mesh,
                              scheme, workspace, Val(1); flux_scale=fs_x)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm, mesh; dir=1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,  mesh; dir=1)
    end

    # ---- Y sweep (subcycled) ----
    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_cs_horizontal!(panels_rm, panels_m, panels_bm, mesh,
                              scheme, workspace, Val(2); flux_scale=fs_y)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm, mesh; dir=2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,  mesh; dir=2)
    end

    # ---- Z sweep × 2 (subcycled) ----
    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        _sweep_z_panels!(panels_rm, panels_m, panels_cm, mesh, scheme, workspace;
                         flux_scale = fs_z)
    end

    midpoint! === nothing || midpoint!()

    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        _sweep_z_panels!(panels_rm, panels_m, panels_cm, mesh, scheme, workspace;
                         flux_scale = fs_z)
    end

    # ---- Reverse: Y sweep (subcycled) ----
    SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm, mesh; dir=2)
    SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,  mesh; dir=2)
    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_cs_horizontal!(panels_rm, panels_m, panels_bm, mesh,
                              scheme, workspace, Val(2); flux_scale=fs_y)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm, mesh; dir=2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,  mesh; dir=2)
    end

    # ---- Reverse: X sweep (subcycled) ----
    SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm, mesh; dir=1)
    SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,  mesh; dir=1)
    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_cs_horizontal!(panels_rm, panels_m, panels_am, mesh,
                              scheme, workspace, Val(1); flux_scale=fs_x)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm, mesh; dir=1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,  mesh; dir=1)
    end

    return nothing
end

@inline function _check_cs_packed_workspace(workspace::CSAdvectionWorkspace, Nt::Int)
    size(workspace.rm_4d_A, 4) >= Nt || throw(ArgumentError(
        "CSAdvectionWorkspace was built for $(size(workspace.rm_4d_A, 4)) packed tracers, " *
        "but the state has $Nt. Rebuild the workspace with `n_tracers = ntracers(state)` " *
        "or construct the `TransportModel` without overriding `workspace`."))
    size(workspace.rm_4d_pp_buf[1], 4) >= Nt || throw(ArgumentError(
        "CSAdvectionWorkspace ping-pong buffers were built for " *
        "$(size(workspace.rm_4d_pp_buf[1], 4)) packed tracers, but the state has $Nt. " *
        "Rebuild the workspace with `n_tracers = ntracers(state)`."))
    size(workspace.m_pp_buf[1], 3) == size(workspace.m_A, 3) || throw(ArgumentError(
        "CSAdvectionWorkspace mass ping-pong buffers are not allocated for packed transport. " *
        "Rebuild the workspace with `n_tracers = ntracers(state)`."))
    return nothing
end

"""
    strang_split_cs_mt!(panels_rm_4d, panels_m, panels_am, panels_bm, panels_cm,
                        mesh, scheme, workspace; ...)

Packed-tracer cubed-sphere split-sweep transport. This is the production CS
path for `CSSplitSweepStyle` schemes: air mass is advanced once per sweep and
all tracers in each panel's fourth dimension are updated inside the same panel
kernel. The sequence and CFL contract match [`strang_split_cs!`](@ref).
"""
function _strang_split_cs_mt_copyback!(panels_rm_4d::NTuple{6},
                                       panels_m::NTuple{6},
                                       panels_am::NTuple{6},
                                       panels_bm::NTuple{6},
                                       panels_cm::NTuple{6},
                                       mesh::CubedSphereMesh,
                                       scheme::AbstractAdvectionScheme,
                                       workspace::CSAdvectionWorkspace;
                                       flux_scale = one(eltype(panels_m[1])),
                                       cfl_limit::Real = 0.95,
                                       subcycle_count::Union{Nothing, Integer} = nothing,
                                       midpoint! = nothing)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    Nt = size(panels_rm_4d[1], 4)
    _check_cs_packed_workspace(workspace, Nt)
    rm_4d_A, m_A = workspace.rm_4d_A, workspace.m_A
    FT = eltype(panels_m[1])
    fs = convert(FT, flux_scale)
    cfl_ft = convert(FT, cfl_limit)

    n_pal = if subcycle_count === nothing
        SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
            panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
            flux_scale = fs)
    else
        n = Int(subcycle_count)
        n >= 1 || throw(ArgumentError("strang_split_cs_mt!: subcycle_count must be ≥ 1, got $(subcycle_count)"))
        if get(ENV, "ATMOSTR_ASSERT_CS_BINARY_CFL", "0") == "1"
            required = SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
                panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
                flux_scale = fs)
            required <= n || throw(ArgumentError(
                "strang_split_cs_mt!: binary substep contract requested " *
                "subcycle_count=$n, but runtime CFL assertion requires $required."))
        end
        n
    end
    n_x = n_pal
    n_y = n_pal
    n_z = n_pal

    _record_cs_subcycle_growth!(workspace, n_x, n_y, n_z)

    fs_x = fs / FT(n_x)
    fs_y = fs / FT(n_y)
    fs_z = fs / FT(n_z)

    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_cs_horizontal!(panels_rm_4d, panels_m, panels_am, mesh,
                              scheme, workspace, Val(1); flux_scale=fs_x)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm_4d, mesh; dir = 1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,     mesh; dir = 1)
    end

    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_cs_horizontal!(panels_rm_4d, panels_m, panels_bm, mesh,
                              scheme, workspace, Val(2); flux_scale=fs_y)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm_4d, mesh; dir = 2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,     mesh; dir = 2)
    end

    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        for p in 1:6
            _sweep_z_panel_mt!(panels_rm_4d[p], panels_m[p], panels_cm[p],
                               scheme, rm_4d_A, m_A, Nc, Hp, Nz, Nt; flux_scale = fs_z)
        end
    end

    midpoint! === nothing || midpoint!()

    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        for p in 1:6
            _sweep_z_panel_mt!(panels_rm_4d[p], panels_m[p], panels_cm[p],
                               scheme, rm_4d_A, m_A, Nc, Hp, Nz, Nt; flux_scale = fs_z)
        end
    end

    SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm_4d, mesh; dir = 2)
    SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,     mesh; dir = 2)
    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_cs_horizontal!(panels_rm_4d, panels_m, panels_bm, mesh,
                              scheme, workspace, Val(2); flux_scale=fs_y)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm_4d, mesh; dir = 2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,     mesh; dir = 2)
    end

    SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm_4d, mesh; dir = 1)
    SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,     mesh; dir = 1)
    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_cs_horizontal!(panels_rm_4d, panels_m, panels_am, mesh,
                              scheme, workspace, Val(1); flux_scale=fs_x)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm_4d, mesh; dir = 1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,     mesh; dir = 1)
    end

    return nothing
end

function strang_split_cs_mt!(panels_rm_4d::NTuple{6},
                             panels_m::NTuple{6},
                             panels_am::NTuple{6},
                             panels_bm::NTuple{6},
                             panels_cm::NTuple{6},
                             mesh::CubedSphereMesh,
                             scheme::AbstractAdvectionScheme,
                             workspace::CSAdvectionWorkspace;
                             flux_scale = one(eltype(panels_m[1])),
                             cfl_limit::Real = 0.95,
                             subcycle_count::Union{Nothing, Integer} = nothing,
                             midpoint! = nothing)
    strang_split_cs_mt_pingpong!(panels_rm_4d, panels_m,
                                 workspace.rm_4d_pp_buf, workspace.m_pp_buf,
                                 panels_am, panels_bm, panels_cm, mesh, scheme,
                                 workspace; flux_scale, cfl_limit,
                                 subcycle_count, midpoint!)
    return nothing
end

"""
    strang_split_cs_mt_pingpong!(panels_rm_4d, panels_m, panels_rm_4d_buf, panels_m_buf,
                                 panels_am, panels_bm, panels_cm, mesh, scheme, workspace; ...)

Packed-tracer CS split-sweep that writes each sweep directly into alternate
panel buffers and swaps active/inactive tuples between sweeps. This removes the
per-sweep copy-back kernels while keeping the existing KA sweep kernels. The
final active `(rm, m)` tuple is returned.
"""
function strang_split_cs_mt_pingpong!(panels_rm_4d::NTuple{6},
                                      panels_m::NTuple{6},
                                      panels_rm_4d_buf::NTuple{6},
                                      panels_m_buf::NTuple{6},
                                      panels_am::NTuple{6},
                                      panels_bm::NTuple{6},
                                      panels_cm::NTuple{6},
                                      mesh::CubedSphereMesh,
                                      scheme::AbstractAdvectionScheme,
                                      workspace::CSAdvectionWorkspace;
                                      flux_scale = one(eltype(panels_m[1])),
                                      cfl_limit::Real = 0.95,
                                      subcycle_count::Union{Nothing, Integer} = nothing,
                                      midpoint! = nothing)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    Nt = size(panels_rm_4d[1], 4)
    _check_cs_packed_workspace(workspace, Nt)
    FT = eltype(panels_m[1])
    fs = convert(FT, flux_scale)
    cfl_ft = convert(FT, cfl_limit)

    n_pal = if subcycle_count === nothing
        SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
            panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
            flux_scale = fs)
    else
        n = Int(subcycle_count)
        n >= 1 || throw(ArgumentError("strang_split_cs_mt_pingpong!: subcycle_count must be ≥ 1, got $(subcycle_count)"))
        if get(ENV, "ATMOSTR_ASSERT_CS_BINARY_CFL", "0") == "1"
            required = SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
                panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
                flux_scale = fs)
            required <= n || throw(ArgumentError(
                "strang_split_cs_mt_pingpong!: binary substep contract requested " *
                "subcycle_count=$n, but runtime CFL assertion requires $required."))
        end
        n
    end
    n_x = n_pal
    n_y = n_pal
    n_z = n_pal

    _record_cs_subcycle_growth!(workspace, n_x, n_y, n_z)

    fs_x = fs / FT(n_x)
    fs_y = fs / FT(n_y)
    fs_z = fs / FT(n_z)

    active_rm = panels_rm_4d
    active_m = panels_m
    spare_rm = panels_rm_4d_buf
    spare_m = panels_m_buf

    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_x_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_am, mesh, scheme; flux_scale = fs_x,
                                     seam_flux = workspace.seam_flux)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(spare_rm, mesh; dir = 1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(spare_m,  mesh; dir = 1)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_y_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_bm, mesh, scheme; flux_scale = fs_y,
                                     seam_flux = workspace.seam_flux)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(spare_rm, mesh; dir = 2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(spare_m,  mesh; dir = 2)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        _sweep_z_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_cm, mesh, scheme, workspace;
                                     flux_scale = fs_z)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    # The midpoint operators (diffusion / surface flux) must act on the CURRENT
    # active ping-pong buffer, not `state.tracers_raw` (which is stale mid-
    # palindrome). A 0-argument `midpoint!()` would silently mutate the wrong
    # array, so require the buffer-aware 2-arg form and fail loudly otherwise.
    if midpoint! !== nothing
        applicable(midpoint!, active_rm, active_m) || throw(ArgumentError(
            "strang_split_cs_mt_pingpong! requires a buffer-aware `midpoint!` " *
            "accepting (active_rm, active_m); a 0-argument midpoint! would mutate " *
            "the wrong (stale) buffer mid-palindrome."))
        midpoint!(active_rm, active_m)
    end

    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        _sweep_z_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_cm, mesh, scheme, workspace;
                                     flux_scale = fs_z)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(active_rm, mesh; dir = 2)
    SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(active_m,  mesh; dir = 2)
    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_y_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_bm, mesh, scheme; flux_scale = fs_y,
                                     seam_flux = workspace.seam_flux)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(spare_rm, mesh; dir = 2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(spare_m,  mesh; dir = 2)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(active_rm, mesh; dir = 1)
    SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(active_m,  mesh; dir = 1)
    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_x_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_am, mesh, scheme; flux_scale = fs_x,
                                     seam_flux = workspace.seam_flux)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(spare_rm, mesh; dir = 1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(spare_m,  mesh; dir = 1)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    return active_rm, active_m
end

"""Packed-tracer Z-sweep of all six panels into the ping-pong buffers."""
function _sweep_z_panels_mt_pingpong!(panels_rm_4d_out::NTuple{6},
                                      panels_m_out::NTuple{6},
                                      panels_rm_4d::NTuple{6},
                                      panels_m::NTuple{6},
                                      panels_cm::NTuple{6},
                                      mesh::CubedSphereMesh,
                                      scheme::AbstractAdvectionScheme,
                                      ::CSAdvectionWorkspace;
                                      flux_scale = one(eltype(panels_m[1])))
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    Nt = size(panels_rm_4d[1], 4)
    for p in 1:6
        _sweep_z_panel_mt_pingpong!(panels_rm_4d_out[p], panels_m_out[p],
                                   panels_rm_4d[p], panels_m[p], panels_cm[p],
                                   scheme, Nc, Hp, Nz, Nt; flux_scale)
    end
    return panels_rm_4d_out, panels_m_out
end

"""Single-tracer Z-sweep of all six panels (the split-sweep `strang_split_cs!` path)."""
function _sweep_z_panels!(panels_rm::NTuple{6}, panels_m::NTuple{6}, panels_cm::NTuple{6},
                          mesh::CubedSphereMesh, scheme::AbstractAdvectionScheme,
                          workspace::CSAdvectionWorkspace;
                          flux_scale = one(eltype(panels_m[1])))
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_rm[1], 3)
    for p in 1:6
        _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p], scheme,
                        workspace.rm_A, workspace.m_A, Nc, Hp, Nz; flux_scale)
    end
    return nothing
end

"""
    _sweep_z!(rm_panels, m_panels, cm_panels, mesh, ws)

Upwind Z-sweep of all six panels (`_sweep_z_panel!` with `UpwindScheme()`), used
by the Lin-Rood adjoint tape and `strang_split_linrood_ppm!`. FV3 itself remaps
tracers vertically with PPM; see `LinRoodPPMScheme`'s `vertical` option.
"""
function _sweep_z!(rm_panels, m_panels, cm_panels,
                   mesh::CubedSphereMesh, ws::CSAdvectionWorkspace)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(rm_panels[1], 3)
    for p in 1:6
        _sweep_z_panel!(rm_panels[p], m_panels[p], cm_panels[p],
                         UpwindScheme(), ws.rm_A, ws.m_A, Nc, Hp, Nz)
    end
    return nothing
end

export strang_split_cs!, strang_split_cs_mt!, CSAdvectionWorkspace
