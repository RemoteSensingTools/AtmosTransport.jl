# Directional sweeps: structured x/y/z (generated with @eval) and face-indexed horizontal/vertical.
# Split from StrangSplitting.jl (refactor phase 4); included by Advection.jl in this order.

# =========================================================================
# Generic directional sweeps — generated via @eval
# =========================================================================
#
# Ping-pong contract:
#   - The 7-arg form `sweep_x!(rm_in, rm_out, m_in, m_out, flux, scheme, ws)`
#     is the KERNEL entry point. It reads from (rm_in, m_in), writes to
#     (rm_out, m_out), synchronizes the backend, and returns. NO copyto!.
#     `strang_split!` uses this form and tracks parity so the output of
#     one sweep becomes the input of the next.
#   - The 5-arg form `sweep_x!(rm, m, flux, scheme, ws)` is the in-place
#     entry point. It kernels into `(ws.rm_B, ws.m_B)` and then copies back
#     to `(rm, m)`.
#   - The 6-arg / 8-arg `flux_scale` variants follow the same pattern.
#
# The three sweeps differ ONLY in:
#   - Which kernel function to call (_xsweep_kernel!, etc.)
#   - Which dimension to use for N (1=Nx, 2=Ny, 3=Nz)

for (sweep_fn, kernel_fn, dim) in (
    (:sweep_x!, :_xsweep_kernel!, 1),
    (:sweep_y!, :_ysweep_kernel!, 2),
    (:sweep_z!, :_zsweep_kernel!, 3),
)
    @eval begin
        # Ping-pong entry point: writes (rm_out, m_out), reads (rm_in, m_in).
        function $sweep_fn(rm_in::AbstractArray{FT,3},  rm_out::AbstractArray{FT,3},
                            m_in::AbstractArray{FT,3},   m_out::AbstractArray{FT,3},
                            flux::AbstractArray{FT,3},
                            scheme::AbstractAdvectionScheme,
                            ws::AdvectionWorkspace{FT}) where FT
            backend = get_backend(m_in)
            kernel! = $kernel_fn(backend, 256)
            kernel!(rm_out, rm_in, m_out, m_in, flux, scheme,
                    Int32(size(m_in, $dim)), one(FT);
                    ndrange=size(m_in))
            synchronize(backend)
            return nothing
        end

        """
            $($sweep_fn)(rm, m, flux, scheme, ws)

        In-place sweep. Writes into the workspace B pair, then copies the
        result back to `rm` and `m`. Palindrome orchestration should use the
        7-argument ping-pong form to avoid the copies.
        """
        function $sweep_fn(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                           flux::AbstractArray{FT,3},
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT}) where FT
            $sweep_fn(rm, ws.rm_B, m, ws.m_B, flux, scheme, ws)
            copyto!(rm, ws.rm_B)
            copyto!(m,  ws.m_B)
            return nothing
        end
    end
end

"""
    _horizontal_face_atomic_kernel!(rm_new, rm, m_new, m, horizontal_flux,
                                     face_left, face_right, scheme, flux_scale)

Face-indexed horizontal advection kernel for **unstructured meshes** (e.g.
reduced-Gaussian). Each work item processes one face `f` at level `k`.

The face topology is defined by `face_left[f]` and `face_right[f]`:
- Both > 0: interior face connecting two cells. Flux is accumulated via
  `@atomic` additions to rm_new/m_new at both left and right cells
  (race-safe for GPU workgroups that share cells across faces).
- `face_left[f] == 0`: south pole boundary stub (only right cell exists).
  **Skipped** — no mass enters or leaves through the pole singularity.
- `face_right[f] == 0`: north pole boundary stub. **Skipped**.

Flux sign convention: positive `horizontal_flux[f, k]` = mass moving from
`left` to `right`. The tracer flux `_hface_tracer_flux` uses the donor
cell's mixing ratio (upwind) or higher-order reconstruction.
"""
@kernel function _horizontal_face_atomic_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                                 @Const(horizontal_flux),
                                                 @Const(face_left), @Const(face_right),
                                                 scheme, flux_scale)
    f, k = @index(Global, NTuple)
    @inbounds begin
        left = Int(face_left[f])
        right = Int(face_right[f])
        # Skip boundary stubs: left=0 (south pole) or right=0 (north pole)
        if left > 0 && right > 0
            flux = flux_scale * horizontal_flux[f, k]
            tracer_flux = _hface_tracer_flux(rm, m, flux, left, right, k, scheme)
            # Atomic accumulation: multiple faces share the same cell
            @atomic rm_new[left,  k] += -tracer_flux   # tracer leaving left cell
            @atomic rm_new[right, k] +=  tracer_flux   # tracer entering right cell
            @atomic m_new[left,   k] += -flux           # mass leaving left cell
            @atomic m_new[right,  k] +=  flux           # mass entering right cell
        end
    end
end

"""
    _vertical_face_kernel!(rm_new, rm, m_new, m, cm, scheme, flux_scale, Nz)

Vertical advection kernel for face-indexed meshes. Each work item processes
one cell `c` at level `k`.

Vertical boundaries: `k=1` is TOA (no flux above), `k=Nz` is deepest level
(no flux below). `cm[c, k]` is the vertical mass flux through the TOP face
of cell `(c, k)`. Positive cm = downward (toward surface).
"""
@kernel function _vertical_face_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                        @Const(cm), scheme, flux_scale, Nz)
    c, k = @index(Global, NTuple)
    FT = eltype(rm)
    @inbounds begin
        # NOTE: `?:` (branch) is used here intentionally, NOT `ifelse`.
        # `ifelse` evaluates BOTH branches, but `_vface_tracer_flux` reads
        # `rm[c, k±1]` which is out-of-bounds at the boundaries:
        #   k=1:  k-1 = 0     → rm[c, 0] is OOB
        #   k=Nz: k+1 = Nz+1 → rm[c, Nz+1] is OOB (rm is nc×Nz, not nc×Nz+1)
        # The `?:` branch avoids evaluating the OOB branch entirely.
        # Warp divergence is not a concern because `k` is typically constant
        # within a warp (KA maps (c, k) with c as the fast dimension).
        # k=1: TOA boundary → flux_t = 0 (no flux above top level)
        flux_t = k > 1  ? _vface_tracer_flux(rm, m, flux_scale * cm[c, k],     c, k - 1, k,     scheme) : zero(FT)
        # k=Nz: surface boundary → flux_b = 0 (no flux below bottom level)
        flux_b = k < Nz ? _vface_tracer_flux(rm, m, flux_scale * cm[c, k + 1], c, k,     k + 1, scheme) : zero(FT)
        rm_new[c, k] = rm[c, k] + flux_t - flux_b
        m_new[c, k]  = m[c, k]  + flux_scale * cm[c, k] - flux_scale * cm[c, k + 1]
    end
end

function _sweep_horizontal_face_gpu!(backend, rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                     horizontal_flux::AbstractArray{FT,2},
                                     scheme::UpwindScheme,
                                     ws::AdvectionWorkspace{FT},
                                     flux_scale::FT) where FT
    isempty(ws.face_left) &&
        throw(ArgumentError("face-indexed GPU sweep requires mesh connectivity in AdvectionWorkspace"))
    copyto!(ws.rm_A, rm)
    copyto!(ws.m_A, m)
    kernel! = _horizontal_face_atomic_kernel!(backend, 256)
    kernel!(ws.rm_A, rm, ws.m_A, m, horizontal_flux, ws.face_left, ws.face_right, scheme, flux_scale;
            ndrange=size(horizontal_flux))
    synchronize(backend)
    copyto!(rm, ws.rm_A)
    copyto!(m, ws.m_A)
    return nothing
end

function _sweep_vertical_face_gpu!(backend, rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                   cm::AbstractArray{FT,2},
                                   scheme::UpwindScheme,
                                   ws::AdvectionWorkspace{FT},
                                   flux_scale::FT) where FT
    kernel! = _vertical_face_kernel!(backend, 256)
    kernel!(ws.rm_A, rm, ws.m_A, m, cm, scheme, flux_scale, Int32(size(m, 2));
            ndrange=size(m))
    synchronize(backend)
    copyto!(rm, ws.rm_A)
    copyto!(m, ws.m_A)
    return nothing
end

function _sweep_horizontal_face_backend!(::KA_CPU,
                                         rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                         horizontal_flux::AbstractArray{FT,2},
                                         mesh::AbstractHorizontalMesh,
                                         scheme::UpwindScheme,
                                         ws::AdvectionWorkspace{FT},
                                         flux_scale::FT) where FT
    _horizontal_face_tendency!(ws.rm_A, rm, ws.m_A, m, horizontal_flux, mesh, scheme, flux_scale)
    copyto!(rm, ws.rm_A)
    copyto!(m, ws.m_A)
    return nothing
end

function _sweep_horizontal_face_backend!(backend,
                                         rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                         horizontal_flux::AbstractArray{FT,2},
                                         mesh::AbstractHorizontalMesh,
                                         scheme::UpwindScheme,
                                         ws::AdvectionWorkspace{FT},
                                         flux_scale::FT) where FT
    _sweep_horizontal_face_gpu!(backend, rm, m, horizontal_flux, scheme, ws, flux_scale)
end

function _sweep_vertical_face_backend!(::KA_CPU,
                                       rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                       cm::AbstractArray{FT,2},
                                       scheme::UpwindScheme,
                                       ws::AdvectionWorkspace{FT},
                                       flux_scale::FT) where FT
    _vertical_column_tendency!(ws.rm_A, rm, ws.m_A, m, cm, scheme, flux_scale)
    copyto!(rm, ws.rm_A)
    copyto!(m, ws.m_A)
    return nothing
end

function _sweep_vertical_face_backend!(backend,
                                       rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                       cm::AbstractArray{FT,2},
                                       scheme::UpwindScheme,
                                       ws::AdvectionWorkspace{FT},
                                       flux_scale::FT) where FT
    _sweep_vertical_face_gpu!(backend, rm, m, cm, scheme, ws, flux_scale)
end

# Additional structured sweep overloads with explicit flux scaling: apply a
# fraction of the directional forcing, so a sweep can be split into smaller
# conservative pieces. Same ping-pong
# contract as the one(FT) variants above: the 8-arg form does the kernel
# launch only (no `copyto!`); the 6-argument form is the in-place entry point.
for (sweep_fn, kernel_fn, dim) in (
    (:sweep_x!, :_xsweep_kernel!, 1),
    (:sweep_y!, :_ysweep_kernel!, 2),
    (:sweep_z!, :_zsweep_kernel!, 3),
)
    @eval begin
        # Ping-pong entry point with explicit flux scale.
        function $sweep_fn(rm_in::AbstractArray{FT,3},  rm_out::AbstractArray{FT,3},
                            m_in::AbstractArray{FT,3},   m_out::AbstractArray{FT,3},
                            flux::AbstractArray{FT,3},
                            scheme::AbstractAdvectionScheme,
                            ws::AdvectionWorkspace{FT},
                            flux_scale::FT) where FT
            backend = get_backend(m_in)
            kernel! = $kernel_fn(backend, 256)
            kernel!(rm_out, rm_in, m_out, m_in, flux, scheme,
                    Int32(size(m_in, $dim)), flux_scale;
                    ndrange=size(m_in))
            synchronize(backend)
            return nothing
        end

        # In-place entry point with explicit flux scaling.
        function $sweep_fn(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                           flux::AbstractArray{FT,3},
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT},
                           flux_scale::FT) where FT
            $sweep_fn(rm, ws.rm_B, m, ws.m_B, flux, scheme, ws, flux_scale)
            copyto!(rm, ws.rm_B)
            copyto!(m,  ws.m_B)
            return nothing
        end
    end
end

function sweep_horizontal!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                           horizontal_flux::AbstractArray{FT,2},
                           mesh::AbstractHorizontalMesh,
                           scheme::UpwindScheme,
                           ws::AdvectionWorkspace{FT}) where FT
    return _sweep_horizontal_face_backend!(get_backend(rm), rm, m, horizontal_flux, mesh, scheme, ws, one(FT))
end

function sweep_horizontal!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                           horizontal_flux::AbstractArray{FT,2},
                           mesh::AbstractHorizontalMesh,
                           scheme::UpwindScheme,
                           ws::AdvectionWorkspace{FT},
                           flux_scale::FT) where FT
    return _sweep_horizontal_face_backend!(get_backend(rm), rm, m, horizontal_flux, mesh, scheme, ws, flux_scale)
end

function _throw_unsupported_face_indexed_scheme(op::Symbol, scheme::AbstractAdvectionScheme)
    throw(ArgumentError("$(op) on face-indexed meshes supports UpwindScheme only; got $(typeof(scheme)). Use structured-grid sweeps for SlopesScheme/PPMScheme or the cubed-sphere driver for LinRoodPPMScheme."))
end

function sweep_horizontal!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                           horizontal_flux::AbstractArray{FT,2},
                           mesh::AbstractHorizontalMesh,
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT}) where FT
    _throw_unsupported_face_indexed_scheme(:sweep_horizontal!, scheme)
end

function sweep_horizontal!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                           horizontal_flux::AbstractArray{FT,2},
                           mesh::AbstractHorizontalMesh,
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT},
                           flux_scale::FT) where FT
    _throw_unsupported_face_indexed_scheme(:sweep_horizontal!, scheme)
end

function sweep_vertical!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                         cm::AbstractArray{FT,2},
                         scheme::UpwindScheme,
                         ws::AdvectionWorkspace{FT}) where FT
    return _sweep_vertical_face_backend!(get_backend(rm), rm, m, cm, scheme, ws, one(FT))
end

function sweep_vertical!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                         cm::AbstractArray{FT,2},
                         scheme::UpwindScheme,
                         ws::AdvectionWorkspace{FT},
                         flux_scale::FT) where FT
    return _sweep_vertical_face_backend!(get_backend(rm), rm, m, cm, scheme, ws, flux_scale)
end

# =========================================================================
# Face-indexed tendency functions
# =========================================================================
#
# For unstructured grids (face-connected meshes like ReducedGaussian),
# advection operates on 2D arrays (cell, level) with face connectivity
# provided by the mesh object.
#
"""
    _horizontal_face_tendency!(rm_new, rm, m_new, m, horizontal_flux, mesh, scheme)

Compute horizontal advection tendency on a face-indexed mesh using
the new scheme dispatch.

Iterates over all faces in the mesh, computing the face tracer flux
via `_hface_tracer_flux` (dispatches on `scheme` type) and accumulating
the flux divergence into `rm_new` and `m_new`.

# Mass conservation
The flux through each face is added to one cell and subtracted from
the other, so ``\\sum_c r_{m,\\text{new}}[c,k] = \\sum_c r_m[c,k]``
holds exactly for each level `k`.
"""
@inline function _horizontal_face_tendency!(rm_new::AbstractArray{FT,2},
                                            rm::AbstractArray{FT,2},
                                            m_new::AbstractArray{FT,2},
                                            m::AbstractArray{FT,2},
                                            horizontal_flux::AbstractArray{FT,2},
                                            mesh::AbstractHorizontalMesh,
                                            scheme::AbstractAdvectionScheme,
                                            flux_scale::FT = one(FT)) where FT
    copyto!(rm_new, rm)
    copyto!(m_new, m)
    nface = nfaces(mesh)
    Nz = size(m, 2)

    @inbounds for k in 1:Nz
        for f in 1:nface
            left, right = face_cells(mesh, f)
            if left > 0 && right > 0
                flux = flux_scale * horizontal_flux[f, k]
                tracer_flux = _hface_tracer_flux(rm, m, flux, left, right, k, scheme)
                rm_new[left,  k] -= tracer_flux
                rm_new[right, k] += tracer_flux
                m_new[left,   k] -= flux
                m_new[right,  k] += flux
            end
        end
    end

    return nothing
end

"""
    _vertical_column_tendency!(rm_new, rm, m_new, m, cm, scheme)

Compute vertical advection tendency for face-indexed grids using
the new scheme dispatch.

Iterates level by level, computing the vertical face tracer flux via
`_vface_tracer_flux` (dispatches on `scheme` type).  Closed boundaries
at TOA (k=1) and surface (k=Nz) are enforced explicitly with branch guards.
"""
@inline function _vertical_column_tendency!(rm_new::AbstractArray{FT,2},
                                            rm::AbstractArray{FT,2},
                                            m_new::AbstractArray{FT,2},
                                            m::AbstractArray{FT,2},
                                            cm::AbstractArray{FT,2},
                                            scheme::AbstractAdvectionScheme,
                                            flux_scale::FT = one(FT)) where FT
    copyto!(rm_new, rm)
    copyto!(m_new, m)
    nc = size(m, 1)
    Nz = size(m, 2)

    @inbounds for k in 1:Nz
        for c in 1:nc
            flux_t = k > 1 ? _vface_tracer_flux(rm, m, flux_scale * cm[c, k], c, k - 1, k, scheme) : zero(FT)
            flux_b = k < Nz ? _vface_tracer_flux(rm, m, flux_scale * cm[c, k + 1], c, k, k + 1, scheme) : zero(FT)
            rm_new[c, k] = rm[c, k] + flux_t - flux_b
            m_new[c, k]  = m[c, k]  + flux_scale * cm[c, k] - flux_scale * cm[c, k + 1]
        end
    end

    return nothing
end

# =========================================================================
# Face-indexed sweep helpers — generated via @eval
# =========================================================================

for (scheme_type, h_args, v_args) in (
    (:AbstractConstantScheme, (:mesh, :scheme), (:scheme,)),
)
    @eval begin
        """
            sweep_horizontal!(rm, m, horizontal_flux, mesh, scheme::$($scheme_type), ws)

        Horizontal advection sweep for face-indexed grids.  Computes the
        face-flux tendency and copies the result from the workspace buffers
        back to `rm` and `m`.
        """
        function sweep_horizontal!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                         horizontal_flux::AbstractArray{FT,2},
                                         mesh::AbstractHorizontalMesh,
                                         scheme::$scheme_type,
                                         ws::AdvectionWorkspace{FT}) where FT
            _horizontal_face_tendency!(ws.rm_A, rm, ws.m_A, m, horizontal_flux, $(h_args...), one(FT))
            copyto!(rm, ws.rm_A)
            copyto!(m, ws.m_A)
            return nothing
        end

        function sweep_horizontal!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                         horizontal_flux::AbstractArray{FT,2},
                                         mesh::AbstractHorizontalMesh,
                                         scheme::$scheme_type,
                                         ws::AdvectionWorkspace{FT},
                                         flux_scale::FT) where FT
            _horizontal_face_tendency!(ws.rm_A, rm, ws.m_A, m, horizontal_flux, $(h_args...), flux_scale)
            copyto!(rm, ws.rm_A)
            copyto!(m, ws.m_A)
            return nothing
        end

        """
            sweep_vertical!(rm, m, cm, scheme::$($scheme_type), ws)

        Vertical advection sweep for face-indexed grids.  Computes the
        vertical face-flux tendency and copies results back.
        """
        function sweep_vertical!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                       cm::AbstractArray{FT,2},
                                       scheme::$scheme_type,
                                       ws::AdvectionWorkspace{FT}) where FT
            _vertical_column_tendency!(ws.rm_A, rm, ws.m_A, m, cm, $(v_args...), one(FT))
            copyto!(rm, ws.rm_A)
            copyto!(m, ws.m_A)
            return nothing
        end

        function sweep_vertical!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                       cm::AbstractArray{FT,2},
                                       scheme::$scheme_type,
                                       ws::AdvectionWorkspace{FT},
                                       flux_scale::FT) where FT
            _vertical_column_tendency!(ws.rm_A, rm, ws.m_A, m, cm, $(v_args...), flux_scale)
            copyto!(rm, ws.rm_A)
            copyto!(m, ws.m_A)
            return nothing
        end
    end
end
