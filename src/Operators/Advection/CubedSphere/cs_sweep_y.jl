# Cubed-sphere Y packed upwind kernel and directional ping-pong entrypoints.
# Split from CubedSphereStrang.jl (refactor phase 4); included by Advection.jl in this order.

@kernel function _cs_ysweep_mt_upwind_kernel!(rm_new_4d, @Const(rm_4d),
                                               m_new, @Const(m),
                                               @Const(bm), Nc, Hp, Nt, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bm_s = flux_scale * bm[i, j,     k]
        bm_n = flux_scale * bm[i, j + 1, k]

        mi   = m[i, j,     k]
        mjm1 = m[i, j - 1, k]
        mjp1 = m[i, j + 1, k]
        m_new[i, j, k] = mi + bm_s - bm_n
        for t in Int32(1):Int32(Nt)
            ri   = rm_4d[i, j,     k, t]
            rjm1 = rm_4d[i, j - 1, k, t]
            rjp1 = rm_4d[i, j + 1, k, t]
            fs = ifelse(bm_s >= zero(bm_s),
                        _gamma_clamped_x_flux(bm_s, mjm1, rjm1),
                        _gamma_clamped_x_flux(bm_s, mi,   ri))
            fn = ifelse(bm_n >= zero(bm_n),
                        _gamma_clamped_x_flux(bm_n, mi,   ri),
                        _gamma_clamped_x_flux(bm_n, mjp1, rjp1))
            rm_new_4d[i, j, k, t] = ri + fs - fn
        end
    end
end

# Keep the directional entry point for callers and backend launch checks.
function _sweep_y_panel_mt_pingpong!(rm_out, m_out, rm, m, flux,
                                         scheme::AbstractAdvectionScheme,
                                         Nc, Hp, Nz, Nt;
                                         flux_scale = one(eltype(m)))
    _sweep_cs_panel_mt_pingpong!(rm_out, m_out, rm, m, flux, scheme,
                                 Nc, Hp, Nz, Nt, Val(2); flux_scale)
end

function _sweep_y_panels_mt_pingpong!(rm_out::NTuple{6}, m_out::NTuple{6},
                                          rm::NTuple{6}, m::NTuple{6}, flux::NTuple{6},
                                          mesh::CubedSphereMesh, scheme::AbstractAdvectionScheme;
                                          flux_scale = one(eltype(m[1])), seam_flux = nothing)
    _sweep_cs_horizontal_pingpong!(rm_out, m_out, rm, m, flux, mesh, scheme,
                                    Val(2); flux_scale, seam_flux)
end
