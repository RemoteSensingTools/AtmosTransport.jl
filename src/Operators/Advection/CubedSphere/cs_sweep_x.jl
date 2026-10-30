# Cubed-sphere X packed upwind kernel and directional ping-pong entrypoints.
# Split from CubedSphereStrang.jl (refactor phase 4); included by Advection.jl in this order.

@kernel function _cs_xsweep_mt_upwind_kernel!(rm_new_4d, @Const(rm_4d),
                                               m_new, @Const(m),
                                               @Const(am), Nc, Hp, Nt, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        am_l = flux_scale * am[i,     j, k]
        am_r = flux_scale * am[i + 1, j, k]

        mi   = m[i,     j, k]
        mim1 = m[i - 1, j, k]
        mip1 = m[i + 1, j, k]
        m_new[i, j, k] = mi + am_l - am_r
        for t in Int32(1):Int32(Nt)
            ri   = rm_4d[i,     j, k, t]
            rim1 = rm_4d[i - 1, j, k, t]
            rip1 = rm_4d[i + 1, j, k, t]
            fl = ifelse(am_l >= zero(am_l),
                        _gamma_clamped_x_flux(am_l, mim1, rim1),
                        _gamma_clamped_x_flux(am_l, mi,   ri))
            fr = ifelse(am_r >= zero(am_r),
                        _gamma_clamped_x_flux(am_r, mi,   ri),
                        _gamma_clamped_x_flux(am_r, mip1, rip1))
            rm_new_4d[i, j, k, t] = ri + fl - fr
        end
    end
end

# Keep the directional entry point for callers and backend launch checks.
function _sweep_x_panel_mt_pingpong!(rm_out, m_out, rm, m, flux,
                                         scheme::AbstractAdvectionScheme,
                                         Nc, Hp, Nz, Nt;
                                         flux_scale = one(eltype(m)))
    _sweep_cs_panel_mt_pingpong!(rm_out, m_out, rm, m, flux, scheme,
                                 Nc, Hp, Nz, Nt, Val(1); flux_scale)
end

function _sweep_x_panels_mt_pingpong!(rm_out::NTuple{6}, m_out::NTuple{6},
                                          rm::NTuple{6}, m::NTuple{6}, flux::NTuple{6},
                                          mesh::CubedSphereMesh, scheme::AbstractAdvectionScheme;
                                          flux_scale = one(eltype(m[1])), seam_flux = nothing)
    _sweep_cs_horizontal_pingpong!(rm_out, m_out, rm, m, flux, mesh, scheme,
                                    Val(1); flux_scale, seam_flux)
end
