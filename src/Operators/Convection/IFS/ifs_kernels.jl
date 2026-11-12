# ---------------------------------------------------------------------------
# IFS convection kernels: one work item per column, two launches per call.
#
#   _ifs_fluxes_*_kernel!    — interface fluxes and the column Courant number
#                              (`_ifs_column_fluxes!`), independent of tracers.
#   _ifs_transport_*_kernel! — all tracers and sub-steps of the column
#                              (`_ifs_column!`).
#
# Topologies differ only in how a column is addressed:
#   LatLon          (Nx, Ny) columns, area per latitude row;
#   ReducedGaussian (ncells,) columns, area per cell;
#   CubedSphere     (Nc, Nc, 6) columns; state, air mass and scratch carry the
#                   panel halo `Hp`, forcing, areas and ledger carry do not.
# The column bodies are shared; scratch is indexed like the air mass, so
# neighbouring work items touch neighbouring memory at every level.
# ---------------------------------------------------------------------------

@kernel function _ifs_fluxes_ll_kernel!(courant, mu, md, @Const(air_mass),
                                        @Const(entu), @Const(detu), @Const(entd), @Const(detd),
                                        @Const(areas_y), dt)
    i, j = @index(Global, NTuple)
    @inbounds courant[i, j] = _ifs_column_fluxes!(
        view(mu, i, j, :), view(md, i, j, :), view(air_mass, i, j, :),
        view(entu, i, j, :), view(detu, i, j, :), view(entd, i, j, :), view(detd, i, j, :),
        areas_y[j], dt)
end

@kernel function _ifs_transport_ll_kernel!(q_raw, carry, @Const(air_mass),
                                           @Const(detu), @Const(detd), @Const(mu), @Const(md),
                                           cu, cd, @Const(courant), @Const(areas_y),
                                           dt, cfl_limit, variant)
    i, j = @index(Global, NTuple)
    @inbounds _ifs_column!(view(q_raw, i, j, :, :), view(carry, i, j, :), view(air_mass, i, j, :),
                           view(detu, i, j, :), view(detd, i, j, :),
                           view(mu, i, j, :), view(md, i, j, :), view(cu, i, j, :), view(cd, i, j, :),
                           areas_y[j], dt, courant[i, j], cfl_limit, variant)
end

@kernel function _ifs_fluxes_rg_kernel!(courant, mu, md, @Const(air_mass),
                                        @Const(entu), @Const(detu), @Const(entd), @Const(detd),
                                        @Const(areas), dt)
    c = @index(Global)
    @inbounds courant[c] = _ifs_column_fluxes!(
        view(mu, c, :), view(md, c, :), view(air_mass, c, :),
        view(entu, c, :), view(detu, c, :), view(entd, c, :), view(detd, c, :),
        areas[c], dt)
end

@kernel function _ifs_transport_rg_kernel!(q_raw, carry, @Const(air_mass),
                                           @Const(detu), @Const(detd), @Const(mu), @Const(md),
                                           cu, cd, @Const(courant), @Const(areas),
                                           dt, cfl_limit, variant)
    c = @index(Global)
    @inbounds _ifs_column!(view(q_raw, c, :, :), view(carry, c, :), view(air_mass, c, :),
                           view(detu, c, :), view(detd, c, :),
                           view(mu, c, :), view(md, c, :), view(cu, c, :), view(cd, c, :),
                           areas[c], dt, courant[c], cfl_limit, variant)
end

# Six panels per launch; the third index selects the panel (`select_panel`).
@kernel function _ifs_fluxes_cs_kernel!(courant, mu, md, @Const(air_mass),
                                        @Const(entu), @Const(detu), @Const(entd), @Const(detd),
                                        @Const(areas), Hp::Int, dt)
    i, j, p = @index(Global, NTuple)
    ii = i + Hp
    jj = j + Hp
    @inbounds select_panel(courant, p)[ii, jj] = _ifs_column_fluxes!(
        view(select_panel(mu, p), ii, jj, :), view(select_panel(md, p), ii, jj, :),
        view(select_panel(air_mass, p), ii, jj, :),
        view(select_panel(entu, p), i, j, :), view(select_panel(detu, p), i, j, :),
        view(select_panel(entd, p), i, j, :), view(select_panel(detd, p), i, j, :),
        select_panel(areas, p)[i, j], dt)
end

@kernel function _ifs_transport_cs_kernel!(q_raw, carry, @Const(air_mass),
                                           @Const(detu), @Const(detd), @Const(mu), @Const(md),
                                           cu, cd, @Const(courant), @Const(areas),
                                           Hp::Int, dt, cfl_limit, variant)
    i, j, p = @index(Global, NTuple)
    ii = i + Hp
    jj = j + Hp
    @inbounds _ifs_column!(view(select_panel(q_raw, p), ii, jj, :, :),
                           view(select_panel(carry, p), i, j, :),
                           view(select_panel(air_mass, p), ii, jj, :),
                           view(select_panel(detu, p), i, j, :), view(select_panel(detd, p), i, j, :),
                           view(select_panel(mu, p), ii, jj, :), view(select_panel(md, p), ii, jj, :),
                           view(select_panel(cu, p), ii, jj, :), view(select_panel(cd, p), ii, jj, :),
                           select_panel(areas, p)[i, j], dt,
                           select_panel(courant, p)[ii, jj], cfl_limit, variant)
end
