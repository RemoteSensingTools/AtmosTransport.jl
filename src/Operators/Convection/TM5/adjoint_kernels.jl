# Transpose solves and CS wrapper of the production TM5 column operator.
function _tm5_solve_vector_transpose!(lambda_col, conv1, pivots, Nz::Integer;
                                      icltop_eff::Integer = 1)
    Nz == 0 && return nothing
    k_lo = max(Int(icltop_eff), 1)
    @inbounds begin
        # U' z = lambda, where U is stored in the upper triangle.
        for k in k_lo:Nz
            s = lambda_col[k]
            for j in k_lo:(k - 1)
                s -= conv1[j, k] * lambda_col[j]
            end
            lambda_col[k] = s / conv1[k, k]
        end
        # L' y = z, where L is unit diagonal and stored below the diagonal.
        for k in Nz:-1:k_lo
            s = lambda_col[k]
            for j in (k + 1):Nz
                s -= conv1[j, k] * lambda_col[j]
            end
            lambda_col[k] = s
        end
        # Forward solve applies pivots in ascending order; the transpose
        # applies the inverse permutation, so replay swaps in reverse.
        for k in Nz:-1:k_lo
            piv = pivots[k]
            if piv != k
                tmp = lambda_col[k]
                lambda_col[k] = lambda_col[piv]
                lambda_col[piv] = tmp
            end
        end
    end
    return nothing
end

# Unpivoted Hessenberg LU has bidiagonal L. U and the active-row convention
# remain unchanged; these helpers use the same ordering as the general solves.
function _tm5_solve_vector_transpose_bidiagonal!(q, A, Nz; icltop_eff=1)
    lo = max(Int(icltop_eff), 1)
    @inbounds begin
        for k in lo:Nz
            value = q[k]
            for j in lo:(k-1)
                value -= A[j,k] * q[j]
            end
            q[k] = value / A[k,k]
        end
        for k in (Nz-1):-1:lo
            q[k] -= A[k+1,k] * q[k+1]
        end
    end
    return nothing
end

function _tm5_solve_column_vector_adjoint!(lambda_col, m_col,
                                           entu_col, detu_col, entd_col, detd_col,
                                           conv1_buf, pivots_buf, cloud_dims, dt;
                                           cell_area = one(eltype(lambda_col)),
                                           f_buf = conv1_buf,
                                           amu_buf,
                                           amd_buf)
    FT = eltype(lambda_col)
    Nz = length(m_col)
    Nz == 0 && return nothing
    icltop, iclbas, icllfs = _tm5_diagnose_cloud_dims(detu_col, entd_col, Nz)
    cloud_dims[1] = icltop
    cloud_dims[2] = iclbas
    cloud_dims[3] = icllfs
    icltop > Nz && return nothing

    icltop_eff = _tm5_effective_cloud_top(Int(icltop), Int(icllfs))
    _tm5_build_conv1!(conv1_buf,
                      entu_col, detu_col, entd_col, detd_col, m_col,
                      icltop, icllfs, FT(dt), Nz;
                      cell_area = FT(cell_area),
                      f = f_buf, amu = amu_buf, amd = amd_buf)
    _tm5_factorize!(conv1_buf, pivots_buf, Nz, icllfs; icltop_eff = icltop_eff)
    if icllfs > Nz && _tm5_identity_pivots(pivots_buf, Nz, icltop_eff)
        _tm5_solve_vector_transpose_bidiagonal!(lambda_col, conv1_buf, Nz; icltop_eff)
    else
        _tm5_solve_vector_transpose!(lambda_col, conv1_buf, pivots_buf, Nz; icltop_eff)
    end
    return nothing
end

@kernel function _tm5_cs_panel_column_adjoint_kernel!(
    lambda_panel, @Const(air_mass_panel),
    @Const(entu_panel), @Const(detu_panel),
    @Const(entd_panel), @Const(detd_panel),
    @Const(cell_areas_panel),
    conv1_panel, pivots_panel, cloud_panel,
    f_panel, amu_panel, amd_panel,
    Hp::Int, tile_offset::Int, Nc::Int, dt)
    t = @index(Global)
    c_global = tile_offset + t
    c1 = ((c_global - 1) % Nc) + 1
    c2 = ((c_global - 1) ÷ Nc) + 1
    i = c1 + Hp
    j = c2 + Hp
    @inbounds begin
        lambda_col = @view lambda_panel[i, j, :]
        m_col = @view air_mass_panel[i, j, :]
        entu_col = @view entu_panel[c1, c2, :]
        detu_col = @view detu_panel[c1, c2, :]
        entd_col = @view entd_panel[c1, c2, :]
        detd_col = @view detd_panel[c1, c2, :]
        conv1_col = @view conv1_panel[:, :, t]
        pivots_col = @view pivots_panel[:, t]
        cloud_col = @view cloud_panel[:, t]
        f_col = @view f_panel[:, :, t]
        amu_col = @view amu_panel[:, t]
        amd_col = @view amd_panel[:, t]
        _tm5_solve_column_vector_adjoint!(
            lambda_col, m_col, entu_col, detu_col, entd_col, detd_col,
            conv1_col, pivots_col, cloud_col, dt;
            cell_area = cell_areas_panel[c1, c2],
            f_buf = f_col, amu_buf = amu_col, amd_buf = amd_col)
    end
end
