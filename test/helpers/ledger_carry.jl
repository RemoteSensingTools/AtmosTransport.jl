using AtmosTransport, Adapt, Test
const LedgerConv = AtmosTransport.Operators.Convection
const LedgerDiff = AtmosTransport.Operators.Diffusion

# Column mass ledger carry on a device (`array_type`, e.g. `CuArray` or
# `MtlArray`). Repeated calls keep, per column and tracer, the tracer mass
# plus the workspace's ledger carry (see `_ledger_deposit`) to rounding of the
# carry, and leave nonzero carries: they are loaded and stored on the device.
_ledger_sums(q) = sum(big, q; dims = ndims(q) - 1)
function _ledger_conserved(q0, q, carry)
    before = _ledger_sums(q0)
    after = _ledger_sums(q) .+ big.(reshape(carry, size(before)))
    return all(abs.(after .- before) .<= 1e-13 .* abs.(before)) && any(!iszero, carry)
end

# TM5 convection on lat-lon, reduced Gaussian and cubed sphere, with the
# per-thread and collaborative kernels (the CPU runs the per-thread ones for
# both), unmerged and merged.
function check_tm5_ledger_carry(array_type; FT = Float32, calls = 30)
    Nz, Nt = 12, 2
    A = collect(FT, range(0, 5f4; length = Nz + 1)); B = collect(FT, range(1, 0; length = Nz + 1))
    A[1] = 0; B[end] = 0; B[1] = 0; A[end] = 0
    vc = HybridSigmaPressure(A, B)
    # Tracer 1 peaks at the surface but holds the most mass in the pass-through
    # top layer, which neither the solve nor the ledger may touch.
    profile(k, t) = t == 1 ? (k == 1 ? FT(1) : FT(1e-3) * exp(-(Nz - k) / FT(3))) :
                             FT(1e-4) * (1 + FT(k) / Nz)
    # Cloud top at layer 5: layers 1-3 (1-2 with merging) pass through.
    rates(cols) = map(((lo, hi, s),) -> [lo <= k <= hi ? s * (1 + FT(I[1]) / 7) : zero(FT)
                                         for I in CartesianIndices(cols), k in 1:Nz],
                      ((6, Nz - 1, FT(0.03)), (5, 8, FT(0.02)), (7, 9, FT(0.005)), (Nz - 2, Nz, FT(0.005))))
    fields(r) = NamedTuple{(:entu, :detu, :entd, :detd)}(r)
    # The default per-thread kernels, the collaborative ones, and level merging
    # (collaborative only).
    for (use_collab_lu, n_merge) in ((false, 1), (true, 1), (true, 2))
        op = TM5Convection(; use_collab_lu, n_merge)
        for (grid, cols, metrics) in
            ((AtmosGrid(LatLonMesh(; Nx = 4, Ny = 3, FT), vc, AtmosTransport.CPU(); FT), (4, 3), ones(FT, 3)),
             (AtmosGrid(ReducedGaussianMesh(FT[-0.9, 0.0, 0.9], [4, 4, 4]; FT), vc,
                        AtmosTransport.CPU(); FT), (12,), ones(FT, 12)))
            m = [FT(5e3) * (1 + FT(k) / Nz) for _ in CartesianIndices(cols), k in 1:Nz]
            q0 = [m[I, k] * profile(k, t) for I in CartesianIndices(cols), k in 1:Nz, t in 1:Nt]
            q, m_d = array_type(q0), array_type(m)
            forcing = ConvectionForcing(nothing, nothing, fields(map(array_type, rates(cols))))
            ws = Adapt.adapt(array_type, LedgerConv.TM5Workspace(m; cell_metrics = metrics))
            for _ in 1:calls
                LedgerConv.apply_convection!(q, m_d, forcing, op, FT(900), ws, grid)
            end
            @test _ledger_conserved(q0, Array(q), Array(ws.ledger_carry))
            @test selectdim(Array(q), length(cols) + 1, 1) == selectdim(q0, length(cols) + 1, 1)
            # Without convection a column is not ledgered: it and its carry stay as they are.
            q1, carry1 = Array(q), Array(ws.ledger_carry)
            calm = ConvectionForcing(nothing, nothing, fields(map(r -> array_type(zero(r)), rates(cols))))
            LedgerConv.apply_convection!(q, m_d, calm, op, FT(900), ws, grid)
            @test Array(q) == q1 && Array(ws.ledger_carry) == carry1
        end
        Nc, Hp = 3, 1
        grid = AtmosGrid(CubedSphereMesh(; Nc, Hp, FT), vc, AtmosTransport.CPU(); FT)
        m = ntuple(_ -> [FT(5e3) * (1 + FT(k) / Nz) for i in 1:Nc+2Hp, j in 1:Nc+2Hp, k in 1:Nz], 6)
        q0 = ntuple(p -> [m[p][i, j, k] * profile(k, t)
                          for i in 1:Nc+2Hp, j in 1:Nc+2Hp, k in 1:Nz, t in 1:Nt], 6)
        q, m_d = map(array_type, q0), map(array_type, m)
        r = rates((Nc, Nc))
        forcing = ConvectionForcing(nothing, nothing,
                                    fields(ntuple(f -> ntuple(_ -> array_type(r[f]), 6), 4)))
        ws = Adapt.adapt(array_type, LedgerConv.TM5Workspace(m; cell_metrics = ntuple(_ -> ones(FT, Nc, Nc), 6)))
        for _ in 1:calls
            LedgerConv.apply_convection!(q, m_d, forcing, op, FT(900), ws, grid)
        end
        interior(a) = a[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :, :]
        @test all(p -> _ledger_conserved(interior(q0[p]), interior(Array(q[p])),
                                         Array(ws.ledger_carry[p])), 1:6)
        @test all(p -> Array(q[p])[:, :, 1, :] == q0[p][:, :, 1, :], 1:6)
    end
    return nothing
end

# Conservative Dkg diffusion on the cubed sphere: packed tracers, and one
# tracer at a time with its carry slot (`tracer = t`), which must agree.
function check_dkg_ledger_carry(array_type; FT = Float32, calls = 20)
    Nc, Nz, Nt, Hp = 3, 10, 2, 1
    N = Nc + 2Hp
    m = ntuple(p -> [FT(1e9) * (1 + 40 * FT(k / Nz)^2) * (1 + FT(0.01) * (i + 2j + p))
                     for i in 1:N, j in 1:N, k in 1:Nz], 6)
    dkg = ntuple(p -> [k < Nz ? FT(0.3) * min(m[p][i+Hp, j+Hp, k], m[p][i+Hp, j+Hp, k+1]) : zero(FT)
                       for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6)
    rm0 = ntuple(p -> [m[p][i, j, k] * (t == 1 ? FT(1e-3) * exp(-(Nz - k) / FT(3)) : FT(1e-4) * (1 + FT(k) / Nz))
                       for i in 1:N, j in 1:N, k in 1:Nz, t in 1:Nt], 6)
    m_d = map(array_type, m)
    field = Adapt.adapt(array_type, AtmosTransport.State.PrecomputedCSDkgField(dkg))
    op = LedgerDiff.ImplicitVerticalDiffusion(; kz_field = field)
    packed, single = map(array_type, rm0), map(array_type, rm0)
    ws_packed, ws_single = LedgerDiff.DiffusionWorkspace(m_d, Hp, Nt), LedgerDiff.DiffusionWorkspace(m_d, Hp, Nt)
    for _ in 1:calls
        LedgerDiff.apply_vertical_diffusion_vmr!(packed, m_d, op, ws_packed, one(FT); halo_width = Hp)
        for t in 1:Nt
            LedgerDiff.apply_vertical_diffusion_vmr!(map(a -> view(a, :, :, :, t), single), m_d, op,
                                                     ws_single, one(FT); halo_width = Hp, tracer = t)
        end
    end
    interior(a) = a[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :, :]
    @test all(p -> _ledger_conserved(interior(rm0[p]), interior(Array(packed[p])),
                                     Array(ws_packed.ledger_carry[p])), 1:6)
    @test map(Array, packed) == map(Array, single)
    @test map(Array, ws_packed.ledger_carry) == map(Array, ws_single.ledger_carry)
    return nothing
end
