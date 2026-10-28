using AtmosTransport, Test
const ResetPlans = AtmosTransport.Operators.StepPlans
const ResetAdvection = AtmosTransport.Operators.Advection

# The preserve-VMR reset rescales all tracers in one broadcast over the packed
# array; it must equal the per-tracer formula bit for bit, including cells
# whose old air mass is at or below the floor.
_reset_reference!(rm, old, new, floor_m) =
    (@. rm = ifelse(old > floor_m, rm / old * new, zero(eltype(rm))); rm)

@testset "Air-mass reset over the packed tracer array" begin
    for FT in (Float32, Float64)
        floor_m = eps(FT)
        @testset "CellState $FT" begin
            Nx, Ny, Nz, Nt = 4, 3, 2, 5
            old = FT[1 + i + 2j + 3k for i in 1:Nx, j in 1:Ny, k in 1:Nz]
            old[2, 1, 1] = 0
            old[3, 2, 2] = floor_m
            new = FT[2 + 0.5i + j - k for i in 1:Nx, j in 1:Ny, k in 1:Nz]
            tracers = (Symbol("t$n") => FT[n * (i + j + k) / 7 for i in 1:Nx, j in 1:Ny, k in 1:Nz]
                       for n in 1:Nt)
            state = CellState(copy(old); tracers...)
            expected = [_reset_reference!(copy(selectdim(state.tracers_raw, 4, n)), old, new, floor_m)
                        for n in 1:Nt]
            ResetPlans._reset_air_mass!(state, new, nothing, :preserve_vmr)
            @test all(selectdim(state.tracers_raw, 4, n) == expected[n] for n in 1:Nt)
            @test state.air_mass == new
        end
        @testset "CellState reduced-Gaussian shape $FT" begin
            Ncell, Nz, Nt = 7, 3, 4
            old = FT[1 + c + 2k for c in 1:Ncell, k in 1:Nz]
            old[4, 2] = 0
            new = FT[3 + 0.25c - k for c in 1:Ncell, k in 1:Nz]
            tracers = (Symbol("t$n") => FT[n * (c + k) / 5 for c in 1:Ncell, k in 1:Nz] for n in 1:Nt)
            state = CellState(copy(old); tracers...)
            expected = [_reset_reference!(copy(selectdim(state.tracers_raw, 3, n)), old, new, floor_m)
                        for n in 1:Nt]
            ResetPlans._reset_air_mass!(state, new, nothing, :preserve_vmr)
            @test all(selectdim(state.tracers_raw, 3, n) == expected[n] for n in 1:Nt)
            @test state.air_mass == new
        end
        @testset "CubedSphereState $FT" begin
            Nc, Hp, Nz, Nt = 3, 1, 2, 5
            mesh = CubedSphereMesh(; Nc, Hp, FT)
            N = Nc + 2Hp
            old = ntuple(p -> FT[p + i + 2j + 3k for i in 1:N, j in 1:N, k in 1:Nz], 6)
            old[2][2, 2, 1] = 0
            new = ntuple(p -> FT[2p + 0.5i + j - k for i in 1:N, j in 1:N, k in 1:Nz], 6)
            tracers = (Symbol("t$n") => ntuple(p -> FT[n * (p + i + j + k) / 7
                                                        for i in 1:N, j in 1:N, k in 1:Nz], 6)
                       for n in 1:Nt)
            state = CubedSphereState(DryBasis, mesh, map(copy, old); tracers...)
            reference = deepcopy(state)
            new_filled = map(copy, new)
            ResetAdvection.fill_panel_halos!(new_filled, mesh; dir = 1)
            for p in 1:6, n in 1:Nt
                _reset_reference!(selectdim(reference.tracers_raw[p], 4, n), old[p],
                                  new_filled[p], floor_m)
            end
            foreach(copyto!, reference.air_mass, new_filled)
            ResetPlans._refresh_state_halos!(reference, mesh)
            ResetPlans._reset_air_mass!(state, map(copy, new), mesh, :preserve_vmr)
            @test all(state.tracers_raw[p] == reference.tracers_raw[p] for p in 1:6)
            @test all(state.air_mass[p] == reference.air_mass[p] for p in 1:6)
        end
    end
end
