# Global surface-pressure pins of the spectral preprocessors: a uniform offset
# of ps that sets the area-weighted mean total ps (climatological humidity) or
# the area-weighted mean dry ps from the native humidity,
#
#     ⟨Σ_k (dA_k + dB_k ps)(1 − q_k)⟩_area = target_ps_dry.
#
# The lat-lon (Nx, Ny) and reduced-Gaussian (ncell,) column layouts give the
# same result, bit for bit (plan item A11: the reduced-Gaussian path pins with
# the native humidity like the lat-lon and cubed-sphere paths).

using Test
using AtmosTransport
using .AtmosTransport.Preprocessing: pin_global_mean_ps!, pin_global_mean_ps_using_qv!,
                                     rescale_reduced_fluxes_to_ps!
using .AtmosTransport.Grids: face_cells

@testset "global ps pins" begin
    Nx, Ny, Nz = 6, 4, 3
    dA, dB = [1000.0, 3000.0, 500.0], [0.1, 0.3, 0.6]
    ps = [98000.0 + 300sin(i) + 200cos(j) for i in 1:Nx, j in 1:Ny]
    area = [1e10 * cospi((j - 0.5) / Ny - 0.5) for i in 1:Nx, j in 1:Ny]
    qv = [0.002k * (1 + 0.1i + 0.05j) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    dry_ps(sp) = sum(sum((dA[k] + dB[k] * sp[c]) * (1 - qv[c + (k - 1) * Nx * Ny]) for k in 1:Nz) * area[c]
                     for c in eachindex(sp)) / sum(area)

    @testset "dry pin with native humidity" begin
        sp = copy(ps)
        offset = pin_global_mean_ps_using_qv!(sp, area, dA, dB, qv; target_ps_dry_pa = 98726.0)
        @test sp == ps .+ offset
        @test dry_ps(sp) ≈ 98726.0 rtol = 1e-14
        # Reduced-Gaussian layout: one column per cell.
        sp_rg = vec(copy(ps))
        @test pin_global_mean_ps_using_qv!(sp_rg, vec(area), dA, dB, reshape(qv, Nx * Ny, Nz);
                                           target_ps_dry_pa = 98726.0) === offset
        @test sp_rg == vec(sp)
        @test_throws ErrorException pin_global_mean_ps_using_qv!(copy(ps), area, dA, dB, qv[:, 1:2, :])
    end

    @testset "total pin with climatological humidity" begin
        sp = copy(ps)
        offset = pin_global_mean_ps!(sp, area; target_ps_dry_pa = 98726.0, qv_global = 0.00247)
        @test sum(sp .* area) / sum(area) ≈ 98726.0 / (1 - 0.00247) rtol = 1e-14
        sp_rg = vec(copy(ps))
        @test pin_global_mean_ps!(sp_rg, vec(area); target_ps_dry_pa = 98726.0, qv_global = 0.00247) === offset
        @test sp_rg == vec(sp)
    end
end

# Reduced-Gaussian face fluxes are wind × Δp at the face, with the face pressure
# the geometric mean of its two cells; after a pin they scale by Δp_new / Δp_old.
@testset "reduced-Gaussian fluxes follow the pinned pressure" begin
    mesh = ReducedGaussianMesh([-60.0, -20.0, 20.0, 60.0], [4, 8, 8, 4])
    faces = [face_cells(mesh, f) for f in 1:nfaces(mesh)]
    left, right = first.(faces), last.(faces)
    dA, dB = [500.0, 2000.0], [0.0, 0.4]
    lnsp = [log(1e5 + 500sin(c)) for c in 1:ncells(mesh)]
    sp = exp.(lnsp) .+ 37.0
    hflux = [1e6 * cos(f + k) for f in 1:nfaces(mesh), k in 1:2]
    expected = copy(hflux)
    for k in 1:2, f in 1:nfaces(mesh)
        (left[f] == 0 || right[f] == 0) && (expected[f, k] = 0; continue)   # polar stubs
        p_old, p_new = exp((lnsp[left[f]] + lnsp[right[f]]) / 2), sqrt(sp[left[f]] * sp[right[f]])
        expected[f, k] *= (dA[k] + dB[k] * p_new) / (dA[k] + dB[k] * p_old)
    end
    rescale_reduced_fluxes_to_ps!(hflux, lnsp, sp, left, right, dA, dB)
    @test hflux ≈ expected rtol = 1e-14
    interior = (left .!= 0) .& (right .!= 0)
    @test hflux[interior, 1] ≈ [1e6 * cos(f + 1) for f in 1:nfaces(mesh)][interior] rtol = 1e-14   # pure-pressure level
end
