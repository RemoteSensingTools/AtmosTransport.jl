# Reference transcription of TM5's ERA5 -> TM5 convection conversion, used to
# check `Preprocessing.ec2tm_from_rates!` and the ERA5 N320 call site against
# TM5 itself (issue #54).
#
# Sources (written separately from the production code):
#   - packing of the archived fields, TM5 `base/src/tmm.F90` (`tmm_Read_Convec_EC`):
#         udmf_gg(:,0:nlev-1) = buf ; udmf_gg(:,nlev) = 0
#     "levels top down, surface implicit zero": GRIB level k (1 = model top) is
#     TM5 half level k - 1, the TOP of layer k; half level nlev is the ground.
#   - subroutine `ECconv_to_TMconv` in TM5 `base/src/phys_convec_ec2tm.F90`:
#     clipping, dz integration, the search `do l = 1, lm; if (mflu(l) > 0)`
#     for the first layer whose BOTTOM half level carries flux, the budget
#     `entu(l) = mflu(l-1) - mflu(l) + detu(l)` from that layer down, and the
#     symmetric repair of negative values;
#   - the column closure in TM5 `base/src/meteo.F90` (lines 2023-2052): the
#     residual `totd - tote` is added at the level of largest entrainment,
#     searched from the surface upward (strict `>`, so the lowest of equal
#     maxima wins).
#
# Index convention: TM5's 0-based half-level arrays `mflu(0:lm)` are emulated
# with `hl(v, l) = v[l + 1]`; layers are numbered 1..lm from the model top as in
# TM5's ECMWF-order arrays. The single documented deviation is the sign of the
# closure: TM5 subtracts from `entu` when the residual is negative, which can
# make it negative; AtmosTransport adds the magnitude to `detu` instead (same
# net `entu - detu`), and so does this reference.

"""
    tm5_ecconv_reference(grib_udmf, grib_ddmf, udrf_rate, ddrf_rate, dz)
        -> (entu, detu, entd, detd)

TM5's conversion of one ERA5 column. `grib_udmf`, `grib_ddmf` are the archived
mass fluxes on GRIB levels 1..lm (kg m⁻² s⁻¹, level 1 at the model top);
`udrf_rate`, `ddrf_rate` the detrainment rates on layers 1..lm (kg m⁻³ s⁻¹);
`dz` the layer thicknesses (m).
"""
function tm5_ecconv_reference(grib_udmf, grib_ddmf, udrf_rate, ddrf_rate, dz)
    lm = length(dz)
    length(grib_udmf) == length(grib_ddmf) == length(udrf_rate) == length(ddrf_rate) == lm ||
        throw(DimensionMismatch("all ERA5 columns must have lm = $lm levels"))
    hl(v, l) = v[l + 1]

    # tmm.F90: mflu(0:lm-1) = GRIB levels 1..lm, mflu(lm) = 0.
    mflu = vcat(Float64.(grib_udmf), 0.0)
    mfld = vcat(Float64.(grib_ddmf), 0.0)
    detu = Float64.(udrf_rate)
    detd = Float64.(ddrf_rate)
    entu = zeros(lm)
    entd = zeros(lm)

    # Removing small values due to gribbing.
    mflu[mflu .< 1e-6] .= 0.0
    detu[detu .< 1e-10] .= 0.0
    mfld[mfld .> -1e-6] .= 0.0
    detd[detd .< 1e-10] .= 0.0

    # Height integral of the detrainment rates.
    detu .*= dz
    detd .*= dz

    # do l = 1, lm: first layer whose bottom half level l carries flux.
    uptop = something(findfirst(l -> hl(mflu, l) > 0.0, 1:lm), 0)
    dotop = something(findfirst(l -> hl(mfld, l) < 0.0, 1:lm), 0)

    if uptop > 0
        detu[1:uptop-1] .= 0.0
        for l in uptop:lm
            entu[l] = hl(mflu, l - 1) - hl(mflu, l) + detu[l]
        end
    else
        detu .= 0.0
    end
    if dotop > 0
        detd[1:dotop-1] .= 0.0
        for l in dotop:lm
            entd[l] = hl(mfld, l - 1) - hl(mfld, l) + detd[l]
        end
    else
        detd .= 0.0
    end

    for l in 1:lm
        if entu[l] < 0.0
            detu[l] -= entu[l]; entu[l] = 0.0
        end
        if detu[l] < 0.0
            entu[l] -= detu[l]; detu[l] = 0.0
        end
        if entd[l] < 0.0
            detd[l] -= entd[l]; entd[l] = 0.0
        end
        if detd[l] < 0.0
            entd[l] -= detd[l]; detd[l] = 0.0
        end
    end

    # meteo.F90 closure; TM5's level 1 is the surface, i.e. layer lm here.
    _tm5_reference_closure!(entu, detu, lm, entu[lm])
    _tm5_reference_closure!(entd, detd, 1, 0.0)
    return (entu, detu, entd, detd)
end

function _tm5_reference_closure!(ent, det, lsave, maxe)
    lm = length(ent)
    for l in lm:-1:1                       # from the surface upward
        if ent[l] > maxe
            maxe = ent[l]
            lsave = l
        end
    end
    residual = sum(det) - sum(ent)
    if residual >= 0
        ent[lsave] += residual
    else
        det[lsave] -= residual             # documented sign-aware deviation
    end
    return nothing
end

"""
    implied_updraft_flux(entu, detu) -> Vector

Upward mass flux through interfaces 1..lm+1 (interface k is the top of layer
k) implied by a TM5 updraft, integrated from zero at the ground.
"""
function implied_updraft_flux(entu, detu)
    lm = length(entu)
    flux = zeros(lm + 1)
    for k in lm:-1:1
        flux[k] = flux[k + 1] + entu[k] - detu[k]
    end
    return flux
end

"""
    implied_downdraft_flux(entd, detd) -> Vector

Downdraft mass flux (ECMWF sign, ≤ 0) through interfaces 1..lm+1 implied by a
TM5 downdraft, integrated downward from zero at the model top.
"""
function implied_downdraft_flux(entd, detd)
    lm = length(entd)
    flux = zeros(lm + 1)
    for k in 1:lm
        flux[k + 1] = flux[k] - (entd[k] - detd[k])
    end
    return flux
end
