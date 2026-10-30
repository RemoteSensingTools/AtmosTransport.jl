# Split-sweep PPM adjoints exist for the default vertical reconstruction only;
# the FV3 vertical profile has an adjoint as Lin–Rood's vertical sweep (below).
const CSAdjointLinearScheme = Union{UpwindScheme, SlopesScheme{NoLimiter},
                                    PPMScheme{NoLimiter, SameAsHorizontal}}
# Limiters whose PPM adjoint replays the forward branches from taped tracer masses.
const _PPMTapedLimiter = Union{MonotoneLimiter, CW84Limiter}
const CSAdjointNonlinearScheme = PPMScheme{<:_PPMTapedLimiter, SameAsHorizontal}
# The FV3 `kord = 8` vertical sweep (`vertical = "fv3_kord8"` or
# `"fv3_kord8_signed"`). Its records stage the tracer mass entering the sweep;
# the reverse replays the forward profile there and applies the transpose at the
# forward's limiter branches (`vertical_fv3_profile_adjoint.jl`). Only Lin–Rood's
# vertical sweeps record it.
const CSAdjointFV3Vertical = PPMScheme{<:AbstractLimiter, <:FV3ScalarProfile}

"""
    CSAdjointWorkspace(mesh, prototype; column_scratch = false)

Scratch storage for CS adjoint sweeps. `lambda_A` is one halo-padded panel
used as the per-panel transpose output. `seam_seeds` caches both neighbors'
output-seed difference before any panel transpose overwrites those values.
`column_scratch` is the per-column working storage of the FV3 vertical sweep's
reverse pass (`Nc × Nc × (Nz+1) × 5`); it is empty unless built with
`column_scratch = true`, which a tape with FV3-profile vertical sweeps needs
(`needs_column_scratch(scheme)`).
"""
struct CSAdjointWorkspace{FT, A <: AbstractArray{FT, 3}, A4 <: AbstractArray{FT, 4}}
    lambda_A::A
    seam_seeds::A
    column_scratch::A4
end

function CSAdjointWorkspace(mesh::CubedSphereMesh,
                            prototype::AbstractArray{FT, 3};
                            column_scratch::Bool = false) where {FT <: AbstractFloat}
    N = mesh.Nc + 2 * mesh.Hp
    Nz = size(prototype, 3)
    lambda_A = similar(prototype, FT, N, N, Nz)
    seam_seeds = similar(prototype, FT, mesh.Nc, Nz, 12)
    scratch = _fv3_adjoint_column_scratch(lambda_A, mesh.Nc, Nz, column_scratch)
    return CSAdjointWorkspace{FT, typeof(lambda_A), typeof(scratch)}(lambda_A, seam_seeds, scratch)
end
