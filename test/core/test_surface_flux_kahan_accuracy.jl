# Float32 emission increments smaller than one ULP must survive repeated
# production kernel calls. References use Float64 sums, without copying Kahan.
using Test, AtmosTransport
using AtmosTransport: apply_surface_flux!

@testset "Float32 surface-flux kernels retain sub-ULP emissions" begin
    background = 1000f0
    # Complete loss, downward rounding and upward rounding with plain +=.
    rates = Float32[1e-7, 0.3eps(background), 0.8eps(background)]
    steps = 1024
    expected = Float64(background) .+ steps .* Float64.(rates)
    tolerance = Float64(eps(background))
    @test all(background .+ rates .!= Float32.(expected))

    for layout in (:structured, :face_packed, :face_single, :cs_packed, :cs_single, :cs_series)
        @testset "$layout" begin
            cs = startswith(string(layout), "cs_")
            q = if layout == :structured
                fill(background, 3, 1, 2, 2)
            elseif layout == :face_packed
                fill(background, 3, 2, 2)
            elseif layout == :face_single
                fill(background, 3, 2)
            else
                shape = layout == :cs_packed || layout == :cs_series ? (5, 5, 2, 2) : (5, 5, 2)
                ntuple(_ -> fill(background, shape), 6)
            end
            # CS interiors are 3×3 with one halo cell on each side.
            rate = cs ? ntuple(_ -> repeat(rates, 1, 3), 6) :
                   layout == :structured ? reshape(copy(rates), 3, 1) : copy(rates)
            source = if layout == :cs_series
                series = map(r -> reshape(copy(r), 3, 3, 1), rate)
                TimeVaryingSurfaceFluxSource(:emitting, series, [0.0], StepwiseFlux())
            else
                SurfaceFluxSource(:emitting, rate)
            end
            op = SurfaceFluxOperator(PerTracerFluxMap(source))
            names = layout in (:face_single, :cs_single) ? (:emitting,) : (:emitting, :unchanged)
            kwargs = cs ? (; tracer_names=names, halo_width=1) : (; tracer_names=names)
            for _ in 1:steps
                apply_surface_flux!(q, op, nothing, 1f0, nothing, nothing; kwargs...)
            end
            if cs
                for panel in q
                    actual = ndims(panel) == 4 ? panel[2:4, 2:4, 2, 1] : panel[2:4, 2:4, 2]
                    @test all(abs.(Float64.(actual) .- expected) .<= tolerance)
                    packed = reshape(panel, 5, 5, 2, :)
                    @test all(==(background), packed[:, :, 1, :])
                    @test all(==(background), packed[[1,end], :, :, :])
                    @test all(==(background), packed[:, [1,end], :, :])
                    if ndims(panel) == 4
                        @test all(==(background), panel[:, :, :, 2])
                    end
                end
            else
                actual = layout == :structured ? q[:, 1, 2, 1] :
                         layout == :face_packed ? q[:, 2, 1] : q[:, 2]
                @test all(abs.(Float64.(actual) .- expected) .<= tolerance)
                @test all(==(background), layout == :structured ? q[:, :, 1, :] : q[:, 1, :])
                if layout != :face_single
                    @test all(==(background), layout == :structured ? q[:, :, :, 2] : q[:, :, 2])
                end
            end
        end
    end
end
