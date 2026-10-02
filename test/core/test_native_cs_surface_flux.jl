using Test
using Dates
using NCDatasets
using AtmosTransport
using AtmosTransport.Models.InitialConditionIO: build_surface_flux_source

@testset "Native CS time-varying surface flux" begin
    for FT in (Float32, Float64)
        mktempdir() do dir
            Nc = 4
            path = joinpath(dir, "native.nc")
            NCDataset(path, "c") do ds
                for (name, n) in (("Xdim", Nc), ("Ydim", Nc), ("nf", 6), ("time", 3))
                    defDim(ds, name, n)
                end
                defVar(ds, "time", Float64, ("time",), attrib=Dict(
                    "units" => "hours since 2014-01-01 00:00:00 UTC"))[:] = [0,24,48]
                v = defVar(ds, "NPP", Float32, ("Xdim","Ydim","nf","time"),
                           attrib=Dict("units" => "kg CO2 m-2 s-1"))
                # Unequal i/j, panel, and time patterns expose any transposition.
                for t in 1:3, p in 1:6, j in 1:Nc, i in 1:Nc
                    v[i,j,p,t] = -Float32(i + 10j + 100p + 1000t)*1f-10
                end
            end
            mesh = CubedSphereMesh(; FT, Nc, Hp=1)
            vertical = HybridSigmaPressure(FT[0,50000,0], FT[1,0.5,0])
            grid = AtmosGrid(mesh, vertical, CPU(); FT)
            cfg = Dict{String,Any}("kind"=>"cs_native", "file"=>path, "variable"=>"NPP",
                "time_varying"=>true, "molar_mass_kg_mol"=>0.0440095, "scale"=>2.0)
            source = build_surface_flux_source(grid, :custom_npp, cfg, FT;
                                               reference_time=DateTime(2014,1,2))
            @test source isa TimeVaryingSurfaceFluxSource
            @test source.scheme isa StepwiseFlux
            @test source.times == [-86400,0,86400]
            for p in 1:6, t in 1:3
                density = [-Float32(i+10j+100p+1000t)*1f-10 for i in 1:Nc, j in 1:Nc]
                expected = Float64.(density) .* Float64.(mesh.cell_areas) .* (2*0.02896546/0.0440095)
                actual = source.cell_mass_rate_series[p][:,:,t]
                @test actual ≈ expected rtol=3e-7
                @test all(<(0), actual)
            end
            NCDataset(path, "a") do ds
                ds["NPP"][1,1,1,1] = NaN32
            end
            @test_throws ArgumentError build_surface_flux_source(grid, :custom_npp, cfg, FT)
            NCDataset(path, "a") do ds
                ds["NPP"][1,1,1,1] = 0
                ds["time"][:] = [48,24,0]
            end
            @test_throws ArgumentError build_surface_flux_source(grid, :custom_npp, cfg, FT)
        end
    end
end
