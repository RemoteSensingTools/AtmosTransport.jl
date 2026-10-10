module CSDriverHandoffFixtures

using AtmosTransport, Logging
const MD = AtmosTransport.MetDrivers

function cs_handoff_fixture(path, strengths; FT=Float64)
    Nc, Nz = 3, 5          # Nc ≥ 3: the PPM runs use a 3-cell halo, which must not exceed the panel
    vc = HybridSigmaPressure(fill(FT(100),Nz+1),FT.(range(0,1;length=Nz+1)))
    writer = MD.open_streaming_cs_transport_binary(path,Nc,6,Nz,length(strengths),vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
        FT,dt_met_seconds=3600.0,steps_per_window=2,mass_basis=:dry,
        include_cmfmc=true,include_dtrain=true,include_tm5conv=true)
    try
        for strength in strengths
            profile = FT(strength) .* FT[0,0.01,0.02,0.02,0.01,0]
            cmfmc = ntuple(_->repeat(reshape(profile,1,1,Nz+1),Nc,Nc,1),6)
            dtrain = ntuple(_->zeros(FT,Nc,Nc,Nz),6)
            entu = ntuple(_->zeros(FT,Nc,Nc,Nz),6)
            detu = ntuple(_->zeros(FT,Nc,Nc,Nz),6)
            for p in 1:6
                dtrain[p][:,:,3] .= FT(strength*0.005)
                entu[p][:,:,Nz] .= FT(strength*0.02)
                detu[p][:,:,2] .= FT(strength*0.02)
            end
            window = (;m=ntuple(_->fill(FT(1e16),Nc,Nc,Nz),6),
                am=ntuple(_->zeros(FT,Nc+1,Nc,Nz),6),
                bm=ntuple(_->zeros(FT,Nc,Nc+1,Nz),6),
                cm=ntuple(_->zeros(FT,Nc,Nc,Nz+1),6),
                ps=ntuple(_->fill(FT(100000),Nc,Nc),6),cmfmc,dtrain,
                tm5_fields=(;entu,detu,entd=ntuple(_->zeros(FT,Nc,Nc,Nz),6),
                                      detd=ntuple(_->zeros(FT,Nc,Nc,Nz),6)))
            MD.write_streaming_cs_window!(writer,window,Nc,6)
        end
    finally
        MD.close_streaming_transport_binary!(writer)
    end
end

function cs_handoff_run(paths, advection, convection; use_gpu=false, FT=Float64)
    cfg = Dict{String,Any}(
        "input"=>Dict("binary_paths"=>paths),
        "architecture"=>Dict("use_gpu"=>use_gpu),
        "numerics"=>Dict("float_type"=>string(FT)),
        "advection"=>Dict("scheme"=>advection),
        "diffusion"=>Dict("kind"=>"constant","value"=>10.0),
        "convection"=>Dict("kind"=>convection),
        "chemistry"=>Dict("kind"=>"decay","half_lives_seconds"=>Dict("co2"=>1e5,"tag"=>2e5)),
        "tracers"=>Dict(
            "co2"=>Dict("init"=>Dict("kind"=>"pressure_layer","lowest_layer"=>true,"total_molecules"=>1e35)),
            "tag"=>Dict("init"=>Dict("kind"=>"pressure_layer","psurf_fraction"=>0.4,"total_molecules"=>2e35))))
    with_logger(NullLogger()) do
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                AtmosTransport.Models.run_driven_simulation(cfg)
            end
        end
    end
end


export cs_handoff_fixture, cs_handoff_run
end
