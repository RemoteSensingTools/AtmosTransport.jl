# Included by both device drivers. Driver pool high-water marks are exact
# within the active CUDA pool; nvidia-smi adds sampled whole-device usage.
function record_face_device_memory(path)
    CUDA.synchronize()
    pool = CUDA.memory_pool(CUDA.device())
    used = CUDA.attribute(UInt64,pool,CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)
    reserved = CUDA.attribute(UInt64,pool,CUDA.MEMPOOL_ATTR_RESERVED_MEM_HIGH)
    open(path,"w") do io
        TOML.print(io,Dict("pool_used_high_bytes"=>Int(used),"pool_reserved_high_bytes"=>Int(reserved),
            "scope"=>"process lifetime including warmup; vendor allocations outside pool excluded"))
    end
    CUDA.pool_status()
end

# Start only after the GPU command acquired gpu1.lock. Sampling while the
# launcher is waiting for the lock would attribute another job's usage here.
function start_face_device_memory(path)
    io = open(path,"w")
    process = run(pipeline(`nvidia-smi -i 1 --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -lms 100`;
        stdout=io,stderr=io);wait=false)
    atexit() do
        try
            process_running(process) && kill(process)
            wait(process)
        catch
            # The monitor is normally terminated by SIGTERM on process exit.
        finally
            close(io)
        end
    end
    return nothing
end
