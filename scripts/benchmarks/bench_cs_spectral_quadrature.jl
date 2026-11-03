# CPU-only reproducible comparison, default C90 and C180 at T639/L137.
# TMPDIR=/temp2/cfranken/tmp JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#   julia --startup-file=no --project=test scripts/benchmarks/bench_cs_spectral_quadrature.jl [90|180]
# Window timings scale warmed eight-level measurements to two L137 endpoints;
# surface/cell overlap, Poisson balance, closure and IO are excluded.
# Set ATMOSTR_BENCH_FULL_WINDOW=1 to additionally time 36 actual level
# batches including the production Cartesian/forcing integrand kernel.
# ATMOSTR_BENCH_EXACT_LATITUDES=1 counts the table storage required by
# a conventional exact-sample-latitude ring plan, exploiting polar symmetry.
using AtmosTransport, LinearAlgebra, SparseArrays, KernelAbstractions, Random
const PP=AtmosTransport.Preprocessing; const SP=PP.Spectral; const MC=PP.MassClosure
const RV=AtmosTransport.Parameters.IFS_EARTH_RADIUS
BLAS.set_num_threads(1)
function benchmark_exact_latitudes(q,grid,T)
 n=q.samples_per_face;ft=grid.face_table
 latitudes=Vector{Int64}(undef,ft.nf*n)
 elapsed=@elapsed for f in 1:ft.nf
  a,b=MC.cs_face_corners(grid.mesh,ft,f);theta=AtmosTransport.Grids.arc3(a,b)
  for i in 1:n
   x=AtmosTransport.Grids.slerp3(a,b,theta,(i-0.5)/n)
   latitudes[(f-1)*n+i]=round(Int64,abs(asin(clamp(x[3],-1.,1.)))*1e12)
  end
 end
 unique!(sort!(latitudes));nl=length(latitudes)
 @info "Exact sample-latitude geometry" nc=grid.Nc points=ft.nf*n unique_absolute_latitudes=nl legendre_f64_gb=8*((T+2)*(T+3)÷2)*nl/1e9 ideal_legendre_work_ratio=nl/length(q.rings.latitudes) seconds=elapsed tolerance_radians=1e-12
end
function benchmark_face_window(nc,q,vo,d,s,polar,op,nz)
  nf=s.plan.batch;ns=q.rings.offsets[end]-1;nfaces=length(op.offsets)-1
  lon=repeat(deg2rad.(q.mesh.λᶜ),q.mesh.Ny)
  lat=repeat(deg2rad.(q.mesh.φᶜ);inner=q.mesh.Nx)
  sl,cl,sp,cp=sin.(lon),cos.(lon),sin.(lat),cos.(lat)
  pressure=fill(100000.,ns,1);surface=zeros(ns,3)
  aa=fill(10000.,nf);bb=zeros(nf)
  U,V,D=SP.synthesize_native_fields(vo,d,s.plan.T-1,RV,s)
  xyz=similar(U,3ns,nf);out=zeros(nfaces,nf)
  forcing=similar(U)
  gravity=AtmosTransport.Parameters.STANDARD_GRAVITY
  backend=get_backend(U)
  SP.cs_spectral_integrands!(backend,256)(xyz,forcing,U,V,D,pressure,surface,
   sl,cl,sp,cp,aa,bb,q.radius,RV,gravity;ndrange=(ns,nf))
  synchronize(backend)
  window=@elapsed for endpoint in 1:2, firstlevel in 1:nf:nz
   U,V,D=SP.synthesize_native_fields(vo,d,s.plan.T-1,RV,s)
   SP.cs_stable_polar_winds!(U,V,vo,d,RV,polar)
   xyz=similar(U,3ns,nf);forcing=similar(U)
   SP.cs_spectral_integrands!(backend,256)(xyz,forcing,U,V,D,pressure,surface,
    sl,cl,sp,cp,aa,bb,q.radius,RV,gravity;ndrange=(ns,nf))
   SP.cs_apply_integral!(out,xyz,op);synchronize(backend)
  end
  @info "Measured face window" nc seconds=window batches=2cld(nz,nf) includes="U/V/D, stable polar, Cartesian product/forcing, face gather; excludes surface synthesis/cell integration/balance/closure/IO"
end
function benchmark(nc)
 T=639; nf=8; nz=137
 grid=PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>nc),Float64)
 q=SP.CSSpectralQuadrature(grid,T,PP.TM5SpectralFluxes())
 if get(ENV,"ATMOSTR_BENCH_EXACT_LATITUDES","0") == "1"
  benchmark_exact_latitudes(q,grid,T)
 end
 @info "benchmark begin" nc shape=(q.mesh.Nx,q.mesh.Ny) samples=q.samples_per_face
 elapsed=@elapsed new=MC.cs_spectral_face_geometry(q.mesh,grid.mesh,grid.face_table;samples_per_face=q.samples_per_face)
 @info "face setup" nc elapsed nnz=nnz(new.stencils) matrix_gb=Base.summarysize(new)/1e9
 elapsed=@elapsed old=MC.line_integral_face_matrix(q.mesh,grid.mesh,grid.face_table;samples_per_face=max(16,4SP.quadrature_refinement(T,rad2deg(minimum(first(AtmosTransport.Grids.cs_face_edge_lengths(grid.mesh)))/q.radius))))
 @info "old face setup" nc elapsed nnz=nnz(old)
 ops=[SP.cs_gather_operator(a,identity,Float64) for a in (old,new)]
 old=nothing;new=nothing;GC.gc()
 s=SP.CPURingSynthesis(q.rings,T+1,nf)
 polar=SP.cs_polar_synthesis(s,T)
 Random.seed!(42)
 vo=zeros(ComplexF64,T+1,T+1,nf);d=copy(vo)
 for k in 1:nf,m in 0:T,n in m:T
  vo[n+1,m+1,k]=1e-7*randn(ComplexF64)/(n+1)^2
  d[n+1,m+1,k]=1e-8*randn(ComplexF64)/(n+1)^2
 end
 vo[1,1,:].=0;d[1,1,:].=0
 # Warm complete operations, then measure a production-sized level batch.
 U,V,D=SP.synthesize_native_fields(vo,d,T,RV,s)
 ts=@elapsed U,V,D=SP.synthesize_native_fields(vo,d,T,RV,s)
 SP.cs_stable_polar_winds!(U,V,vo,d,RV,polar)
 tp=@elapsed SP.cs_stable_polar_winds!(U,V,vo,d,RV,polar)
 ns=size(U,1)
 # Matrices exercise the real production gather sizes and actual weights.
 xyz=repeat(U,3,1); out=zeros(grid.face_table.nf,nf)
 tg=Float64[]
 for op in ops
  SP.cs_apply_integral!(out,xyz,op); synchronize(get_backend(out))
  push!(tg,@elapsed begin SP.cs_apply_integral!(out,xyz,op);synchronize(get_backend(out)) end)
 end
 lat=repeat([0.01,0.3,0.9,1.56],16);lon=collect(range(0.,6.;length=64))
 H=zeros(ComplexF64,T+1,T+1);H[1]=log(100000.)
 SP.cs_exact_samples(H,vo,d,T,RV,lon,lat)
 te=@elapsed SP.cs_exact_samples(H,vo,d,T,RV,lon,lat)
 bytes_s=Base.summarysize(s);bytes_p=Base.summarysize(polar)
 bytes_op=Base.summarysize(ops[2])
 buffers=Base.summarysize((U,V,D,xyz,out))
 # Additional production xyz/forcing and surface buffers; batch peaks are
 # estimates, whereas Julia summary sizes above are measured live storage.
 live=bytes_s+bytes_p+bytes_op+buffers+8ns*nf+8ns*5
 perwindow=2nz/nf
 @info "cost measurement" nc ts tp gather_old=tg[1] gather_new=tg[2] old_seconds_window=perwindow*(ts+tg[1]) new_seconds_window=perwindow*(ts+tp+tg[2]) exact_seconds_window=te/64*grid.face_table.nf*q.samples_per_face*perwindow ratio_exact_to_old=(te/64*grid.face_table.nf*q.samples_per_face)/(ts+tg[1]) plan_gb=bytes_s/1e9 polar_gb=bytes_p/1e9 operator_gb=bytes_op/1e9 estimated_live_gb=live/1e9 exact_chunk_seconds=te
 # Explicit model, not a hardware measurement: 500 GB/s gather traffic,
 # 1 TFLOP/s FP64 recurrence throughput. Gather reads index/weight/value.
 traffic=length(ops[2].weights)*nf*(12+3*8)
 @info "GPU model" nc gather_seconds_window=2cld(nz,nf)*traffic/500e9 gather_seconds_unpadded=perwindow*traffic/500e9 direct_seconds_window=2nz*grid.face_table.nf*q.samples_per_face*(T+1)*(T+2)/2*45/1e12 assumptions="500 GB/s effective gather, 1 TFLOP/s FP64 recurrence; excludes launch/FFT/GEMM and transfer"
 if get(ENV,"ATMOSTR_BENCH_FULL_WINDOW","0") == "1"
  Base.invokelatest(benchmark_face_window,nc,q,vo,d,s,polar,ops[2],nz)
 end
 @info "Process peak memory" nc maxrss_gb=Sys.maxrss()/1e9
 flush(stderr)
end
for nc in (isempty(ARGS) ? (90,180) : parse.(Int,ARGS))
 benchmark(nc)
end
