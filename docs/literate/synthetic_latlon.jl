# # Tutorial: synthetic lat-lon end-to-end
#
# This tutorial builds a tiny synthetic transport binary locally,
# loads it with the runtime driver, runs a couple of advection steps,
# and checks mass conservation — all without any external met data.
# Everything here uses public API; the same pattern underlies
# `test/core/test_driven_simulation.jl`.
#
# Use this as a template when you want to:
#
# - smoke-test the runtime on a fresh install,
# - explore the data flow (binary → driver → state → simulation),
# - build a custom synthetic case for a unit test.
#
# Run the source directly from the repository root with
# `julia --project=. docs/literate/synthetic_latlon.jl`, or read the executed
# version in the documentation site.

using AtmosTransport

# ## 1. Build a tiny synthetic binary locally
#
# We assemble two windows on an 8 × 4 × 2 lat-lon grid. A small constant
# eastward mass flux is periodic and divergence-free: it moves a tracer blob
# while total air mass and conservative tracer storage remain constant.

function build_demo_latlon_binary(path::AbstractString;
                                  FT::Type{<:AbstractFloat} = Float64,
                                  Nx::Int = 8, Ny::Int = 4, Nz::Int = 2,
                                  nwindow::Int = 2)
    mesh     = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid     = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    windows = [
        (; m  = ones(FT, Nx, Ny, Nz),
           am = fill(FT(0.03), Nx + 1, Ny, Nz),
           bm = zeros(FT, Nx, Ny + 1, Nz),
           cm = zeros(FT, Nx, Ny, Nz + 1),
           ps = fill(FT(95_000 + 100w), Nx, Ny),
           qv_start = fill(FT(0.01w),         Nx, Ny, Nz),
           qv_end   = fill(FT(0.01w + 0.01),  Nx, Ny, Nz),
           dam      = zeros(FT, Nx + 1, Ny, Nz),
           dbm      = zeros(FT, Nx, Ny + 1, Nz),
           dcm      = zeros(FT, Nx, Ny, Nz + 1),
           dm       = zeros(FT, Nx, Ny, Nz))
        for w in 1:nwindow
    ]

    write_transport_binary(path, grid, windows;
                           FT                   = FT,
                           dt_met_seconds       = 3600.0,
                           half_dt_seconds      = 1800.0,
                           steps_per_window     = 2,
                           mass_basis           = :dry,
                           surface_pressure_basis = TotalSurfacePressure(),
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling        = :window_constant,
                           extra_header         = Dict(
                               "poisson_balance_target_scale" => 0.25,
                               "poisson_balance_target_semantics" =>
                                   "forward_window_mass_difference / (2 * steps_per_window)",
                           ))
    return grid
end

bin_path = joinpath(mktempdir(), "synthetic_latlon.bin")
grid = build_demo_latlon_binary(bin_path; FT = Float64)
@info "Wrote synthetic binary" bin_path

# ## 2. Inspect the binary
#
# `inspect_binary` is the public diagnostic; it returns a NamedTuple
# of capability flags so a runtime can decide which operators are
# eligible.

caps = AtmosTransport.inspect_binary(bin_path)
caps

# ## 3. Construct driver, state, model, simulation
#
# - `TransportBinaryDriver` memory-maps the binary.
# - `CellState` carries air mass + tracers.
# - `TransportModel` bundles state + fluxes + grid + advection scheme.
# - `DrivenSimulation` ties the model to the driver and steps it
#   forward window by window.

driver = TransportBinaryDriver(bin_path; FT = Float64, arch = CPU())

# `CellState` stores the conservative model quantity `χ × air_mass`, not VMR
# and not physical kilograms of CO₂. We initialize a Gaussian-like dry-VMR
# enhancement so the transport is visible.
air_mass_arr = ones(Float64, 8, 4, 2)
co2_vmr = [400e-6 + 80e-6 * exp(-((i - 3) / 1.2)^2 - ((j - 2) / 0.8)^2)
           for i in 1:8, j in 1:4, _ in 1:2]
state = CellState(DryBasis,
                  air_mass_arr;
                  CO2 = co2_vmr .* air_mass_arr)

initial_air_mass = total_air_mass(state)
initial_storage = total_mass(state, :CO2)
initial_vmr = copy(co2_vmr)

fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = Float64,
                              basis = DryBasis)

model = TransportModel(state, fluxes, grid, UpwindScheme())

sim = DrivenSimulation(model, driver;
                       start_window = 1,
                       stop_window  = 2);    # `;` suppresses the object dump

# ## 4. Step the simulation
#
# Each `step!` advances by one substep (here: 30 minutes; the binary
# declares `dt_met_seconds = 3600` and `steps_per_window = 2`).

step!(sim)
step!(sim)
step!(sim)

# ## 5. Confirm mass conservation
#
# The checks below verify all three promises of the example: air-mass
# conservation, tracer-storage conservation, and actual displacement.

using Statistics
m_min, m_max = extrema(sim.model.state.air_mass)
@info "Air mass extrema after 3 steps" m_min m_max

vmr = mixing_ratio(sim.model.state, :CO2)
vmr_min, vmr_max = extrema(vmr)
@info "CO2 dry-VMR extrema after 3 steps" vmr_min vmr_max

@assert isapprox(total_air_mass(sim.model.state), initial_air_mass; rtol = 1e-12)
@assert isapprox(total_mass(sim.model.state, :CO2), initial_storage; rtol = 1e-12)
@assert maximum(abs.(vmr .- initial_vmr)) > 1e-8

# Cleanup the driver (closes the memory map):
close(driver)

# ## What's next
#
# - Change the synthetic fluxes or initial blob and inspect how transport
#   responds while the two conservation assertions remain satisfied.
# - Swap `UpwindScheme()` for `SlopesScheme()` or `PPMScheme()` to
#   see scheme-dependent behavior on the same forcing.
# - Add another tracer with its own initial pattern and compare how both
#   conservative storage fields move through the same air-mass fluxes.
# - Swap the synthetic binary for a real preprocessed ERA5 day
#   produced by `scripts/preprocessing/preprocess_transport_binary.jl`
#   and the same code structure runs against real meteorology.
