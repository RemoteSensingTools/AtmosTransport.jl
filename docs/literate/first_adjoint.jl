# # [Tutorial: your first emission footprint](@id First-emission-footprint)
#
# A forward calculation asks: **what mixing ratio follows from these emissions?**
# An adjoint asks: **how would one chosen result change if I changed emissions
# in each cell and at each earlier time?** This tutorial runs both calculations
# on a tiny CPU cubed sphere and checks one sensitivity with finite differences.
# No meteorological download or GPU is needed.
#
# Run this file from the repository root:
# `julia --project=. docs/literate/first_adjoint.jl`.
# The documentation build also executes every assertion below.

using AtmosTransport
using AtmosTransport.Adjoints: CSLayerMeanObjective, FullCheckpoint,
    cs_surface_emission_footprint, run_cs_footprint_forward

# ## 1. Define a small transport problem
#
# Here `C3` means six faces with 3 × 3 physical cells each. Each face has three
# vertical layers: `k=1` is the top, `k=3` the surface. Halo cells around a face
# supply neighboring values to advection stencils. Objective and emission
# indices use the **physical** 1:3 range, without a halo offset.
#
# We use artificial 10 kg air masses, zero initial tracer, and upward vertical
# mass flux. Horizontal fluxes and optional physics are zero. The column is
# closed at its top and bottom, so air moves between layers without leaving it.
# These are teaching values, not a realistic meteorological state.

mesh = CubedSphereMesh(Nc=3, Hp=3, FT=Float64);
Nc, N, Nz, nsteps = mesh.Nc, mesh.Nc + 2mesh.Hp, 3, 3;
dt = 1.0;  # seconds per complete transport step
air0 = ntuple(_ -> fill(10.0, N, N, Nz), 6);
tracer0 = ntuple(_ -> zeros(N, N, Nz), 6);
xflux = [ntuple(_ -> zeros(N+1, N, Nz), 6) for _ in 1:nsteps];
yflux = [ntuple(_ -> zeros(N, N+1, Nz), 6) for _ in 1:nsteps];
zflux = [ntuple(_ -> zeros(N, N, Nz+1), 6) for _ in 1:nsteps];
for step in 1:nsteps, p in 1:6
    zflux[step][p][:, :, 2:Nz] .= -0.04
end

# Low-level flux arrays contain air-mass **amounts per directional half-sweep**,
# not winds in m/s and not kg/s. Negative vertical flux is upward. `dt` scales
# the surface-emission rate; it does not convert these already integrated flux
# amounts. A real-data caller obtains consistent fluxes and step schedules from
# the transport binary instead of inventing them.

# ## 2. Choose emissions and a scalar result
#
# State stores `rm = dry mixing ratio × dry air mass`. Accordingly, this API's
# emission arrays contain **model-storage rate per cell**, with no division by
# area and no implicit molecular-weight conversion. An inventory expressed in
# physical kg of a gas per m² per second needs both conversions before use.
# See [State & basis](@ref) and the forward surface-flux builders.
#
# Emit at the center of face 1 during all three steps. The scalar objective
# `J` is the final mixing ratio in the layer immediately above that source.
# `CSLayerMeanObjective` refers to one cell/layer; it does not average an entire
# horizontal layer. Upwind is linear in tracer storage for fixed meteorology,
# making it a clear first gradient check.

rates = [ntuple(_ -> zeros(Nc, Nc), 6) for _ in 1:nsteps];
for step in 1:nsteps
    rates[step][1][2,2] = 0.01
end
objective = CSLayerMeanObjective(1, 2, 2, 2);
scheme = UpwindScheme();

function receptor_value(emissions)
    run_cs_footprint_forward(tracer0, air0, xflux, yflux, zflux, mesh,
        objective; scheme, dt, emission_rates=emissions)
end

J = receptor_value(rates)
@assert isfinite(J) && J > 0
println("Final receptor mixing ratio: ", J, " mol/mol")

# ## 3. Run the reverse pass
#
# For this one objective, a forward recording followed by a reverse pass gives
# a sensitivity for **every** surface cell and emission step. It does not run a
# separate perturbed simulation for every cell. The meteorology is held fixed:
# these are emission sensitivities, not derivatives with respect to winds.

result = cs_surface_emission_footprint(tracer0, air0, xflux, yflux, zflux,
    mesh, objective; scheme, dt, base_emission_rates=rates,
    checkpoint=FullCheckpoint());

@assert length(result.footprints) == nsteps
@assert size(result.footprints[1][1]) == (Nc, Nc)
sensitivities = [result.footprints[t][1][2,2] for t in 1:nsteps]
@assert all(isfinite, sensitivities) && all(>(0), sensitivities)
println("Sensitivity to source rate at each step: ", sensitivities)

# `result.footprints[t][p][i,j]` is `∂J/∂E[t,p,i,j]`, including the source's
# timestep factor. Increasing that rate by a small `δE` predicts a receptor
# change of approximately `footprint × δE`. For many simultaneous changes, sum
# these products over cells and steps. A footprint is a derivative, not a
# probability or an inferred emission map. `lag_steps` counts steps before the
# final step; keep the actual timestamps when interpreting variable step sizes.
# Use `J = receptor_value(rates)` above for the forward receptor mixing ratio.
# The reverse result stores the objective, footprints, lag steps, timestep,
# and initial adjoint.

# ## 4. Check one derivative independently
#
# Change only the first-step source rate, run the forward calculation at both
# perturbations, and compare its centered finite difference with the adjoint.
# All three calls receive identical initial arrays and meteorology; the public
# forward helper copies state before evolving it.

epsilon = 1e-4;
plus, minus = deepcopy(rates), deepcopy(rates);
plus[1][1][2,2] += epsilon;
minus[1][1][2,2] -= epsilon;
finite_difference = (receptor_value(plus) - receptor_value(minus)) / (2epsilon)
adjoint_derivative = result.footprints[1][1][2,2]
@assert isapprox(adjoint_derivative, finite_difference; rtol=1e-8, atol=1e-12)
println("Adjoint: ", adjoint_derivative, "; finite difference: ", finite_difference)
println("FIRST_ADJOINT_PASSED")

# With nonlinear limiters, the derivative is local to the base trajectory.
# Supply `base_emission_rates`, vary the perturbation size, and avoid interpreting
# a finite difference across a limiter switch as the derivative of one branch.
# Do not infer support for every forward scheme from this upwind example.
# [Adjoint status](@ref) lists the supported schemes and physics restrictions.
#
# ## 5. From sensitivity to an inversion
#
# A footprint describes sensitivity. An inversion also needs observations,
# their uncertainties, a prior emission estimate and its covariance, a control
# parameterization, and an optimizer. In 4D-Var the minimized cost combines an
# observation mismatch with departure from the prior. The adjoint supplies the
# cost gradient; it does not by itself choose emissions or quantify uncertainty.
#
# Continue with [Learning adjoints and inversions](@ref Learning-adjoints) for
# a runnable synthetic inversion and an explanation of those pieces.
