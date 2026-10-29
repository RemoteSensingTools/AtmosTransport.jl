# Models API


The `Models` module contains the runtime stepper —
`TransportModel`, `DrivenSimulation`, the IC pipeline, and the
recipe builders that turn a TOML config into a fully-wired runtime
object.

Use `step!(model, dt)` for a full model step; driven runs execute their
`StepPlan` phases. `build_initial_mixing_ratio` returns dry VMR and accepts
`surface_pressure`, `pressure_basis`, and `qv` for pressure remapping. Set
`mass_basis` when converting that VMR with `pack_initial_tracer_mass`.

```@autodocs
Modules = [
    AtmosTransport.Models,
    AtmosTransport.Models.InitialConditionIO,
    AtmosTransport.Models.BinaryPathExpander,
    AtmosTransport.Models.InputStaging,
    AtmosTransport.Models.DrivenRunner,
]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
