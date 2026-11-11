# Float32 tracer-mass drift of the advection update — diagnostics

The measurements behind "What remains / Advection" in
`docs/src/theory/float32_conservation.md`: in the idealized flows a uniform tracer in `Float32`
drifts by 1e-7 to 2e-6 of its mass in 12 days. These scripts show that the drift is the
rounding of the stored cell update, not face fluxes that fail to cancel, and test two fixes. The
kernels were not changed (owner decision, 2026-11-10).

| File | What it does |
|---|---|
| `linrood_stage_harness.jl` | Runs the production Lin–Rood horizontal step (`fv_tp_2d_cs!`) on the flow-B transfers of `../IdealizedAdvection.jl` for 12 days (C48, four layers, uniform tracer, 2× the substeps of the default case). After every call it recomputes each cell update exactly in `Float64` from the `Float32` face values the kernel used and splits the change of the global mass into the rounding of the stored results and the flux mismatch (net of the face fluxes). Modes `single_rounding` and `carry` store the correctly rounded exact update, or also keep the rounding residual per cell and add it to the next update. |
| `emulate_upwind_core.jl` | Stand-alone emulation of the flux-form upwind sweep (`rm + F_in − F_out`) on a doubly periodic grid with a deformational stream-function flow. |
| `emulate_upwind.jl` | Drift of a uniform tracer in that emulation: time-dependent and steady flow, with and without the window reset of the air mass, three resolutions, and the `Float64` control. |
| `emulate_fixes.jl` | The candidate fixes in the emulation: two roundings (current kernels), one rounding of `F_in − F_out`, one rounding of the exact three-term sum (error-free transformations or `Float64`), and the per-cell carry; uniform field and a hill on a uniform background. |

## Commands (from the repository root)

```bash
julia --project=. -t 8 examples/paper/idealized_advection/float32_drift/linrood_stage_harness.jl c48 uniform Float32 production
julia --project=. -t 8 examples/paper/idealized_advection/float32_drift/linrood_stage_harness.jl c48 uniform Float32 single_rounding
julia --project=. -t 8 examples/paper/idealized_advection/float32_drift/linrood_stage_harness.jl c48 uniform Float32 carry
julia --project=. examples/paper/idealized_advection/float32_drift/emulate_upwind.jl
julia --project=. examples/paper/idealized_advection/float32_drift/emulate_fixes.jl
```

The three harness runs took about 15 minutes together on 8 threads (on a shared machine); the
emulations under a minute each.

## What they print (rerun 2026-11-10, flow B with κ = 2)

Lin–Rood 7, C48, uniform tracer, `Float32`, relative change of the global mass at day 12:

| Mode | Total | Detail |
|---|---|---|
| production | −5.35e-7 | rounding −5.35e-7 (panel-edge cells −7.5e-8, others −4.6e-7); flux mismatch +9e-18 (≤ 3e-17 at every check); mean −0.00145 ulp per cell update |
| single rounding | +1.32e-8 | max \|χ − 1\| 1.1e-6 |
| carry | −2.10e-9 | the carry holds +2.1e-9, i.e. the field plus the carry conserves the mass; max \|χ − 1\| 2.9e-7 |

Emulation of the upwind update (96 × 96, uniform tracer, day 12): current kernels +5.85e-7;
one rounding of `F_in − F_out` −2.55e-7; one rounding of the three-term sum −2.53e-7; carry:
field plus carry −5.5e-12, with −2.9e-10 held in the carry (the stored field alone is at +2.8e-10). A hill on a uniform background: +7.4e-9 (about 80×
less). Steady flow: +5.8e-8, constant after day 6. Without the window reset of the air mass:
+7.8e-7. Resolution 48 / 192: +3.2e-7 / +3.0e-7 (no growth with the number of substeps).
`Float64` control: −1.3e-15.
