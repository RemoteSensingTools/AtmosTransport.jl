# Output and visualization API

`Output` defines snapshot capture and persistence, and observation sampling
(`[output.observations]`: point events and station series sampled at
met-window ends). `Visualization` provides
topology-aware field views and optional Makie rendering for those snapshots.

## Output

```@autodocs
Modules = [AtmosTransport.Output]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## Visualization

`AtmosTransport.Visualization` lives in `src/Extras/Visualization/`. It is
non-core, has no tests, and is intended to become its own package. The Makie
extension stays in `ext/AtmosTransportMakieExt.jl`, Julia's extension location.
Core modules never import Visualization.

Field selection uses typed values. Pass `ColumnMean()`, `ColumnSum()`,
`LevelSlice()`, or `SurfaceSlice()` as the `transform` keyword to `fieldview`;
`LevelSlice()` also requires `level`. Pass `NativeUnit()` or `PPMUnit()` as
`unit`, and pass `AllFrames()` to `frame_indices` to select every frame. These
selectors are exported by `AtmosTransport.Visualization`, not by the top-level
module. For example:

```julia
using AtmosTransport.Visualization
snapshot = open_snapshot("output.nc")
surface = fieldview(snapshot, :co2; transform=SurfaceSlice(), time=1,
                    unit=PPMUnit())
levels = fieldview(snapshot, :co2; transform=LevelSlice(), level=10, time=1)
```

```@autodocs
Modules = [AtmosTransport.Visualization]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
