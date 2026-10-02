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

```@autodocs
Modules = [AtmosTransport.Visualization]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
