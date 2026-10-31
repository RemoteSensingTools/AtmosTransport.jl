# Preprocessing API


For narrative coverage of the source × target dispatch model and
per-source pipelines, see [Preprocessing overview](@ref) and the
per-source pages [ERA5 spectral path](@ref) and
[GEOS native cubed-sphere](@ref).

```@autodocs
Modules = [AtmosTransport.Preprocessing,
           AtmosTransport.Preprocessing.Core,
           AtmosTransport.Preprocessing.Vertical,
           AtmosTransport.Preprocessing.Spectral,
           AtmosTransport.Preprocessing.MassClosure,
           AtmosTransport.Preprocessing.PhysicsInputs,
           AtmosTransport.Preprocessing.TransportBinary,
           AtmosTransport.Preprocessing.LatLon,
           AtmosTransport.Preprocessing.ReducedGaussian,
           AtmosTransport.Preprocessing.CubedSphere]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
