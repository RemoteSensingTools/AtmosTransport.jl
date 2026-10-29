# Grids API


For narrative coverage of how the three mesh types fit together,
see the narrative [Grids](@ref) page under Concepts.

The exported `_default_cs_definition(convention)` returns the convention's
standard cubed-sphere definition (equiangular for gnomonic panels, GMAO for
GEOS-native panels). The non-core snapshot reader uses this public helper.

```@autodocs
Modules = [AtmosTransport.Grids]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
