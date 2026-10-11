# Operators API


For narrative coverage of the operator hierarchy, the `apply!`
contract, and the Strang palindrome, see [Operators](@ref Operator-concepts).
For per-scheme advection properties, [Advection schemes](@ref).

## `Operators` (top-level)

```@autodocs
Modules = [AtmosTransport.Operators]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## Advection

```@autodocs
Modules = [AtmosTransport.Operators.Advection]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## Convection

```@autodocs
Modules = [AtmosTransport.Operators.Convection]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## Diffusion

```@autodocs
Modules = [AtmosTransport.Operators.Diffusion]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## SurfaceFlux

Source containers, the surface-flux operator, and the emission totals
(`total_emission_rate`, `emission_segments`, `integrated_emission`) used by
logs and budget diagnostics. [Emission totals](@ref) gives the equations.

```@autodocs
Modules = [AtmosTransport.Operators.SurfaceFlux]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## Chemistry

```@autodocs
Modules = [AtmosTransport.Operators.Chemistry]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
