# Infrastructure API

These small support modules make physical field semantics explicit, expose
opt-in host/GPU timing instrumentation, and check configuration tables.

## Quantity traits

```@autodocs
Modules = [AtmosTransport.Quantities]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## Section timing

```@autodocs
Modules = [AtmosTransport.SectionTimer]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## Configuration checks

```@autodocs
Modules = [AtmosTransport.ConfigChecks]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
