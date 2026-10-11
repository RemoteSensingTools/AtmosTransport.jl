# Infrastructure API

These small support modules make physical field semantics explicit, expose
opt-in host/GPU timing instrumentation, check configuration tables, and
publish files only when they are complete.

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

## Log levels

```@autodocs
Modules = [AtmosTransport.LogLevels]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## Atomic file publication

Every transport-binary writer, the spectral day cache and the regridder cache
write to a staging file next to the target and rename it onto the target only
when the file is complete; a failed write removes the staging file and leaves
a previous target untouched. Publication is one replacing rename (`rename(2)`
on POSIX systems) on every Julia version, so readers see the previous file
until the new one replaces it, and a failed rename never deletes the target.
The staging file is created with `O_EXCL` and mode `0o666`, so published files
get `0o666 & ~umask` (never the `0o600` of `mktemp`).

```@autodocs
Modules = [AtmosTransport.AtomicFiles]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
