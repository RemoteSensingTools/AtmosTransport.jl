# Architectures API


Use `array_adapter(arch)` for backend adaptation and
`KernelAbstractions.get_backend(array)` for kernel execution. A grid stores
its execution architecture in `grid.arch`.

The declared shared `_neumaier_sum(f, ks, T)` helper returns `(sum, compensation)`;
add the pair to obtain the total. It evaluates terms in the order of `ks`
in precision `T`. Convection and non-core Analysis import it explicitly;
`src/README.md` declares sharing without adding an export.

```@autodocs
Modules = [AtmosTransport.Architectures]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
