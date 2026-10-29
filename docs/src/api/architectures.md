# Architectures API


Use `array_adapter(arch)` for backend adaptation and
`KernelAbstractions.get_backend(array)` for kernel execution. A grid stores
its execution architecture in `grid.arch`.

The exported `_neumaier_sum(f, ks, T)` helper returns `(sum, compensation)`;
add the pair to obtain the total. It evaluates terms in the order of `ks`
in precision `T` and is shared with non-core Analysis.

```@autodocs
Modules = [AtmosTransport.Architectures]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
