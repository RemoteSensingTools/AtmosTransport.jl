# Architectures API


Use `array_adapter(arch)` for backend adaptation and
`KernelAbstractions.get_backend(array)` for kernel execution. A grid stores
its execution architecture in `grid.arch`.

```@autodocs
Modules = [AtmosTransport.Architectures]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
