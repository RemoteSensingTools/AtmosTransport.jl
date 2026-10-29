# Architectures API


The exported `_neumaier_sum(f, ks, T)` helper returns `(sum, compensation)`;
add the pair to obtain the total. It evaluates terms in the order of `ks`
in precision `T` and is shared with non-core Analysis.

```@autodocs
Modules = [AtmosTransport.Architectures]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
