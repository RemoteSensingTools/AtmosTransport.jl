# MetDrivers API


For narrative coverage of the transport-binary format and the
replay-gate contract, see [Binary format](../concepts/binary_format.md).

```@autodocs
Modules = [AtmosTransport.MetDrivers,
           AtmosTransport.MetDrivers.BinaryFormat,
           AtmosTransport.MetDrivers.BinaryReaders]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

Native ERA5 GRIB geometry readers are documented under `Preprocessing`; runtime
binary formats and drivers are owned by the two submodules listed above.
