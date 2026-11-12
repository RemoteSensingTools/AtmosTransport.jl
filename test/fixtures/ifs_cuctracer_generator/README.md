# Generator of the IFS tracer-routine fixture

`../ifs_cuctracer_columns.txt` holds twelve columns and, for six tracers each, the result of one
step of the tracer transport of the IFS convection scheme. The results come from the unmodified
OpenIFS routines `cuctracer.F90` and `cubidiag.F90`, compiled with gfortran. The OpenIFS sources
(Apache-2.0) are not kept here.

To regenerate:

```bash
base=https://raw.githubusercontent.com/ecmwf-ifs/openifs/48c1db541c1839b3b8f55f1b005df1930d29f41b/ifs-source/arpifs/phys_ec
mkdir work && cd work
curl -sLO $base/cuctracer.F90 && curl -sLO $base/cubidiag.F90
touch cubidiag.intfb.h                       # the interface block is not needed
flags="-O0 -cpp -ffree-line-length-none -fcheck=bounds -I."
gfortran $flags -c <this folder>/stubs.F90
gfortran $flags -c cubidiag.F90 cuctracer.F90
gfortran $flags -o ifs_driver stubs.o cubidiag.o cuctracer.o <this folder>/driver.F90
julia <this folder>/generate.jl ./ifs_driver .
```

- `stubs.F90`: the few IFS modules the routines use, with the settings of `sucumf.F90`
  (`RMFSOLCT = 1`, `RMFCMIN = 1e-8`, `RMFADVW = 0`).
- `driver.F90`: reads columns on half levels, calls `CUCTRACER`, writes the tendencies.
- `generate.jl`: builds the columns as layer fields, converts them to the half-level input of the
  routine (mass fluxes from the cumulative entrainment minus detrainment; the downdraft detrainment
  of layer `k` goes on half level `k + 1`), runs the driver and writes the fixture. It also prints
  the largest difference between the Fortran result and `test/helpers/ifs_convection_reference.jl`.

The columns satisfy the plume assumptions of the scheme (detrainment of a layer no larger than the
mass flux entering it, fluxes that close at cloud top and at the ground). Three of them have a cloud
base above the two lowest layers, which exercises the correction of the lowest layer.
