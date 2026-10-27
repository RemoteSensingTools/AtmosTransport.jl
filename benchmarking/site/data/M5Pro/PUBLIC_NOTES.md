- Apple M5 Pro MacBook Pro, 64 GB unified memory; Metal, Float32. Same met binaries and protocol as on the
  GPUs (8 simulated days; day 8 replays day 3).
- A laptop in normal use during the runs (load average 2.4–8.2 at run start), so its timings are noisier than
  those of the servers.
- Peak memory is the Metal allocation of the process; Metal gives no power or utilisation readings without
  administrator rights.
- With v0.5.0 at C48 and 1 tracer, *adv* alone is reproducibly slower than *all* (4 runs); the advection
  step itself takes longer when it runs alone. The cause is not established.
