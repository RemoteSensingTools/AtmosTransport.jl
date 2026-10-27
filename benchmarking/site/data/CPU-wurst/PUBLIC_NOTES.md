- The host of the L40S server, 32 Julia threads, no GPU; Float32. Reduced matrix: C24–C90, *adv* and *all*,
  one pass.
- The server was shared: the load average at run start was about 7 (v0.5.0), 23 (v0.6.0) and 12 (main), so
  the v0.6.0 CPU timings are the most affected.
