- NVIDIA L40S 48 GB in a shared server (Intel Xeon host, met binaries on local NVMe); CUDA, Float32.
- Seven input-only (*none*) runs read their met files cold from disk: the median of days 3–7 was more than
  1.3× the time of day 8, which replays day 3 from the page cache. They were rerun with the grid's binaries
  read into the page cache first; the cold runs are kept in the records with status `cold_page_cache`.
- After a driver update on 26 October 2026 (13:31 PDT) the GPU telemetry was unavailable, so later runs
  have no power or utilisation readings. The timings are unaffected; every run confirmed it ran on the L40S.
- C720 with 12 tracers does not fit in 48 GB.
- 200 tracers with all operators failed to compile: the decay operator passes its tracer list to the GPU kernel
  as a tuple, which does not compile for 50 decay tracers. 100 tracers (25 decay tracers) work.
