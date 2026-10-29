- NVIDIA L40S 48 GB in a shared server (Intel Xeon host, met binaries on local NVMe); CUDA, Float32.
- Seven input-only (*none*) runs read their met files cold from disk: the median of days 3–7 was more than
  1.3× the time of day 8, which replays day 3 from the page cache. They were rerun with the grid's binaries
  read into the page cache first; the cold runs are kept in the records with status `cold_page_cache`.
- After a driver update on 26 October 2026 (13:31 PDT) the GPU telemetry was unavailable, so runs from then
  until 27 October have no power or utilisation readings; it was back for the e1f9dfe4c reruns. The timings are
  unaffected; every run confirmed it ran on the L40S.
- C720 with 12 tracers does not fit in 48 GB.
- The tracer-count series (C90 and C180 with 50, 100 and 200 tracers) was rerun on main at e1f9dfe4c. At
  16ea4b3c5 the 200-tracer runs with all operators failed to compile: the decay operator passed its tracer
  list to the GPU kernel as one tuple, which does not compile for 50 decay tracers. Since #99 it launches at
  most 16 decay tracers at a time. The other ten configurations agree with the earlier run within 1 %, although
  the C90 reruns started at a host load of 10 to 13.
