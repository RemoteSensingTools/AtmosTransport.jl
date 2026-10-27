- NVIDIA A100 40 GB PCIe in a shared server (AMD EPYC host, met binaries on a spinning-disk RAID); CUDA,
  Float32. Same harness, configurations and code as on the L40S.
- Other jobs occasionally used the same GPU. Runs that overlapped them were found from a process log taken
  every 5 s and rerun once the GPU had been free for 5 minutes; the originals are kept with status
  `foreign_gpu_job`.
- 23 runs, mostly *none* and 1-tracer runs, read their met files cold from disk (days 3–7 up to 3–7× slower
  than day 8, which replays day 3 from the page cache). They were rerun with the binaries read into the page
  cache first; the originals are kept with status `cold_page_cache`. Reruns ran after the main queue, so they
  are not interleaved with the other versions of their pass.
- Host load average at run start was 5.7–21. Host-bound runs (*none*, small grids at 1 tracer) still vary by
  up to 2–3× between passes; compare devices with 12 tracers at C90 and C180.
