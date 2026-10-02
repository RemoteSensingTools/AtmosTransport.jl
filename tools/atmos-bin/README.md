# atmos-bin

Fast, standalone header queries for AtmosTransport `MFLX` binary stores. The
tool reads only the NUL-terminated JSON header, scans directories in parallel,
and does not load Julia or mmap payload arrays.

Build it once from the repository root:

```bash
cargo build --release --manifest-path tools/atmos-bin/Cargo.toml
```

The executable is `tools/atmos-bin/target/release/atmos-bin`.

Common queries:

```bash
# Inspect and validate one binary.
tools/atmos-bin/target/release/atmos-bin inspect FILE.bin --validate

# Summarize every transport binary below a data root.
tools/atmos-bin/target/release/atmos-bin scan ~/data/AtmosTransport/met --validate

# Machine-readable inventory of obsolete formats.
tools/atmos-bin/target/release/atmos-bin scan ~/data/AtmosTransport/met \
    --older-than 4 --format tsv

# Safe input for an explicit downstream command.
tools/atmos-bin/target/release/atmos-bin scan ~/data/AtmosTransport/met \
    --older-than 4 --format paths --null
```

Output columns distinguish legacy layer-centred `kz`, exact TM5 interface
exchange `dkg`, TM5 convection, and CMFMC. With `--validate`, format-v4 files
carrying forbidden legacy `kz`, malformed schedules, incomplete geometry, or
contradictory payload declarations are reported as invalid.

The scanner intentionally has no delete operation. Use `--format paths
--null` only after reviewing a table or TSV manifest, and revalidate targets
immediately before any destructive action.
