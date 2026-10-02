use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const READ_CHUNK: usize = 256 * 1024;
const MAX_HEADER: usize = 16 * 1024 * 1024;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect one transport binary header.
    Inspect {
        path: PathBuf,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        validate: bool,
    },
    /// Recursively scan files/directories for MFLX transport binaries.
    Scan {
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        /// Return only headers with format_version lower than this value.
        #[arg(long)]
        older_than: Option<i64>,
        /// Return only headers with exactly this format version.
        #[arg(long)]
        version: Option<i64>,
        /// Validate timing, geometry, and payload contracts.
        #[arg(long)]
        validate: bool,
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
        /// NUL-terminate path output for safe xargs/find pipelines.
        #[arg(long)]
        null: bool,
        /// Include invalid/unreadable candidates in table/TSV/JSONL output.
        #[arg(long)]
        show_errors: bool,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Table,
    Tsv,
    Jsonl,
    Paths,
}

#[derive(Debug, Serialize)]
struct Record {
    path: PathBuf,
    bytes: u64,
    format_version: i64,
    grid_type: String,
    resolution: Option<i64>,
    nlevel: Option<i64>,
    float_type: String,
    has_kz: bool,
    has_dkg: bool,
    has_tm5_convection: bool,
    has_cmfmc: bool,
    valid: bool,
    errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ScanError {
    path: PathBuf,
    error: String,
}

enum ScanResult {
    Transport(Record),
    NotTransport,
    Error(ScanError),
}

fn read_json_header(path: &Path) -> Result<(Value, u64), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let bytes = file.metadata().map_err(|e| e.to_string())?.len();
    let mut reader = BufReader::with_capacity(READ_CHUNK, file);
    let mut raw = Vec::with_capacity(READ_CHUNK);
    let mut chunk = vec![0_u8; READ_CHUNK];

    while raw.len() < MAX_HEADER {
        let want = (MAX_HEADER - raw.len()).min(READ_CHUNK);
        let n = reader.read(&mut chunk[..want]).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("JSON header has no NUL terminator".into());
        }
        if let Some(pos) = chunk[..n].iter().position(|b| *b == 0) {
            raw.extend_from_slice(&chunk[..pos]);
            break;
        }
        raw.extend_from_slice(&chunk[..n]);
    }
    if raw.is_empty() || raw.len() >= MAX_HEADER {
        return Err("empty or oversized JSON header".into());
    }
    if raw[0] != b'{' {
        return Err("not a JSON-header binary".into());
    }
    let value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    Ok((value, bytes))
}

fn string_field(header: &Value, key: &str) -> String {
    header
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned()
}

fn payload_sections(header: &Value) -> BTreeSet<String> {
    header
        .get("payload_sections")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

fn validate_header(header: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let required = [
        "magic",
        "format_version",
        "header_bytes",
        "float_type",
        "grid_type",
        "horizontal_topology",
        "ncell",
        "nface_h",
        "nlevel",
        "nwindow",
        "steps_per_window",
        "steps_per_window_by_window",
        "payload_sections",
        "source_flux_sampling",
        "air_mass_sampling",
        "flux_sampling",
        "flux_kind",
        "humidity_sampling",
        "delta_semantics",
        "mass_basis",
    ];
    for key in required {
        if header.get(key).is_none() {
            errors.push(format!("missing {key}"));
        }
    }
    if header.get("magic").and_then(Value::as_str) != Some("MFLX") {
        errors.push("magic is not MFLX".into());
    }
    let nwindow = header.get("nwindow").and_then(Value::as_i64);
    let steps = header.get("steps_per_window").and_then(Value::as_i64);
    let schedule = header
        .get("steps_per_window_by_window")
        .and_then(Value::as_array);
    if let (Some(nw), Some(spw), Some(sched)) = (nwindow, steps, schedule) {
        if sched.len() as i64 != nw {
            errors.push(format!("schedule length {} != nwindow {nw}", sched.len()));
        }
        let vals: Option<Vec<i64>> = sched.iter().map(Value::as_i64).collect();
        match vals {
            Some(vals) if vals.iter().all(|v| *v > 0) => {
                if vals.iter().max().copied() != Some(spw) {
                    errors.push("steps_per_window != maximum(schedule)".into());
                }
            }
            _ => errors.push("schedule contains non-positive/non-integer values".into()),
        }
    }
    let grid = string_field(header, "grid_type").to_ascii_lowercase();
    if grid == "cubed_sphere" {
        for key in [
            "Nc",
            "npanel",
            "panel_convention",
            "cs_definition",
            "cs_coordinate_law",
            "cs_center_law",
        ] {
            if header.get(key).is_none() {
                errors.push(format!("missing cubed-sphere geometry field {key}"));
            }
        }
    }
    let sections = payload_sections(header);
    if sections.contains("kz") && sections.contains("dkg") {
        errors.push("payload contains both legacy kz and exact dkg".into());
    }
    if header.get("format_version").and_then(Value::as_i64) == Some(4) && sections.contains("kz") {
        errors.push("format v4 forbids legacy kz; exact dkg is required".into());
    }
    if sections.contains("dkg") && string_field(header, "mass_basis").to_ascii_lowercase() != "dry"
    {
        errors.push("dkg requires dry mass_basis".into());
    }
    errors
}

fn scan_one(path: &Path, validate: bool) -> ScanResult {
    let (header, bytes) = match read_json_header(path) {
        Ok(x) => x,
        Err(error) => {
            return if error == "not a JSON-header binary" || error.starts_with("expected value") {
                ScanResult::NotTransport
            } else {
                ScanResult::Error(ScanError {
                    path: path.to_owned(),
                    error,
                })
            };
        }
    };
    if header.get("magic").and_then(Value::as_str) != Some("MFLX") {
        return ScanResult::NotTransport;
    }
    let sections = payload_sections(&header);
    let errors = if validate {
        validate_header(&header)
    } else {
        Vec::new()
    };
    let grid_type = string_field(&header, "grid_type");
    let resolution = if grid_type.eq_ignore_ascii_case("cubed_sphere") {
        header.get("Nc").and_then(Value::as_i64)
    } else {
        header.get("Nx").and_then(Value::as_i64)
    };
    ScanResult::Transport(Record {
        path: path.to_owned(),
        bytes,
        format_version: header
            .get("format_version")
            .and_then(Value::as_i64)
            .unwrap_or(-1),
        grid_type,
        resolution,
        nlevel: header.get("nlevel").and_then(Value::as_i64),
        float_type: string_field(&header, "float_type"),
        has_kz: sections.contains("kz"),
        has_dkg: sections.contains("dkg"),
        has_tm5_convection: sections.contains("entu")
            && sections.contains("detu")
            && sections.contains("entd")
            && sections.contains("detd"),
        has_cmfmc: sections.contains("cmfmc"),
        valid: errors.is_empty(),
        errors,
    })
}

fn candidates(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for path in paths {
        if path.is_file() {
            out.push(path.clone());
        } else if path.is_dir() {
            out.extend(
                WalkDir::new(path)
                    .follow_links(false)
                    .into_iter()
                    .filter_map(Result::ok)
                    .filter(|e| e.file_type().is_file())
                    .map(|e| e.into_path())
                    .filter(|p| p.extension().is_some_and(|x| x == "bin")),
            );
        }
    }
    out.sort();
    out.dedup();
    out
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

fn print_records(records: &[Record], format: OutputFormat, null: bool) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    match format {
        OutputFormat::Paths => {
            for r in records {
                out.write_all(r.path.as_os_str().as_encoded_bytes())?;
                out.write_all(if null { b"\0" } else { b"\n" })?;
            }
        }
        OutputFormat::Jsonl => {
            for r in records {
                serde_json::to_writer(&mut out, r)?;
                out.write_all(b"\n")?;
            }
        }
        OutputFormat::Tsv => {
            writeln!(
                out,
                "version\tbytes\tgrid\tresolution\tnlevel\tfloat\tkz\tdkg\ttm5conv\tcmfmc\tvalid\tpath"
            )?;
            for r in records {
                writeln!(
                    out,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    r.format_version,
                    r.bytes,
                    r.grid_type,
                    r.resolution.map_or("".into(), |x| x.to_string()),
                    r.nlevel.map_or("".into(), |x| x.to_string()),
                    r.float_type,
                    r.has_kz,
                    r.has_dkg,
                    r.has_tm5_convection,
                    r.has_cmfmc,
                    r.valid,
                    r.path.display()
                )?;
            }
        }
        OutputFormat::Table => {
            writeln!(
                out,
                "VER  SIZE         GRID          RES  LEV  KZ DKG TM5 CMF OK  PATH"
            )?;
            for r in records {
                writeln!(
                    out,
                    "{:<4} {:<12} {:<13} {:<4} {:<4} {:<2} {:<3} {:<3} {:<3} {:<3} {}",
                    r.format_version,
                    human_bytes(r.bytes),
                    r.grid_type,
                    r.resolution.map_or("-".into(), |x| x.to_string()),
                    r.nlevel.map_or("-".into(), |x| x.to_string()),
                    if r.has_kz { "Y" } else { "-" },
                    if r.has_dkg { "Y" } else { "-" },
                    if r.has_tm5_convection { "Y" } else { "-" },
                    if r.has_cmfmc { "Y" } else { "-" },
                    if r.valid { "Y" } else { "N" },
                    r.path.display()
                )?;
                for error in &r.errors {
                    writeln!(out, "     ! {error}")?;
                }
            }
            let bytes: u64 = records.iter().map(|r| r.bytes).sum();
            writeln!(out, "\n{} binaries, {}", records.len(), human_bytes(bytes))?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect {
            path,
            json,
            validate,
        } => {
            let (header, _) = read_json_header(&path).map_err(io::Error::other)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&header)?);
            } else {
                match scan_one(&path, validate) {
                    ScanResult::Transport(record) => {
                        print_records(&[record], OutputFormat::Table, false)?
                    }
                    ScanResult::NotTransport => return Err("not an MFLX transport binary".into()),
                    ScanResult::Error(e) => return Err(e.error.into()),
                }
            }
        }
        Command::Scan {
            paths,
            older_than,
            version,
            validate,
            format,
            null,
            show_errors,
        } => {
            if null && format != OutputFormat::Paths {
                return Err("--null requires --format paths".into());
            }
            let files = candidates(&paths);
            let results: Vec<_> = files.par_iter().map(|p| scan_one(p, validate)).collect();
            let mut records = Vec::new();
            let mut errors = Vec::new();
            for result in results {
                match result {
                    ScanResult::Transport(r)
                        if older_than.is_none_or(|v| r.format_version < v)
                            && version.is_none_or(|v| r.format_version == v) =>
                    {
                        records.push(r)
                    }
                    ScanResult::Error(e) => errors.push(e),
                    _ => {}
                }
            }
            records.sort_by(|a, b| a.path.cmp(&b.path));
            print_records(&records, format, null)?;
            if show_errors && format != OutputFormat::Paths {
                for e in errors {
                    eprintln!("{}: {}", e.path.display(), e.error);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_header;
    use serde_json::json;

    fn valid_v4_header() -> serde_json::Value {
        json!({
            "magic": "MFLX", "format_version": 4, "header_bytes": 131072,
            "float_type": "Float32", "grid_type": "cubed_sphere",
            "horizontal_topology": "StructuredDirectional", "ncell": 24,
            "nface_h": 48, "nlevel": 2, "nwindow": 2,
            "steps_per_window": 3, "steps_per_window_by_window": [2, 3],
            "payload_sections": ["m", "am", "bm", "cm", "dkg"],
            "source_flux_sampling": "window_start_endpoint",
            "air_mass_sampling": "window_start_endpoint",
            "flux_sampling": "window_constant", "flux_kind": "substep_mass_amount",
            "humidity_sampling": "none", "delta_semantics": "none",
            "mass_basis": "dry", "Nc": 2, "npanel": 6,
            "panel_convention": "geos_native", "cs_definition": "gmao_equal_distance",
            "cs_coordinate_law": "gmao_equal_distance_gnomonic",
            "cs_center_law": "four_corner_normalized"
        })
    }

    #[test]
    fn accepts_current_v4_dkg_contract() {
        assert!(validate_header(&valid_v4_header()).is_empty());
    }

    #[test]
    fn rejects_legacy_kz_in_v4() {
        let mut header = valid_v4_header();
        header["payload_sections"] = json!(["m", "am", "bm", "cm", "kz"]);
        let errors = validate_header(&header);
        assert!(errors.iter().any(|e| e.contains("v4 forbids legacy kz")));
    }

    #[test]
    fn rejects_inconsistent_schedule() {
        let mut header = valid_v4_header();
        header["steps_per_window"] = json!(4);
        let errors = validate_header(&header);
        assert!(errors.iter().any(|e| e.contains("maximum(schedule)")));
    }
}
