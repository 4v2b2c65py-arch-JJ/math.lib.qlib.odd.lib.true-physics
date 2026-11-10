use mathlib::{base_clock_hash_units_per_sec, base_clock_outputs_per_sec, run_year_series};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

fn push_u32_le(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_u64_le(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_i64_le(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_f64_le(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_str(buf: &mut Vec<u8>, s: &str) {
    push_u64_le(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

/// Store benchmark results in a GGUF container.
/// Layout:
/// - "GGUF" magic, version u32=3, tensor_count u64=0, kv_count u64
/// - KV pairs (key string, type, value) with summary + per-row hashes/timings
/// - blob section: little-endian records per output:
///   [year:i64][agent_id:u32][address x,y,z:i64*3][value:i64][hash:u64][elapsed_ns:u64]
fn write_benchmark_gguf(path: &Path, base: i64, years: &[i64]) -> std::io::Result<()> {
    let run = run_year_series(base, years);
    let outs_count = run.grid.rows.iter().map(|r| r.cells.len()).sum::<usize>() as u64;
    let outs_per_sec = base_clock_outputs_per_sec(&run);
    let hash_units_per_sec = base_clock_hash_units_per_sec(&run);
    let total_ns = run.timing.total.as_nanos() as u64;

    // Build blob
    let mut blob: Vec<u8> = Vec::with_capacity((outs_count as usize) * 64);
    for (i, row) in run.grid.rows.iter().enumerate() {
        let year = years[i];
        let elapsed_ns = run.timing.per_row[i].as_nanos() as u64;
        for cell in &row.cells {
            push_i64_le(&mut blob, year);
            push_u32_le(&mut blob, cell.agent_id as u32);
            push_i64_le(&mut blob, cell.address.x);
            push_i64_le(&mut blob, cell.address.y);
            push_i64_le(&mut blob, cell.address.z);
            push_i64_le(&mut blob, cell.value);
            push_u64_le(&mut blob, cell.hash);
            push_u64_le(&mut blob, elapsed_ns);
        }
    }

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Header
    w.write_all(b"GGUF")?;
    push_u32_le_buf(&mut w, 3)?; // version
    push_u64_le_buf(&mut w, 0)?; // tensor_count
    // KV count: general + per-run summary + blob marker
    let kv_count: u64 = 10;
    push_u64_le_buf(&mut w, kv_count)?;

    // KV helper: type 8 = string, 4 = u64, 10 = u32, 11 = i64, 12 = f64
    write_kv_str(&mut w, "general.architecture", "mathlib_times_plank")?;
    write_kv_str(&mut w, "general.name", "mathlib-benchmark")?;
    write_kv_i64(&mut w, "mathlib.base", base)?;
    write_kv_u64(&mut w, "mathlib.years_count", years.len() as u64)?;
    write_kv_u64(&mut w, "mathlib.rows", run.grid.rows.len() as u64)?;
    write_kv_u64(&mut w, "mathlib.outputs", outs_count)?;
    write_kv_u64(&mut w, "mathlib.total_ns", total_ns)?;
    write_kv_f64(&mut w, "mathlib.clock_outputs_per_sec", outs_per_sec)?;
    write_kv_f64(&mut w, "mathlib.clock_hash_units_per_sec", hash_units_per_sec)?;
    // blob as string key marker + raw bytes length and data in a trailing section
    write_kv_u64(&mut w, "mathlib.blob_bytes", blob.len() as u64)?;

    // Trailing blob payload
    w.write_all(&blob)?;
    w.flush()?;
    Ok(())
}

fn push_u32_le_buf(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn push_u64_le_buf(w: &mut impl Write, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn push_i64_le_buf(w: &mut impl Write, v: i64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn push_f64_le_buf(w: &mut impl Write, v: f64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_kv_key(w: &mut impl Write, key: &str) -> std::io::Result<()> {
    push_u64_le_buf(w, key.len() as u64)?;
    w.write_all(key.as_bytes())
}
fn write_kv_str(w: &mut impl Write, key: &str, val: &str) -> std::io::Result<()> {
    write_kv_key(w, key)?;
    push_u32_le_buf(w, 8)?; // GGUF_TYPE_STRING
    push_u64_le_buf(w, val.len() as u64)?;
    w.write_all(val.as_bytes())
}
fn write_kv_u64(w: &mut impl Write, key: &str, val: u64) -> std::io::Result<()> {
    write_kv_key(w, key)?;
    push_u32_le_buf(w, 4)?; // GGUF_TYPE_UINT64
    push_u64_le_buf(w, val)
}
fn write_kv_i64(w: &mut impl Write, key: &str, val: i64) -> std::io::Result<()> {
    write_kv_key(w, key)?;
    push_u32_le_buf(w, 11)?; // GGUF_TYPE_INT64
    push_i64_le_buf(w, val)
}
fn write_kv_f64(w: &mut impl Write, key: &str, val: f64) -> std::io::Result<()> {
    write_kv_key(w, key)?;
    push_u32_le_buf(w, 12)?; // GGUF_TYPE_FLOAT64
    push_f64_le_buf(w, val)
}

fn main() -> std::io::Result<()> {
    // Extensive year/x series: 1..=100, plus 1000, 8000 (timeline base clock)
    let mut years: Vec<i64> = (1..=100).collect();
    years.push(1000);
    years.push(8000);
    let base = 8i64; // base multiplier for year series (e.g. 8x timeline)

    let out_path = Path::new("benchmark.gguf");
    write_benchmark_gguf(out_path, base, &years)?;

    let run = run_year_series(base, &years);
    println!("benchmark written: {}", out_path.display());
    println!("years={} rows={} outputs={}", years.len(), run.grid.rows.len(), run.grid.rows.iter().map(|r| r.cells.len()).sum::<usize>());
    println!("total_ns={} clock_out_per_s={:.3} clock_hash_units_per_s={:.3}",
        run.timing.total.as_nanos(),
        base_clock_outputs_per_sec(&run),
        base_clock_hash_units_per_sec(&run));
    Ok(())
}