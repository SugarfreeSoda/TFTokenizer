//! Rust core of TFTokenizer: stream motif hits, merge them into occupancy sites, and sum sites into bin features.
//! A site record is four little-endian u32: start, end, motif index, per-base weight (f32 bits). Sites are written sorted by end.
use flate2::read::MultiGzDecoder;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::ffi::{c_char, CStr};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::slice;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SCORES: usize = 1001;

fn text<'a>(p: *const c_char) -> Result<&'a str> {
    Ok(unsafe { CStr::from_ptr(p) }.to_str()?)
}

fn report(result: Result<()>, error: *mut u8, capacity: usize) -> i32 {
    let Err(e) = result else { return 0 };
    let message = e.to_string();
    let n = message.len().min(capacity - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(message.as_ptr(), error, n);
        *error.add(n) = 0;
    }
    -1
}

fn int(field: &[u8]) -> Result<u32> {
    if field.is_empty() || !field.iter().all(u8::is_ascii_digit) {
        return Err(format!("not an integer: {}", String::from_utf8_lossy(field)).into());
    }
    Ok(field.iter().fold(0u32, |v, d| v * 10 + (d - b'0') as u32))
}

/// Score bin: floor(max(score, 0)), capped at 1000.
fn score_bin(field: &[u8]) -> usize {
    if field.first() == Some(&b'-') {
        return 0;
    }
    let digits = field.iter().take_while(|d| d.is_ascii_digit());
    digits.fold(0usize, |v, d| (v * 10 + (d - b'0') as usize).min(SCORES - 1))
}

/// Calls `f(start, end, motif, score_bin)` for every hit of `chrom` in a (gzipped) TSV with a UCSC JASPAR header.
fn each_hit(path: &str, chrom: &str, mut f: impl FnMut(u32, u32, &[u8], usize) -> Result<()>) -> Result<()> {
    let file = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut input: Box<dyn BufRead> = if path.ends_with(".gz") {
        Box::new(BufReader::with_capacity(1 << 20, MultiGzDecoder::new(file)))
    } else {
        Box::new(file)
    };
    let mut line = Vec::new();
    input.read_until(b'\n', &mut line)?;
    let header: Vec<&[u8]> = line.trim_ascii_end().strip_prefix(b"#").unwrap_or(line.trim_ascii_end()).split(|&b| b == b'\t').collect();
    let column = |name: &str| header.iter().position(|h| *h == name.as_bytes()).ok_or(format!("missing column {name}"));
    let index = [column("chrom")?, column("chromStart")?, column("chromEnd")?, column("name")?, column("score")?];
    let last = *index.iter().max().unwrap();
    let mut line = Vec::with_capacity(256);
    while { line.clear(); input.read_until(b'\n', &mut line)? > 0 } {
        let mut fields: [&[u8]; 5] = [b""; 5];
        for (i, field) in line.trim_ascii_end().split(|&b| b == b'\t').enumerate().take(last + 1) {
            if let Some(k) = index.iter().position(|&j| j == i) {
                fields[k] = field;
            }
        }
        if fields[0] == chrom.as_bytes() {
            f(int(fields[1])?, int(fields[2])?, fields[3], score_bin(fields[4]))?;
        }
    }
    Ok(())
}

/// Hit counts per motif and score bin. Motif names are written '\n'-joined in order of first appearance.
#[no_mangle]
pub extern "C" fn tt_histogram(hits: *const c_char, chrom: *const c_char, max_motifs: usize, counts: *mut u64,
                               names: *mut u8, names_capacity: usize, n_motifs: *mut usize, error: *mut u8, capacity: usize) -> i32 {
    report((|| {
        let counts = unsafe { slice::from_raw_parts_mut(counts, max_motifs * SCORES) };
        let mut order: Vec<Vec<u8>> = Vec::new();
        let mut index: HashMap<Vec<u8>, usize> = HashMap::new();
        each_hit(text(hits)?, text(chrom)?, |_, _, motif, score| {
            let k = match index.get(motif) {
                Some(&k) => k,
                None if order.len() < max_motifs => {
                    index.insert(motif.to_vec(), order.len());
                    order.push(motif.to_vec());
                    order.len() - 1
                }
                None => return Err("too many motifs".into()),
            };
            counts[k * SCORES + score] += 1;
            Ok(())
        })?;
        let joined = order.join(&b'\n');
        if joined.len() >= names_capacity {
            return Err("motif names exceed buffer".into());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(joined.as_ptr(), names, joined.len());
            *names.add(joined.len()) = 0;
            *n_motifs = order.len();
        }
        Ok(())
    })(), error, capacity)
}

struct Sink {
    out: BufWriter<File>,
    theta_max: f64,
    count: u64,
    longest: u32,
    max_end: u32,
}

impl Sink {
    /// Site occupancy θ = Σa / (1 + Σa) (capped), load H = −ln(1 − θ), written as per-base weight H / length.
    fn close(&mut self, motif: usize, (start, end, total): (u32, u32, f64)) -> Result<()> {
        let theta = (total / (1.0 + total)).min(self.theta_max);
        let load = -(-theta).ln_1p();
        if load <= 0.0 {
            return Ok(());
        }
        let w = (load / (end - start) as f64) as f32;
        for v in [start, end, motif as u32, w.to_bits()] {
            self.out.write_all(&v.to_le_bytes())?;
        }
        self.count += 1;
        self.longest = self.longest.max(end - start);
        self.max_end = self.max_end.max(end);
        Ok(())
    }
}

/// Merges overlapping hits of the same motif (both strands) into sites and writes them sorted by end.
/// `table[m * 1001 + score]` is the relative binding strength a of motif m. Also returns the longest site and the last hit end.
/// Hits must be sorted by chromStart (bigBedToBed output is).
#[no_mangle]
pub extern "C" fn tt_scan(hits: *const c_char, chrom: *const c_char, motifs: *const c_char, table: *const f32, theta_max: f64,
                          sites: *const c_char, n_sites: *mut u64, max_len: *mut u32, max_end: *mut u32,
                          error: *mut u8, capacity: usize) -> i32 {
    report((|| {
        let names: HashMap<&[u8], usize> = text(motifs)?.as_bytes().split(|&b| b == b'\n').enumerate().map(|(i, m)| (m, i)).collect();
        let table = unsafe { slice::from_raw_parts(table, names.len() * SCORES) };
        let mut sink = Sink { out: BufWriter::with_capacity(1 << 22, File::create(text(sites)?)?), theta_max, count: 0, longest: 0, max_end: 0 };
        let mut open = vec![(0u32, 0u32, 0f64); names.len()];        // open site per motif; end = 0 means none
        let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();   // (end, motif); stale entries are skipped
        let mut previous = 0;
        each_hit(text(hits)?, text(chrom)?, |start, end, motif, score| {
            if start < previous || start >= end {
                return Err(format!("hits must be sorted by chromStart (at {start}-{end})").into());
            }
            previous = start;
            let Some(&m) = names.get(motif) else { return Ok(()) };
            while let Some(&Reverse((e, k))) = heap.peek() {
                if e > start {
                    break;
                }
                heap.pop();
                if open[k].1 == e {
                    sink.close(k, open[k])?;
                    open[k].1 = 0;
                }
            }
            let a = table[m * SCORES + score] as f64;
            let site = &mut open[m];
            if site.1 > start {
                site.2 += a;
                if end > site.1 {
                    site.1 = end;
                    heap.push(Reverse((end, m)));
                }
            } else {
                *site = (start, end, a);
                heap.push(Reverse((end, m)));
            }
            Ok(())
        })?;
        while let Some(Reverse((e, k))) = heap.pop() {
            if open[k].1 == e {
                sink.close(k, open[k])?;
                open[k].1 = 0;
            }
        }
        sink.out.flush()?;
        unsafe {
            *n_sites = sink.count;
            *max_len = sink.longest;
            *max_end = sink.max_end;
        }
        Ok(())
    })(), error, capacity)
}

/// Summed per-base weight of all sites: out[x] = Σ weight of the sites covering base x.
#[no_mangle]
pub extern "C" fn tt_density(sites: *const [u32; 4], n_sites: usize, length: usize, out: *mut f64) {
    let sites = unsafe { slice::from_raw_parts(sites, n_sites) };
    let out = unsafe { slice::from_raw_parts_mut(out, length) };
    out.fill(0.0);
    for s in sites {
        let w = f32::from_bits(s[3]) as f64;
        out[s[0] as usize] += w;
        if (s[1] as usize) < length {
            out[s[1] as usize] -= w;
        }
    }
    let mut total = 0.0;
    for v in out.iter_mut() {
        total += *v;
        *v = total;
    }
}

/// Bin features: out[b, m] = Σ over sites of motif m of weight × overlap with bin b (= Σ site H × fraction inside the bin).
#[no_mangle]
pub extern "C" fn tt_features(sites: *const [u32; 4], n_sites: usize, max_len: u32, edges: *const i64, n_edges: usize,
                              n_motifs: usize, out: *mut f64) {
    let sites = unsafe { slice::from_raw_parts(sites, n_sites) };
    let edges = unsafe { slice::from_raw_parts(edges, n_edges) };
    let out = unsafe { slice::from_raw_parts_mut(out, (n_edges - 1) * n_motifs) };
    let rows = (n_edges - 1).div_ceil(rayon::current_num_threads()).max(1);
    out.par_chunks_mut(rows * n_motifs).enumerate().for_each(|(i, block)| {
        block.fill(0.0);
        let local = &edges[i * rows..=(i * rows + block.len() / n_motifs)];
        let (lo, hi) = (local[0], *local.last().unwrap());
        let first = sites.partition_point(|s| s[1] as i64 <= lo);
        for s in &sites[first..] {
            let (start, end) = (s[0] as i64, s[1] as i64);
            if end - max_len as i64 >= hi {
                break;
            }
            if start >= hi {
                continue;
            }
            let (m, w) = (s[2] as usize, f32::from_bits(s[3]) as f64);
            let mut b = local.partition_point(|&x| x <= start).saturating_sub(1);
            while b + 1 < local.len() && local[b] < end {
                block[b * n_motifs + m] += (end.min(local[b + 1]) - start.max(local[b])) as f64 * w;
                b += 1;
            }
        }
    });
}
