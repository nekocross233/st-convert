//! RWKV `.pth` (torch zip checkpoint) -> safetensors converter.
//!
//! Output is byte-identical to the Python reference implementation
//! (`safetensors.torch.save_file` layout: header JSON padded to 8 bytes,
//! tensors sorted by key, fp16 payload).
//!
//! Pipeline: a reader thread pulls big sequential chunks out of the checkpoint,
//! N worker threads convert bf16/f32 -> f16 and write each tensor straight to
//! its final offset.

mod half;
mod pickle;
mod sysio;
mod zip;

use pickle::{Tensor, Val};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use sysio::{free_memory_bytes, pread, pwrite};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DType {
    F16,
    BF16,
    F32,
}

impl DType {
    fn size(self) -> u64 {
        match self {
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
        }
    }
}

struct Args {
    input: String,
    output: String,
    batch_mb: u64,
    threads: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut input = None;
    let mut output = None;
    let mut batch_mb = 0u64;
    let mut threads = 0usize;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--input" => {
                i += 1;
                input = Some(argv.get(i).cloned().ok_or("--input needs a value")?);
            }
            "--output" => {
                i += 1;
                output = Some(argv.get(i).cloned().ok_or("--output needs a value")?);
            }
            "--batch-mb" => {
                i += 1;
                batch_mb = argv.get(i).ok_or("--batch-mb needs a value")?.parse().map_err(|_| "bad --batch-mb")?;
            }
            "--threads" => {
                i += 1;
                threads = argv.get(i).ok_or("--threads needs a value")?.parse().map_err(|_| "bad --threads")?;
            }
            "-h" | "--help" => {
                println!(
                    "usage: st-convert --input model.pth --output model.st [--batch-mb N] [--threads N]\n\
                     \n\
                       --batch-mb   0 (default) sizes the read buffer from free memory\n\
                       --threads    0 (default) uses min(8, cores)"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
        i += 1;
    }
    Ok(Args {
        input: input.ok_or("missing --input")?,
        output: output.ok_or("missing --output")?,
        batch_mb,
        threads,
    })
}

#[derive(Clone)]
struct PlanItem {
    out_name: String,
    out_off: u64,
    src_off: u64,
    dtype: DType,
    dims: Vec<u64>,
    n_elem: u64,
    transpose: bool,
}

struct Job {
    out_off: u64,
    rel: u64,
    n_elem: u64,
    dtype: DType,
    dims: Vec<u64>,
    transpose: bool,
}

struct Batch {
    start: u64,
    span: u64,
    jobs: Vec<Job>,
}

fn rename_key(name: &str) -> String {
    let mut n = name.to_string();
    for (from, to) in [
        ("time_faaaa", "time_first"),
        ("time_maa", "time_mix"),
        ("lora_A", "lora.0"),
        ("lora_B", "lora.1"),
    ] {
        if n.contains(from) {
            n = n.replace(from, to);
        }
    }
    n.to_lowercase()
}

const TRANSPOSE_NAMES: [&str; 14] = [
    "time_mix_w1",
    "time_mix_w2",
    "time_decay_w1",
    "time_decay_w2",
    "w1",
    "w2",
    "a1",
    "a2",
    "g1",
    "g2",
    "v1",
    "v2",
    "time_state",
    "lora.0",
];

fn is_transposed(name: &str) -> bool {
    TRANSPOSE_NAMES.iter().any(|t| name.contains(t))
}

fn product(v: &[u64]) -> u64 {
    v.iter().product()
}

fn dtype_of(name: &str) -> Result<DType, String> {
    if name.contains("BFloat16") {
        Ok(DType::BF16)
    } else if name.contains("Half") {
        Ok(DType::F16)
    } else if name.contains("Float") {
        Ok(DType::F32)
    } else {
        Err(format!("unsupported storage dtype {name}"))
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out
}

fn convert(job: &Job, src: &[u8]) -> Vec<u8> {
    let n = job.n_elem as usize;
    let mut words: Vec<u16> = Vec::with_capacity(n);
    match job.dtype {
        DType::F16 => {
            words.extend(src[..n * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
        }
        DType::BF16 => {
            for c in src[..n * 2].chunks_exact(2) {
                words.push(half::bf16_to_f16_bits(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        DType::F32 => {
            for c in src[..n * 4].chunks_exact(4) {
                words.push(half::f32_to_f16_bits(f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
            }
        }
    }

    if job.transpose && job.dims.len() >= 2 {
        // job.dims is the *output* shape, so the input is (rows=r, cols=c)
        // with output (c, r): out[j*r + i] = in[i*c + j].
        let n_dim = job.dims.len();
        let r = job.dims[n_dim - 1] as usize;
        let c = job.dims[n_dim - 2] as usize;
        let plane = r * c;
        let planes = n / plane.max(1);
        let mut t = vec![0u16; n];
        for p in 0..planes {
            let base = p * plane;
            for i in 0..r {
                for j in 0..c {
                    t[base + j * r + i] = words[base + i * c + j];
                }
            }
        }
        words = t;
    }

    let mut bytes = Vec::with_capacity(n * 2);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

fn read_state_dict(v: &Val) -> Result<Vec<(String, Tensor)>, String> {
    let dict = match v {
        Val::Dict(d) => d,
        _ => return Err("top-level pickle value is not a dict".into()),
    };
    let mut d = dict;
    if let Some((_, inner)) = dict.iter().find(|(k, _)| matches!(k, Val::Str(s) if s == "state_dict")) {
        if let Val::Dict(inner) = inner {
            d = inner;
        }
    }
    let mut out = Vec::new();
    for (k, v) in d {
        let name = match k {
            Val::Str(s) => s.clone(),
            _ => continue,
        };
        match v {
            Val::Tensor(t) => out.push((name, t.clone())),
            _ => return Err(format!("entry {name} is not a tensor")),
        }
    }
    Ok(out)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let input = File::open(&args.input).map_err(|e| format!("open {}: {e}", args.input))?;
    let file_len = input.metadata().map_err(|e| e.to_string())?.len();

    let entries = zip::read_dir(&input, file_len).map_err(|e| format!("read zip directory: {e}"))?;
    let pkl = entries
        .iter()
        .find(|e| e.name.ends_with("data.pkl"))
        .ok_or("checkpoint has no data.pkl")?;
    let prefix = pkl.name[..pkl.name.len() - "data.pkl".len()].to_string();
    let pkl_bytes = zip::read_entry(&input, pkl).map_err(|e| format!("read data.pkl: {e}"))?;

    let root = pickle::parse(&pkl_bytes)?;
    let tensors = read_state_dict(&root)?;
    println!("checkpoint tensors: {}", tensors.len());
    if tensors.iter().any(|(n, _)| n.contains("gate.weight")) {
        return Err(
            "v5.1 checkpoint (gate.weight): the time_decay/time_faaaa broadcast expansion is not implemented"
                .into(),
        );
    }

    // storage key -> (payload offset, byte size)
    let mut storages: HashMap<String, (u64, u64)> = HashMap::new();
    for e in &entries {
        if let Some(key) = e.name.strip_prefix(&format!("{prefix}data/")) {
            let off = zip::payload_offset(&input, e).map_err(|err| err.to_string())?;
            storages.insert(key.to_string(), (off, e.size));
        }
    }

    // ---- build the output plan -----------------------------------------
    let mut plan: Vec<PlanItem> = Vec::with_capacity(tensors.len());
    for (src_name, t) in &tensors {
        let dtype = dtype_of(&t.dtype)?;
        let (payload, storage_bytes) = *storages
            .get(&t.key)
            .ok_or_else(|| format!("storage {} for {src_name} not found in the archive", t.key))?;
        let n_elem: u64 = product(&t.size.iter().map(|&x| x as u64).collect::<Vec<_>>());
        if t.offset < 0 {
            return Err(format!("negative storage offset for {src_name}"));
        }
        let src_off = payload + t.offset as u64 * dtype.size();
        if (t.offset as u64 + n_elem) * dtype.size() > storage_bytes {
            return Err(format!("tensor {src_name} points outside its storage"));
        }
        // only dense, row-major tensors are supported (true for RWKV checkpoints)
        let mut expect_stride = 1i64;
        for k in (0..t.size.len()).rev() {
            if t.size[k] == 0 {
                continue;
            }
            if t.stride[k] != expect_stride {
                return Err(format!("tensor {src_name} is not contiguous (stride {:?})", t.stride));
            }
            expect_stride *= t.size[k];
        }

        let mut dims: Vec<u64> = t.size.iter().map(|&x| x as u64).collect();
        let out_name = rename_key(src_name);
        let transpose = dims.len() >= 2 && is_transposed(&out_name);
        if transpose {
            let n = dims.len();
            dims.swap(n - 2, n - 1);
        }
        plan.push(PlanItem {
            out_name,
            out_off: 0,
            src_off,
            dtype,
            dims,
            n_elem,
            transpose,
        });
    }

    let mut seen = HashMap::new();
    for (i, p) in plan.iter().enumerate() {
        if let Some(prev) = seen.insert(p.out_name.clone(), i) {
            return Err(format!("name collision: {} and {} both map to {}", tensors[prev].0, tensors[i].0, p.out_name));
        }
    }

    // ---- safetensors header (sorted keys, offsets in key order) ---------
    let mut order_by_name: Vec<usize> = (0..plan.len()).collect();
    order_by_name.sort_by(|&a, &b| plan[a].out_name.as_bytes().cmp(plan[b].out_name.as_bytes()));

    let mut js = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
    let mut offset: u64 = 0;
    for &i in &order_by_name {
        let nbytes = plan[i].n_elem * 2;
        plan[i].out_off = offset;
        js.push_str(&format!(
            ",\"{}\":{{\"dtype\":\"F16\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            json_escape(&plan[i].out_name),
            plan[i].dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
            offset,
            offset + nbytes
        ));
        offset += nbytes;
    }
    js.push('}');
    let total_bytes = offset;
    let pad = (8 - js.len() % 8) % 8;
    let data_start = (8 + js.len() + pad) as u64;

    // ---- batches --------------------------------------------------------
    let mut order_by_src: Vec<usize> = (0..plan.len()).collect();
    order_by_src.sort_by_key(|&i| plan[i].src_off);

    let threads = if args.threads > 0 {
        args.threads
    } else {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8)
    };
    let largest = plan.iter().map(|p| p.n_elem * p.dtype.size()).max().unwrap_or(0);
    let batch_bytes = if args.batch_mb > 0 {
        args.batch_mb << 20
    } else {
        let avail = free_memory_bytes();
        let per = avail / 8 / (threads as u64 + 2);
        per.clamp(32 << 20, 256 << 20)
    };
    println!(
        "threads: {threads}; batch: {:.0} MiB (free RAM {:.1} GiB); largest tensor {:.0} MiB",
        batch_bytes as f64 / 1048576.0,
        free_memory_bytes() as f64 / 1073741824.0,
        largest as f64 / 1048576.0
    );

    let mut batches: Vec<Batch> = Vec::new();
    let mut i = 0usize;
    while i < order_by_src.len() {
        let start = plan[order_by_src[i]].src_off;
        let mut j = i;
        let mut end = start + plan[order_by_src[i]].n_elem * plan[order_by_src[i]].dtype.size();
        while j + 1 < order_by_src.len() {
            let it = &plan[order_by_src[j + 1]];
            let nxt_end = it.src_off + it.n_elem * it.dtype.size();
            if nxt_end.max(end) - start > batch_bytes {
                break;
            }
            end = nxt_end.max(end);
            j += 1;
        }
        let jobs = order_by_src[i..=j]
            .iter()
            .map(|&k| {
                let it = &plan[k];
                Job {
                    out_off: data_start + it.out_off,
                    rel: it.src_off - start,
                    n_elem: it.n_elem,
                    dtype: it.dtype,
                    dims: it.dims.clone(),
                    transpose: it.transpose,
                }
            })
            .collect();
        batches.push(Batch {
            start,
            span: end - start,
            jobs,
        });
        i = j + 1;
    }

    // ---- write ----------------------------------------------------------
    let out = Arc::new(
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&args.output)
            .map_err(|e| format!("create {}: {e}", args.output))?,
    );
    let mut header = Vec::with_capacity(8 + js.len() + pad);
    header.extend_from_slice(&((js.len() + pad) as u64).to_le_bytes());
    header.extend_from_slice(js.as_bytes());
    header.extend(std::iter::repeat(b' ').take(pad));
    pwrite(&out, &header, 0).map_err(|e| format!("write header: {e}"))?;

    let (tx, rx) = sync_channel::<(Vec<u8>, Batch)>(threads.max(1));
    let (free_tx, free_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let rx = Arc::new(Mutex::new(rx));
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let done = Arc::new(AtomicU64::new(0));
    let finished = Arc::new(AtomicBool::new(false));

    let monitor = {
        let done = Arc::clone(&done);
        let finished = Arc::clone(&finished);
        std::thread::spawn(move || {
            let t0 = Instant::now();
            while !finished.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(2));
                let d = done.load(Ordering::Relaxed);
                let el = t0.elapsed().as_secs_f64().max(0.001);
                eprintln!(
                    "{:6.2} / {:6.2} GiB   {:5.0} MiB/s",
                    d as f64 / 1073741824.0,
                    total_bytes as f64 / 1073741824.0,
                    d as f64 / 1048576.0 / el
                );
            }
        })
    };

    let t_start = Instant::now();
    std::thread::scope(|scope| {
        let reader = {
            scope.spawn(move || -> Result<(), String> {
                for b in &batches {
                    let span = b.span as usize;
                    let mut buf = match free_rx.try_recv() {
                        Ok(mut v) => {
                            v.resize(span, 0);
                            v
                        }
                        Err(_) => vec![0u8; span],
                    };
                    pread(&input, &mut buf, b.start).map_err(|e| format!("read at {}: {e}", b.start))?;
                    let batch = Batch {
                        start: b.start,
                        span: b.span,
                        jobs: b.jobs.iter().map(|j| Job {
                            out_off: j.out_off,
                            rel: j.rel,
                            n_elem: j.n_elem,
                            dtype: j.dtype,
                            dims: j.dims.clone(),
                            transpose: j.transpose,
                        }).collect(),
                    };
                    if tx.send((buf, batch)).is_err() {
                        break;
                    }
                }
                Ok(())
            })
        };

        let mut workers = Vec::new();
        for _ in 0..threads {
            let rx = Arc::clone(&rx);
            let out = Arc::clone(&out);
            let free_tx: Sender<Vec<u8>> = free_tx.clone();
            let errors = Arc::clone(&errors);
            let done = Arc::clone(&done);
            workers.push(scope.spawn(move || {
                loop {
                    let msg = {
                        let guard = rx.lock().unwrap();
                        guard.recv()
                    };
                    let (buf, batch) = match msg {
                        Ok(m) => m,
                        Err(_) => break,
                    };
                    let has_error = !errors.lock().unwrap().is_empty();
                    if !has_error {
                        for job in &batch.jobs {
                            let src_len = (job.n_elem * job.dtype.size()) as usize;
                            let src = &buf[job.rel as usize..job.rel as usize + src_len];
                            let bytes = convert(job, src);
                            if let Err(e) = pwrite(&out, &bytes, job.out_off) {
                                errors.lock().unwrap().push(format!("write at {}: {e}", job.out_off));
                                break;
                            }
                            done.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                        }
                    }
                    let _ = free_tx.send(buf);
                }
            }));
        }

        let mut reader_error: Option<String> = None;
        match reader.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => reader_error = Some(e),
            Err(_) => reader_error = Some("reader thread panicked".into()),
        }
        for w in workers {
            let _ = w.join();
        }
        if let Some(e) = reader_error {
            errors.lock().unwrap().push(e);
        }
    });
    finished.store(true, Ordering::Relaxed);
    let _ = monitor.join();

    if let Some(e) = errors.lock().unwrap().first() {
        return Err(e.clone());
    }

    let written = done.load(Ordering::Relaxed);
    if written != total_bytes {
        return Err(format!("wrote {written} bytes, expected {total_bytes}"));
    }
    let len = std::fs::metadata(&args.output).map_err(|e| e.to_string())?.len();
    if len != data_start + total_bytes {
        return Err(format!("output file has {len} bytes, expected {}", data_start + total_bytes));
    }
    println!(
        "saved {} ({:.2} GiB) in {:.1}s",
        args.output,
        (data_start + total_bytes) as f64 / 1073741824.0,
        t_start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
