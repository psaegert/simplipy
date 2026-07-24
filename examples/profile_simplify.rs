//! Allocation-profiling harness for the simplify hot path (counting allocator).
//! Usage: profile_simplify <config.yaml> <rules.json> <corpus.txt>
//! corpus.txt = one expression per line, space-separated prefix tokens.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(new_size as u64, Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let eng = _core::Engine::from_paths(&args[1], &args[2]).expect("engine");
    let corpus: Vec<Vec<String>> = std::fs::read_to_string(&args[3])
        .expect("corpus")
        .lines()
        .map(|l| l.split(' ').map(str::to_string).collect())
        .collect();
    for e in corpus.iter().take(200) {
        eng.simplify(e, 5, None, true, false);
    }
    let (a0, b0) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    let t0 = std::time::Instant::now();
    let mut sink = 0usize;
    for e in &corpus {
        sink += eng.simplify(e, 5, None, true, false).len();
    }
    let dt = t0.elapsed();
    let (a1, b1) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    let n = corpus.len() as f64;
    println!(
        "n={} sink={} us/expr(with counting)={:.1}",
        corpus.len(),
        sink,
        dt.as_micros() as f64 / n
    );
    println!(
        "allocs/expr={:.0} bytes/expr={:.0}",
        (a1 - a0) as f64 / n,
        (b1 - b0) as f64 / n
    );
}
