//! Run (smoke / quick samples):  
//! `cargo bench -p memristor --features memristor-bench -- --quick`

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use memristor::memristor::crossbar::Crossbar;
use memristor::services::vchip_api::OmegaVChip;

fn uniform_crossbar(size: usize) -> Crossbar {
    let mut bar = Crossbar::new(size, 0.001, 0.0);
    for i in 0..size {
        for j in 0..size {
            bar.set_resistance(i, j, 1.0).unwrap();
        }
    }
    bar
}

fn uniform_chip(size: usize) -> OmegaVChip {
    let mut chip = OmegaVChip::new(size, 0.001, 0.0);
    for i in 0..size {
        for j in 0..size {
            chip.crossbar_mut().set_resistance(i, j, 1.0).unwrap();
        }
    }
    chip
}

fn bench_crossbar_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossbar_forward");
    for size in [32_usize, 64, 128, 256] {
        let bar = uniform_crossbar(size);
        let input = vec![1.0_f32; size];
        group.throughput(Throughput::Elements(size as u64 * size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_s| {
            b.iter(|| bar.forward(black_box(&input)).unwrap());
        });
    }
    group.finish();
}

fn bench_infer_cascade(c: &mut Criterion) {
    let size = 64_usize;
    let chip = uniform_chip(size);
    let input = vec![0.5_f32; size];
    let mut group = c.benchmark_group("infer_cascade");
    for depth in [1_usize, 2, 4, 8] {
        group.throughput(Throughput::Elements((size * depth) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &d| {
            b.iter(|| chip.infer_cascade(black_box(&input), d).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_crossbar_forward, bench_infer_cascade);
criterion_main!(benches);
