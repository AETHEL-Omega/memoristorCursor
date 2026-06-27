//! Compare wall time: CPU `Crossbar::forward` vs Metal (`MetalRunner`, pooled buffers).
//! `conductance_epoch` unverändert ⇒ **kein** erneutes `n²`‑`G`‑Memcpy pro Metal‑Iter.
//!
//! `cargo run -p memristor_metal --example compare_cpu_metal --release`
//! `cargo run -p memristor_metal --example compare_cpu_metal --release --features memristor-parallel`

#[cfg(target_os = "macos")]
fn fill_bar(bar: &mut memristor::memristor::crossbar::Crossbar, n: usize) {
    #[cfg(feature = "memristor-parallel")]
    {
        bar.fill_resistance_par(|i, j| 1.1 + 0.0007 * ((i * n + j) as f32));
    }
    #[cfg(not(feature = "memristor-parallel"))]
    {
        bar.fill_resistance(|i, j| 1.1 + 0.0007 * ((i * n + j) as f32));
    }
}

#[cfg(target_os = "macos")]
fn main() {
    use memristor::memristor::crossbar::Crossbar;
    use memristor_metal::{warm_metal_pipeline, MetalRunner};
    use std::time::Instant;

    let sizes = [128usize, 256, 512, 1024, 2048, 4096];
    let cascade_depth = 2usize;

    warm_metal_pipeline().expect("Metal pipeline warm");
    let mut runner = MetalRunner::new().expect("MetalRunner");

    for &n in &sizes {
        let iters = if n >= 4096 {
            8
        } else if n >= 2048 {
            20
        } else {
            50
        };

        let mut bar = Crossbar::new(n, 0.001, 0.0);
        fill_bar(&mut bar, n);
        let g = bar.conductance_matrix();
        let epoch = bar.conductance_epoch();
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

        let cpu_1 = bar.forward(&input).unwrap();
        let gpu_1 = runner.forward(n, g, epoch, &input).unwrap();
        let max_err = cpu_1
            .iter()
            .zip(gpu_1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let err_tol = if n >= 4096 { 5e-2 } else { 1e-3 };
        assert!(max_err < err_tol, "n={n} max_err={max_err}");

        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = bar.forward(&input).unwrap();
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

        let mut metal_out = vec![0.0_f32; n];
        let t1 = Instant::now();
        for _ in 0..iters {
            runner
                .forward_into(n, g, epoch, &input, &mut metal_out)
                .unwrap();
        }
        let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

        let t1b = Instant::now();
        runner
            .forward_repeated_into(n, g, epoch, &input, iters, &mut metal_out)
            .unwrap();
        let gpu_batched_ms = t1b.elapsed().as_secs_f64() / (iters as f64) * 1000.0;

        println!(
            "n={n:4}  forward  cpu {:8.3} ms  metal {:8.3} ms  metal/cpu {:5.2}x  batched {:8.3} ms  batched/cpu {:5.2}x",
            cpu_ms,
            gpu_ms,
            gpu_ms / cpu_ms.max(1e-12),
            gpu_batched_ms,
            gpu_batched_ms / cpu_ms.max(1e-12),
        );

        let cpu_c = bar.forward_cascade(&input, cascade_depth).unwrap();
        let mut gpu_c = vec![0.0_f32; n];
        runner
            .forward_cascade_into(n, g, epoch, &input, cascade_depth, &mut gpu_c)
            .unwrap();
        let c_err = cpu_c
            .iter()
            .zip(gpu_c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let c_tol = if n >= 4096 {
            1.0
        } else if n >= 2048 {
            0.12
        } else {
            8e-2
        };
        assert!(c_err < c_tol, "cascade n={n} max_err={c_err}");

        let t2 = Instant::now();
        for _ in 0..iters {
            let _ = bar.forward_cascade(&input, cascade_depth).unwrap();
        }
        let cpu_c_ms = t2.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

        let t3 = Instant::now();
        for _ in 0..iters {
            runner
                .forward_cascade_into(n, g, epoch, &input, cascade_depth, &mut gpu_c)
                .unwrap();
        }
        let gpu_c_ms = t3.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

        let t3b = Instant::now();
        runner
            .forward_cascade_repeated_into(n, g, epoch, &input, cascade_depth, iters, &mut gpu_c)
            .unwrap();
        let gpu_c_batched_ms = t3b.elapsed().as_secs_f64() / (iters as f64) * 1000.0;

        println!(
            "n={n:4}  cascade d={cascade_depth}  cpu {:8.3} ms  metal {:8.3} ms  metal/cpu {:5.2}x  batched {:8.3} ms  batched/cpu {:5.2}x",
            cpu_c_ms,
            gpu_c_ms,
            gpu_c_ms / cpu_c_ms.max(1e-12),
            gpu_c_batched_ms,
            gpu_c_batched_ms / cpu_c_ms.max(1e-12),
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This example requires macOS + Metal.");
}
