//! macOS-only smoke test for shared-buffer GPU crossbar.
#[cfg(target_os = "macos")]
fn main() {
    let n = 8_usize;
    // Einheitlicher Widerstand R=2 → Leitwert G=1/2
    let conductance = vec![0.5_f32; n * n];
    let input: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
    match memristor_metal::crossbar_forward_metal(n, &conductance, &input) {
        Ok(out) => println!("ok: {:?}", out),
        Err(e) => eprintln!("err: {e}"),
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("memristor_metal examples require macOS.");
}
