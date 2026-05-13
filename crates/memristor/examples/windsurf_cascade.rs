//! Run: `cargo run -p memristor --example windsurf_cascade`

use memristor::services::vchip_api::OmegaVChip;

fn main() {
    let mut chip = OmegaVChip::new(4, 0.001, 0.01);
    for i in 0..4 {
        chip.crossbar_mut().set_resistance(i, i, 1.0).unwrap();
    }
    let input = vec![0.25_f32, 0.5, 0.75, 1.0];
    let depth = 3_usize;
    let out = chip.infer_cascade(&input, depth).expect("cascade");
    println!("input:  {input:?}");
    println!("depth:  {depth}");
    println!("output: {out:?}");
}
