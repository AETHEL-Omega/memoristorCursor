# Apple Silicon (M5): Unified Memory, CPU, GPU, NPU

Dieses Dokument beschreibt, **was dieses Repo heute tut**, **was fehlt**, und wie ihr **UMA plus CPU/GPU/ANE** auf macOS sinnvoll nutzt – ohne Marketingzahlen, mit klaren technischen Leitplanken.

## Kurz: Ist-Zustand (`memristor` / `aethel_mlx_runtime`)

| Bereich | Stand |
|--------|--------|
| **Rechenkern (`memristor`)** | **CPU-f32**: **aarch64** NEON `forward_aarch64` (**`vfmaq`**, Leitwert `G`); **gecachte** `conductance_matrix` (kein Zellen‑Scan pro Zeile); optional **Rayon** (macOS: niedrigere Parallel‑Schwelle). |
| **`memristor_metal` (macOS)** | **Metal Compute**: `crossbar_forward` mit **`fma`** + 4‑Entrollung; **Cascade** als **ein Command Buffer** (`forward_cascade_into` / [`crossbar_forward_metal_cascade`]) mit Blit Ping‑Pong; Threadgroup‑Tuning; gepoolte shared `MTLBuffer`s; **`forward_into`** ohne `Vec`/Schritt. |
| **„Mlx“ / „CoreMl“ in `InferenceExecutor`** | **Stubs** (Pass-through bzw. nur Memristor/CPU-Pfad). Kein echtes MLX, kein Core ML Aufruf. |
| **ANE (NPU)** | Nicht angebunden. Typischer Weg: **Core ML** / **MLX**-Graphen außerhalb dieses Crates. |

Das ist für einen **Simulations-Prototyp** in Ordnung; **`memristor_metal`** liefert den ersten **GPU-UMA-Schrift**; für Produktions-MLX/Core-ML-Pipelines fehlen weiterhin die **Framework-Adapter**.

---

## Metal-MVP (`memristor_metal`)

- **API:** `crossbar_forward_metal(n, conductance, input)` nimmt **`conductance` als Leitwert‑Matrix `G`**; bequem aus dem Chip: **`forward_crossbar` / `forward_crossbar_cascade`** + [`Crossbar::conductance_matrix`] (gecacht auf der CPU‑Seite).
- **`MetalRunner`:** `MetalRunner::new()` + `forward(...)` für Hot-Loops einmalig pro Thread/Task anlegen; bei festem `n` bleiben die **shared** `MTLBuffer`s allokiert und werden nur per Host-Write + `didModifyRange` aktualisiert.
- **Shader:** Pro Zeile \(y_i=\sum_j \mathrm{fma}(x_j,G_{ij},…)\); **`G_ij=1/R_ij`** (**Leitwert**, wie NEON‑CPU) — keine Division im Innerloop auf der GPU.
- **Dispatch:** `threadsPerThreadgroup` = min(Pipeline‑Maximum, **1024**), auf **`threadExecutionWidth`** (SIMD‑Breite, typ. 32) abgerundet — weniger Gruppen bei großem **n**.
- **Speicher:** `newBuffer` mit **shared** storage — passend zu **Unified Memory** auf Apple Silicon (kein dediziertes „Video-RAM“ wie bei PCIe-GPUs).
- **Tests:** `metal_matches_cpu_crossbar` vergleicht gegen `memristor::Crossbar` (64×64).
- **Pipeline-Cache:** MSL + Compute-Pipeline + Queue liegen in einem **`OnceLock`** (`warm_metal_pipeline()` optional vorab). Pro Aufruf nur noch Puffer-Füllung + Encode + GPU (kein erneutes Kompilieren); **`CompileOptions::set_fast_math_enabled(false)`** für konsistentere FP-Nähe zur CPU.
- **Cascade (M‑Serie‑tauglich):** Mehrfaches \(y\leftarrow Gy\) in **einem** `MTLCommandBuffer` ([`MetalRunner::forward_cascade_into`], Hilfs‑API [`crossbar_forward_metal_cascade`]): zwischen den Kern‑Dispatches kopiert **`MTLBlitCommandEncoder`** `output→input`, sodass keine CPU‑Zwischenschleife nötig ist.
- **Ausblick:** MSL‑Tiles / simdgroup‑Reduktion; Example `compare_cpu_metal` — bei kleinem `n` oft GPU‑Overhead.

---

## Was „Unified Memory“ für uns praktisch bedeutet

Auf M‑Serie teilen sich CPU, GPU und (über Treiber/Frameworks) auch Workloads Richtung **ANE** denselben physischen RAM-Pool (**UMA**). Vorteil: weniger klassisches „DDR-PCIe“-Flaschenhals-Modell wie bei Diskret-GPUs.

**Aber:** Rust-`Vec<f32>` liegt im **Anwendungsspeicher**. Damit GPU oder ANE **ohne Kopie** lesen/schreiben, müssen Daten in **API-kompatiblen Buffern** landen, z. B.:

- **Metal:** `MTLBuffer` mit `storageMode: .shared` (macOS), gleicher physikalischer Speicher von CPU und GPU nutzbar – **trotzdem** sauber mit Command Queues und Synchronisation (Fences/Events) fahren.
- **Core ML:** Eingaben oft als `MLMultiArray`; große, stabile Gewichte/KV oft über **memory-mapped** oder Framework-interne Puffer – nicht automatisch identisch mit einem zufälligen Rust-`Vec`.

**Folgerung:** „UMA nutzen“ = **API wählen**, die **zero-copy oder single-copy** in „Accelerator-sichtbare“ Puffer erlaubt – nicht bloß schnelle CPU-Schleifen.

---

## Rollenverteilung: CPU vs GPU vs ANE (NPU)

### CPU (Performance-/Efficiency-Kerne)

- **Orchestrierung:** Routing, Session-Lifecycle, Drift-Zeitstempel, I/O, Metriken.
- **Simulation kleiner Crossbars** oder **Vektorbreite Hot-Pfade** mit **SIMD** (NEON auf Arm64) oder optional **Zeilenparallelität** (z. B. Rayon): gut, wenn Arbeit **pro Zeile unabhängig** ist (wie `forward`).

**Wo CPU falsch wäre:** Große, reguläre **Matmul-/Batch-Matmul**-Lasten, die ihr eh schon in MLX/NN packt – dort GPU/ANE vorziehen.

### GPU (Metal)

- **Große** Matrix-Vektor-/Matrix-Matrix-Operationen, **viele gleichartige** Schritte (Batch), Custom-Kernels (z. B. Drift-Update über das ganze Array).
- **UMA:** Kernel lesen/schreiben auf **shared** Buffern; CPU füllt/liest dieselben – **kein** unnötiges `memcpy` pro Frame.

### ANE (via Core ML) bzw. „Apple Neural Engine“

- Beschleunigt **bestimmte** Teilgraphen (quantisierte/festgeformte Ops), **nicht** beliebiges Rust.
- Typisch: **Embeddings**, **kleine Klassifikatoren**, **Teile** einer Pipeline, die ihr als **Core ML Model** exportiert und mit **MLModel** ladet.

**Nicht erwarten:** Dass ein freies, simuliertes Memristor-Gitter „automatisch“ die ANE füllt – dafür braucht es ein **modellfähiges Mapping** oder Metal.

---

## Konkrete nächste Schritte (priorisiert)

1. **CPU näher an die Hardware**
   - Auf **aarch64** (Apple Silicon): **NEON‑Pfad** für `Crossbar::forward`; optional **Accelerate/vDSP** nur bei klarem Gewinn und `cfg(macOS)` (aktuell nicht nötig).
   - Optional: Apple **Accelerate/vDSP** nur auf **macOS** hinter `cfg` + FFI (Zusatzpflege).

2. **GPU-Pfad** — **MVP erledigt** (`memristor_metal`): Compute-Shader, Shared Buffers, Pipeline-Cache, gepoolte Puffer, **GPU‑Cascade** in einem Command Buffer. **Als Nächstes:** größere Tile-/SIMD‑Optimierung im MSL; optional **Conductance nur bei Programmierung** auf GPU hochladen (heute: volles `G` pro Aufruf).

3. **ANE / Core ML**
   - Nur wenn ein Teilproblem **als Core ML Modell** sinnvoll ist (z. B. feste Tensorformen).
   - Swift-Brücke oder bestehende App-Schicht; nicht in diesem reinen Rust-Crate zwingend.

4. **Thermik / Power**
   - Kein `powermetrics` im **Hot Path** (Latenz, Rechte). Eher: periodisches Sampling in einem **Hintergrundthread** oder IOKit/API-Stubs mit klarer Fehlerbehandlung.

5. **Messbarkeit**
   - `cargo bench` mit `--features memristor-bench`, dokumentierte Matrixgrößen, **Energy** optional über `powermetrics` **einmal** pro Szenario (Skript), nicht inline.

---

## Feature-Flags in diesem Repo

| Feature | Zweck |
|---------|--------|
| `memristor-parallel` (optional) | Zeilen-paralleles `Crossbar::forward` ab konfigurierbarem Schwellwert (Rayon) – gut für große `n` auf vielen CPU-Kernen. |
| `memristor-bench` | Criterion-Benchmarks für Durchsatz. |
| `memristor_metal` | Separates Crate (nur macOS): Metal Compute + Shared Buffers; kein Cargo-Feature nötig. |

---

## Fazit

- **CPU:** `memristor` + optional **Rayon**. **GPU (UMA-Pfad):** `memristor_metal` mit **Shared `MTLBuffer`** und `MetalRunner`/thread-local Pooling. **ANE/Core ML** weiterhin **nicht** in diesem Workspace verdrahtet — dafür braucht es Modell- oder App-Schicht.
- Nächster Hebel für „mehr M5“: **Cascade auf der GPU**, NEON innerhalb der CPU-Zeile, später **Core ML** für klar abgegrenzte Teilnetze.
