# MemristiveRelation Feature Specification
## Phase 1, Aufgabe 1.1

**Datum**: 14. Mai 2026  
**Repo**: memoristorCursor  
**Datei**: crates/memristor/src/memristive_relation.rs (NEU)  
**Status**: Spezifikation

---

## Feature Beschreibung

MemristiveRelation ist eine erweiterbare Struktur für "erinnernde Beziehungen" im MEMORISSTORE System. Sie baut auf der existierenden MemristorCell-Implementierung auf und fügt semantische, historische und resonanz-bezogene Eigenschaften hinzu.

### Kern-Prinzip

Eine Beziehung ist nicht mehr statisch, sondern "lebendig":
- Sie hat eine Stärke (conductance)
- Sie hat eine Aktivierungsenergie (threshold)
- Sie hat einen Zerfall (decay)
- Sie hat eine Kohärenz (consistency)
- Sie hat eine Phase (für Resonanz)
- Sie hat eine Frequenz (für Resonanz)
- Sie hat eine Herkunft (provenance)
- Sie hat eine historische Tiefe (depth)
- Sie hat eine semantische Energie (meaning)
- Sie hat ein Vertrauensniveau (confidence)

---

## Use Cases

### Use Case 1: Erstellen einer MemristiveRelation aus MemristorCell

**Beschreibung**: Eine MemristiveRelation wird aus einer existierenden MemristorCell erstellt.

**Preconditions**:
- MemristorCell existiert mit conductance, w, alpha, noise_level, v_th

**Steps**:
1. MemristorCell wird erstellt mit Standard-Parametern
2. MemristiveRelation wird aus MemristorCell konvertiert
3. Physikalische Eigenschaften werden übertragen (conductance, activation_energy, decay_rate, coherence)
4. Semantische Eigenschaften werden initialisiert (phase=0, frequency=1, semantic_energy=0, confidence=1)
5. Historische Eigenschaften werden initialisiert (provenance=0, depth=0, activation_count=0)

**Postconditions**:
- MemristiveRelation existiert mit korrekten physikalischen Eigenschaften
- Semantische Eigenschaften sind auf Standard-Werten initialisiert
- Historische Eigenschaften sind auf Standard-Werten initialisiert

**Acceptance Criteria**:
- conductance == MemristorCell.conductance()
- activation_energy == MemristorCell.v_th
- decay_rate == MemristorCell.alpha
- coherence == MemristorCell.w
- phase == 0.0
- frequency == 1.0
- semantic_energy == 0.0
- confidence == 1.0
- provenance == Hash::default()
- historical_depth == 0
- activation_count == 0

### Use Case 2: Manuelles Erstellen einer MemristiveRelation

**Beschreibung**: Eine MemristiveRelation wird manuell mit spezifischen Werten erstellt.

**Preconditions**:
- Keine

**Steps**:
1. MemristiveRelation wird mit spezifischen Werten erstellt
2. Alle Eigenschaften werden explizit gesetzt

**Postconditions**:
- MemristiveRelation existiert mit den spezifizierten Werten

**Acceptance Criteria**:
- Alle Eigenschaften haben die spezifizierten Werte
- Keine Default-Werte werden verwendet

### Use Case 3: Aktivierung einer MemristiveRelation

**Beschreibung**: Eine MemristiveRelation wird aktiviert, was ihre Eigenschaften ändert.

**Preconditions**:
- MemristiveRelation existiert

**Steps**:
1. MemristiveRelation wird aktiviert
2. conductance wird erhöht (Lernen)
3. activation_count wird erhöht
4. last_activation wird aktualisiert
5. phase wird angepasst (Resonanz-Learning)

**Postconditions**:
- conductance ist erhöht
- activation_count ist erhöht
- last_activation ist aktualisiert
- phase ist angepasst

**Acceptance Criteria**:
- conductance > vorheriger conductance
- activation_count == vorheriger activation_count + 1
- last_activation == jetzt
- phase ist angepasst (nicht gleich wie vorher)

### Use Case 4: Zerfall einer MemristiveRelation

**Beschreibung**: Eine MemristiveRelation zerfällt über Zeit.

**Preconditions**:
- MemristiveRelation existiert

**Steps**:
1. Zeit vergeht
2. conductance wird basierend auf decay_rate reduziert
3. coherence wird basierend auf decay_rate reduziert

**Postconditions**:
- conductance ist reduziert
- coherence ist reduziert

**Acceptance Criteria**:
- conductance < vorheriger conductance
- coherence < vorherige coherence
- Reduktion basiert auf decay_rate

### Use Case 5: Resonanz-Berechnung zwischen Signal und MemristiveRelation

**Beschreibung**: Die Resonanz zwischen einem Signal und einer MemristiveRelation wird berechnet.

**Preconditions**:
- Signal existiert mit phase und frequency
- MemristiveRelation existiert mit phase und frequency

**Steps**:
1. Phase-Alignment wird berechnet: cos(signal_phase - relation_phase)
2. Frequency-Match wird berechnet: 1 - |signal_frequency - relation_frequency|
3. Resonanz wird berechnet: conductance × coherence × phase_alignment × frequency_match

**Postconditions**:
- Resonanz-Wert ist berechnet

**Acceptance Criteria**:
- Resonanz ist im Bereich [0, 1]
- Resonanz ist höher bei besserer Phase-Alignment
- Resonanz ist höher bei besserem Frequency-Match
- Resonanz ist höher bei höherer conductance
- Resonanz ist höher bei höherer coherence

### Use Case 6: Serialisierung und Deserialisierung

**Beschreibung**: Eine MemristiveRelation wird serialisiert und deserialisiert.

**Preconditions**:
- MemristiveRelation existiert

**Steps**:
1. MemristiveRelation wird zu JSON serialisiert
2. JSON wird zu MemristiveRelation deserialisiert
3. Original und Deserialisiert werden verglichen

**Postconditions**:
- Deserialisierte MemristiveRelation ist identisch mit Original

**Acceptance Criteria**:
- Alle Eigenschaften sind identisch
- Keine Information geht verloren

---

## Technical Specification

### Struct Definition

```rust
pub struct MemristiveRelation {
    // Physikalische Eigenschaften (aus MemristorCell)
    pub conductance: f32,
    pub activation_energy: f32,
    pub decay_rate: f32,
    
    // Kohärenz-Eigenschaften
    pub coherence: f32,
    pub phase: f32,
    pub frequency: f32,
    
    // Historische Eigenschaften
    pub provenance: Hash,
    pub historical_depth: u64,
    pub activation_count: u64,
    pub last_activation: Timestamp,
    
    // Semantische Eigenschaften
    pub resonance_class: ResonanceClass,
    pub semantic_energy: f32,
    pub confidence: f32,
    
    // Transformations-Eigenschaften
    pub transformability: f32,
    pub plasticity: f32,
}
```

### Methoden

```rust
impl MemristiveRelation {
    pub fn new() -> Self;
    pub fn from_memristor_cell(cell: MemristorCell) -> Self;
    pub fn activate(&mut self);
    pub fn decay(&mut self, dt: f32);
    pub fn compute_resonance(&self, signal: &Signal) -> f32;
}
```

### Dependencies

- `memristor::memristor::cell::MemristorCell`
- `serde` für Serialisierung
- `sha2` für Hash
- `chrono` für Timestamp

---

## Non-Functional Requirements

### Performance
- Erstellung einer MemristiveRelation: < 1μs
- Aktivierung: < 1μs
- Resonanz-Berechnung: < 100ns

### Memory
- Größe einer MemristiveRelation: < 128 bytes

### Correctness
- Alle Berechnungen sind deterministisch
- Keine Race Conditions
- Keine Memory Leaks

---

## Test Strategy

### Unit Tests
- Test 1: Erstellen aus MemristorCell
- Test 2: Manuelles Erstellen
- Test 3: Aktivierung
- Test 4: Zerfall
- Test 5: Resonanz-Berechnung
- Test 6: Serialisierung/Deserialisierung

### Integration Tests
- Test 1: Integration mit MemristorCell
- Test 2: Integration mit Signal

### Property-Based Tests
- Test 1: Resonanz ist immer im Bereich [0, 1]
- Test 2: Aktivierung erhöht conductance
- Test 3: Zerfall reduziert conductance

---

## Success Criteria

- Alle Use Cases sind implementiert
- Alle Unit Tests passieren
- Alle Integration Tests passieren
- Alle Property-Based Tests passieren
- Performance Requirements sind erfüllt
- Memory Requirements sind erfüllt
