# PSA v3 Enhancement Changelog

This document tracks all enhancements to the PSA (Plastic State Agent) system.

---

## [v3.1.0] - 2024-12-06 - Stabilized State Dynamics & Hybrid RAG

### Overview
Major refactor to fix hallucination issues and improve factual recall through:
1. **Stabilized State Dynamics** - Fix kernel physics (unbounded addition, device fragility)
2. **Entity Sidecar** - Extract high-entropy facts during imprinting
3. **Liquid RAG** - Inject entities into context at runtime (invisible to user)

### Phase 1: Kernel Refactor (`src/psa/kernel.py`)

#### 1.1 Fixed Device Detection
- **Problem**: Hardcoded `blocks.0.ln1.weight` for device/dtype detection could fail
- **Solution**: Iterate through `self.model.w.values()` to find first available tensor
- **Files Changed**: `kernel.py` - `_get_zero_state()`, `load_state()`

#### 1.2 Epoch-Based Imprinting
- **Problem**: Single-pass imprinting doesn't reinforce early content (like port numbers)
- **Solution**: Added `epochs: int = 1` parameter to `learn_stream()`
- **Behavior**: Text processed `epochs` times, state carried between epochs
- **Files Changed**: `kernel.py` - `learn_stream()`

#### 1.3 Energy Normalization
- **Problem**: Unbounded state addition causes "exploding state" and garbage output
- **Solution**: Added `normalize: bool = True` parameter to `apply_delta()`
- **Behavior**: Preserves base state's L2 norm after delta application
- **Formula**: `result = result * (base_norm / result_norm)`
- **Files Changed**: `kernel.py` - `apply_delta()`

---

### Phase 2: Imprinter Refactor (`src/psa/cli.py`)

#### 2.1 Entity Extraction in Imprint
- **Problem**: High-entropy facts (ports, hex codes) lost in state compression
- **Solution**: Regex extraction of entities saved to JSON sidecar
- **Entities Captured**:
  - All-caps words: `SIGUSR1`, `HYDRA`
  - Hex strings: `0xDEADBEEF`
  - Numbers: `8088`, `5002`
  - CamelCase: `HydraClient`
- **Files Changed**: `cli.py` - `imprint()` command

#### 2.2 Epochs CLI Option
- **Added**: `--epochs` option (default: 5) to `imprint` command
- **Files Changed**: `cli.py` - `imprint()` command

#### 2.3 Entity Merging in Mix
- **Problem**: Mixed deltas lose entity metadata
- **Solution**: Aggregate entities from all source deltas
- **Files Changed**: `cli.py` - `mix()` command

---

### Phase 3: Runtime Refactor (`src/psa/cli.py`)

#### 3.1 Context Injection (Liquid RAG)
- **Problem**: State algebra biases reasoning but doesn't enable precise recall
- **Solution**: Prepend entity context to prompts invisibly
- **Format**: `"Context: [Entities: Port 8088, SIGUSR1, 0xDEADBEEF, ...]"`
- **Visibility**: Hidden from user terminal, visible only to model
- **Files Changed**: `cli.py` - `chat()` command

---

### Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────┐         ┌─────────────────┐          │
│   │  State Vector   │         │ Entity Sidecar  │          │
│   │  (Delta File)   │         │  (JSON File)    │          │
│   │                 │         │                 │          │
│   │  Handles:       │         │  Handles:       │          │
│   │  - Reasoning    │         │  - Facts        │          │
│   │  - Domain Bias  │         │  - Recall       │          │
│   │  - "SRE Vibe"   │         │  - Precision    │          │
│   └────────┬────────┘         └────────┬────────┘          │
│            │                           │                    │
│            └───────────┬───────────────┘                    │
│                        │                                    │
│                        ▼                                    │
│            ┌───────────────────────┐                        │
│            │   Runtime Inference   │                        │
│            │                       │                        │
│            │  State + Context      │                        │
│            │  Injection            │                        │
│            └───────────────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Testing

#### Verification Commands
```bash
# Re-imprint with epochs
psa imprint devops_data/docs_hydra.txt --name hydra_v2 --epochs 10

# Check entities extracted
cat ~/.psa/deltas/hydra_v2.json | jq '.entities'

# Chat with hybrid mode (lower gain since context handles facts)
psa chat --deltas hydra_v2 --gains 0.6
```

#### Test Questions
1. "What port does it run on?" → Expected: 8088
2. "How do I restart it?" → Expected: SIGUSR1
3. "What are the magic bytes?" → Expected: 0xDEADBEEF

---

### Deprecations
- None in this release

### Breaking Changes
- `learn_stream()` signature changed (new `epochs` param, backward compatible)
- `apply_delta()` signature changed (new `normalize` param, backward compatible)

---
