# PSA v3 - Plastic State Agent: Complete Codebase Analysis

## Executive Summary

PSA (Plastic State Agent) is a system for **inference-time knowledge injection** using RWKV's recurrent state. The core idea is "State Algebra" - treating the hidden state of an RNN-like model as a vector that can be manipulated algebraically to inject learned knowledge without retraining.

### Core Concept
```
Learning:   S_t = process(text, S_0)      # Run text through model
Delta:      ΔS = S_t - S_0                # Extract "skill vector"
Injection:  S_new = S_base + w*ΔS         # Apply skill at inference time
Mixing:     S_composite = Σ(w_i * ΔS_i)   # Combine multiple skills
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         PSA v3 System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   CLI       │───▶│   Kernel    │───▶│   RWKV      │         │
│  │  (cli.py)   │    │ (kernel.py) │    │   Model     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                                    │
│         ▼                  ▼                                    │
│  ┌─────────────┐    ┌─────────────┐                            │
│  │   Deltas    │    │   Agent     │                            │
│  │ (deltas.py) │    │ (agent.py)  │                            │
│  └─────────────┘    └─────────────┘                            │
│                                                                 │
│  Storage: ~/.psa/deltas/*.delta + *.json (entity sidecar)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## File 1: kernel.py - The Liquid State Engine

This is the core engine implementing State Algebra operations.

```python
"""
PSA v3 Kernel: Liquid State Engine (RWKV-v6)
--------------------------------------------
Implements "State Algebra" for Neuro-Symbolic Agents.
Concepts:
  - S_0: Zero State (Tabula Rasa)
  - S_t: The state after t tokens.
  - Delta(S): S_t - S_0 (The "Skill Vector")
  - Merge(S_a, S_b): S_a + S_b (Skill Composition)

This kernel allows "Learning" to be reduced to "Vector Addition".
"""

import os
import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Tuple, Callable
from pathlib import Path

os.environ["RWKV_JIT_ON"] = "1"
if torch.cuda.is_available():
    os.environ["RWKV_CUDA_ON"] = "1"
else:
    os.environ["RWKV_CUDA_ON"] = "0"

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


class LiquidAgent:
    """
    The Liquid State Engine - treats LLM memory as algebraic vectors.

    State Algebra Operations:
    - Zero State (S_0): Fresh model with no context
    - Learn: S_t = process(text, S_0)
    - Delta: ΔS = S_t - S_0 (extract learned knowledge)
    - Apply: S_new = S_base + w*ΔS (inject knowledge)
    - Merge: S_composite = Σ(w_i * ΔS_i) (combine multiple skills)
    """

    def __init__(self, model_path: str, strategy: str = "cuda fp16", tokenizer: str = "auto"):
        print(f"[*] Loading Liquid Kernel: {model_path} ({strategy})")
        self.model = RWKV(model=model_path, strategy=strategy)
        self.model_path = model_path
        self.strategy = strategy
        self.n_layer = self.model.args.n_layer
        self.n_embd = self.model.args.n_embd

        # Auto-detect tokenizer from vocab size
        emb_shape = self.model.w['emb.weight'].shape
        actual_vocab_size = emb_shape[0]

        home = Path.home()
        pile_tokenizer_path = home / "models" / "20B_tokenizer.json"

        if tokenizer == "auto":
            if actual_vocab_size == 50277:
                tokenizer = str(pile_tokenizer_path)
            else:
                tokenizer = "rwkv_vocab_v20230424"

        self.pipeline = PIPELINE(self.model, tokenizer)
        self._raw_tokenizer = self.pipeline.tokenizer
        self._is_pile_tokenizer = (actual_vocab_size == 50277)
        self.S_0 = None

        # Cache device/dtype (Fix #5 - avoid scanning on every call)
        self._cached_device, self._cached_dtype = self._scan_device_dtype()

    def _encode(self, text: str) -> List[int]:
        if self._is_pile_tokenizer:
            return self._raw_tokenizer.encode(text).ids
        else:
            return self._raw_tokenizer.encode(text)

    def _decode(self, tokens: List[int]) -> str:
        return self._raw_tokenizer.decode(tokens)

    def _scan_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        for key, tensor in self.model.w.items():
            if 'emb' in key:
                continue
            if isinstance(tensor, torch.Tensor):
                return tensor.device, tensor.dtype
        return torch.device('cpu'), torch.float32

    def _get_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        return self._cached_device, self._cached_dtype

    def infer(
        self,
        ctx: str,
        state: Optional[List[torch.Tensor]] = None,
        token_count: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.85,
        stop_tokens: List[str] = None,
        chunk_len: int = 256
    ) -> Tuple[str, List[torch.Tensor]]:
        """
        Run inference with chunked forward pass (Fix #1).
        """
        # Pre-tokenize stop tokens for O(1) lookup (Fix #2)
        stop_token_ids = set()
        if stop_tokens:
            for st in stop_tokens:
                encoded = self._encode(st)
                if encoded:
                    stop_token_ids.add(encoded[0])

        tokens = self._encode(ctx)

        # Process context in chunks (Fix #1)
        current_state = state
        with torch.no_grad():
            for i in range(0, len(tokens), chunk_len):
                chunk = tokens[i:i + chunk_len]
                out, current_state = self.model.forward(chunk, current_state)

        # Generate tokens
        generated_tokens = []
        for _ in range(token_count):
            probs = torch.softmax(out, dim=-1)

            if temperature > 0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_token = sorted_indices[torch.multinomial(sorted_probs, 1)].item()
            else:
                next_token = torch.argmax(probs).item()

            generated_tokens.append(next_token)

            if next_token in stop_token_ids:
                break
            if stop_tokens:
                decoded = self._decode([next_token])
                if any(s in decoded for s in stop_tokens):
                    break

            out, current_state = self.model.forward([next_token], current_state)

        return self._decode(generated_tokens), current_state

    def learn_stream(
        self,
        text: str,
        initial_state: Optional[List[torch.Tensor]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        epochs: int = 1,
        chunk_len: int = 256
    ) -> List[torch.Tensor]:
        """
        Ingest text and return the FINAL STATE tensor.
        Uses chunked forward pass (Fix #1b).
        """
        tokens = self._encode(text)
        total_tokens = len(tokens) * epochs
        state = initial_state
        processed = 0

        with torch.no_grad():
            for epoch in range(epochs):
                for i in range(0, len(tokens), chunk_len):
                    chunk = tokens[i:i + chunk_len]
                    out, state = self.model.forward(chunk, state)
                    processed += len(chunk)
                    if progress_callback:
                        progress_callback(processed, total_tokens)

        return state

    def compute_delta(
        self,
        state_final: List[torch.Tensor],
        state_base: Optional[List[torch.Tensor]] = None,
        clone: bool = False
    ) -> List[torch.Tensor]:
        """
        Math: ΔS = S_final - S_base
        """
        if state_base is None:
            if clone:
                return [s.clone() if s is not None else None for s in state_final]
            return state_final

        delta_state = []
        for s_f, s_b in zip(state_final, state_base):
            if s_f is None:
                delta_state.append(None)
            elif s_b is None:
                delta_state.append(s_f.clone() if clone else s_f)
            else:
                delta_state.append(s_f - s_b)
        return delta_state

    def apply_delta(
        self,
        state_base: Optional[List[torch.Tensor]],
        delta: List[torch.Tensor],
        weight: float = 1.0,
        normalize: bool = True,
        check_interference: bool = False
    ) -> List[torch.Tensor]:
        """
        Math: S_new = S_base + (weight * ΔS)

        Energy Normalization preserves L2 norm to prevent exploding state.
        """
        if state_base is None:
            state_base = [torch.zeros_like(d) if d is not None else None for d in delta]

        new_state = []
        for s_base, s_delta in zip(state_base, delta):
            if s_delta is None:
                new_state.append(s_base)
                continue
            if s_base is None:
                new_state.append(s_delta * weight)
                continue

            result = s_base + (s_delta * weight)

            # Energy normalization (Fix #4 - optimized)
            if normalize:
                base_norm_sq = torch.dot(s_base.flatten(), s_base.flatten())
                if base_norm_sq > 1e-12:
                    result_norm_sq = torch.dot(result.flatten(), result.flatten())
                    if result_norm_sq > 1e-12:
                        scale = torch.sqrt(base_norm_sq / result_norm_sq)
                        result = result * scale

            new_state.append(result)

        return new_state

    def save_state(self, state: List[torch.Tensor], filepath: str):
        cpu_state = [s.cpu() if s is not None else None for s in state]
        torch.save({
            'state': cpu_state,
            'n_layer': self.n_layer,
            'n_embd': self.n_embd,
            'version': 'psa_v3',
        }, filepath)

    def load_state(self, filepath: str) -> List[torch.Tensor]:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        state = data['state'] if isinstance(data, dict) else data
        device, dtype = self._get_device_dtype()
        return [s.to(device=device, dtype=dtype) if s is not None else None for s in state]
```

---

## File 2: deltas.py - The Delta Mixer

```python
"""
PSA v3 - Cognitive Delta System
-------------------------------
The "Liquid Mixer" for inference-time knowledge injection.

Core Concept:
    ΔS = S_final - S_0  (The "Imprinting" step)
    S_active = S_0 + Σ(α_i * ΔS_i)  (The "Injection" step)
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import torch


class DeltaMixer:
    """
    The Liquid Mixer - like an audio mixing board.
    Each Delta is a "track", each Gain Factor is a "fader".
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self._injection_history: List[Dict[str, Any]] = []

    def inject(
        self,
        delta_manifest: Dict[str, float],
        base_state: Optional[List[torch.Tensor]] = None,
        deltas_dir: Optional[Path] = None
    ) -> Optional[List[torch.Tensor]]:
        if deltas_dir is None:
            deltas_dir = Path.home() / ".psa" / "deltas"

        active_state = base_state

        for delta_name, gain in delta_manifest.items():
            delta_path = deltas_dir / f"{delta_name}.delta"
            if not delta_path.exists():
                delta_path = deltas_dir / f"{delta_name}.psa"
            if not delta_path.exists():
                continue

            delta = self.kernel.load_state(str(delta_path))
            active_state = self.kernel.apply_delta(active_state, delta, weight=gain)

        return active_state

    def inject_from_paths(
        self,
        manifest: Dict[str, float],
        base_state: Optional[List[torch.Tensor]] = None
    ) -> Optional[List[torch.Tensor]]:
        active_state = base_state

        for path_str, gain in manifest.items():
            path = Path(path_str).expanduser()
            if not path.exists():
                continue

            delta = self.kernel.load_state(str(path))
            active_state = self.kernel.apply_delta(active_state, delta, weight=gain)

        return active_state
```

---

## File 3: cli.py - Entity Extraction (Key Feature)

```python
def extract_entities(text: str) -> List[str]:
    """
    Extract full lines containing high-entropy tokens for the Entity Sidecar.

    Captures lines containing:
    - Hex strings (0xDEADBEEF)
    - All-caps words (SIGUSR1, HYDRA, API)
    - Numbers 3+ digits (8088, 5002)
    - CamelCase terms (HydraClient)
    """
    entity_lines: Set[str] = set()
    lines = text.split('\n')

    patterns = [
        r'0x[A-Fa-f0-9]+',           # Hex strings
        r'\b[A-Z][A-Z0-9_]{1,}\b',   # All-caps words
        r'\b\d{3,}\b',               # Numbers 3+ digits
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
    ]

    noise = {'THE', 'AND', 'FOR', 'NOT', 'WITH', 'THIS', 'THAT', 'FROM', 'HAVE', 'ARE'}

    for line in lines:
        line = line.strip()
        if not line or len(line) > 200:
            continue

        for pattern in patterns:
            matches = re.findall(pattern, line)
            if matches:
                if pattern == r'\b[A-Z][A-Z0-9_]{1,}\b':
                    matches = [m for m in matches if m not in noise]
                    if not matches:
                        continue
                entity_lines.add(line)
                break

    return sorted(list(entity_lines))
```

---

## Test Data: docs_hydra.txt

```
================================================================================
META-CORP INFRASTRUCTURE DOCUMENTATION v2024.12
================================================================================

SECTION 1: LEGACY SYSTEM "HYDRA"
--------------------------------

CRITICAL INFORMATION:
---------------------
1. The Hydra API runs on Port 8088 (NOT standard 8080).

2. It uses a CUSTOM BINARY PROTOCOL, not JSON or REST.
   - All requests must be prefixed with magic bytes: 0xDE 0xAD 0xBE 0xEF
   - Response format: [4-byte length][payload][CRC32]

3. RESTART PROCEDURE (CRITICAL):
   - Do NOT use SIGTERM or SIGKILL to restart Hydra.
   - You MUST assert the SIGUSR1 signal.
   - Command: kill -USR1 $(cat /var/run/hydra.pid)
   - If you use SIGTERM, you WILL corrupt the transaction log.

4. ERROR CODES:
   - ERROR 5001: Database Lock - Run: hydra-cli unlock-db
   - ERROR 5002: Memory Leak in Worker - Requires graceful restart (SIGUSR1)
   - ERROR 5003: Transaction Queue Full - Scale up workers

5. MONITORING:
   - Prometheus endpoint: localhost:9090/hydra/metrics

PYTHON INTEGRATION:
    from hydra_client import HydraClient

    client = HydraClient(
        port=8088,  # NOT 8080!
    )

    if result.error_code == 5002:
        client.request_graceful_restart()  # Sends SIGUSR1
```

---

## Extracted Entities (hydra_v3.json)

```json
{
  "entities": [
    "1. The Hydra API runs on Port 8088 (NOT standard 8080).",
    "- All requests must be prefixed with magic bytes: 0xDE 0xAD 0xBE 0xEF",
    "- Command: kill -USR1 $(cat /var/run/hydra.pid)",
    "- Do NOT use SIGTERM or SIGKILL to restart Hydra.",
    "- ERROR 5001: Database Lock - Run: hydra-cli unlock-db",
    "- ERROR 5002: Memory Leak in Worker - Requires graceful restart (SIGUSR1)",
    "port=8088,  # NOT 8080!",
    "client.request_graceful_restart()  # Sends SIGUSR1"
  ]
}
```

---

## Test Results Comparison

### Ground Truth

| Question | Correct Answer |
|----------|----------------|
| Port | **8088** (NOT 8080) |
| Restart | `kill -USR1 $(cat /var/run/hydra.pid)` |
| Magic bytes | **0xDE 0xAD 0xBE 0xEF** |

### Results by Gain

#### "What port does Hydra run on?"

| Gain | Response | Correct? |
|------|----------|----------|
| 0.0 | port **8088** | ✅ |
| 0.2 | port **8088** | ✅ |
| 0.4 | port **8080** | ❌ |
| 0.6 | port **8080** | ❌ |
| 0.8 | **8080** | ❌ |
| 1.0 | (no port) | ❌ |

#### "How do I restart Hydra?"

| Gain | Response | Correct? |
|------|----------|----------|
| 0.0 | "open a shell..." | ❌ |
| 0.2 | "container-based..." | ❌ |
| 0.4 | "commands will restart..." | ❌ |
| 0.6 | "8gb process heap" | ❌ |
| 0.8 | "docs don't mention..." | ❌ |
| 1.0 | "ask forums..." | ❌ |

**None got SIGUSR1 correct.**

#### "What are the magic bytes?"

| Gain | Response | Correct? |
|------|----------|----------|
| 0.0 | "4 bytes for I/O" | ❌ |
| 0.2 | "magic bytes are:" | ❓ |
| 0.4 | "port 8080" | ❌ |
| 0.6 | "base64 encoded" | ❌ |
| 0.8 | `0xDE 0xAD 0x5B...` | ⚠️ 2/4 |
| 1.0 | "Here is..." | ❓ |

---

## Key Problems Identified

### 1. Normalization Bug with Zero Base State

```python
if state_base is None:
    state_base = [torch.zeros_like(d) for d in delta]
```

When base is zeros, the normalization:
```python
base_norm_sq = torch.dot(s_base.flatten(), s_base.flatten())  # = 0!
if base_norm_sq > 1e-12:  # False, skips normalization
```

This means when applying delta to fresh state, normalization is skipped. But the zeros base has no "energy" to preserve, so the delta is applied at full magnitude. This may cause instability at high gains.

### 2. Context Injection Works Better Than Delta

At gain=0.0 (no delta), the model gets port **correct** because of entity context injection in prompt:
```python
entity_context = "SYSTEM DATA / GROUNDING:\n- 1. The Hydra API runs on Port 8088...\n\nInstruction: Prioritize the data above..."
```

The delta at higher gains actually **degrades** performance by interfering with the context processing.

### 3. RWKV State Captures "Vibes" Not Facts

RWKV's recurrent state encodes the "feel" of text more than specific factual information. The state after processing "Port 8088" doesn't explicitly store "8088" - it stores a distributed representation that influences probability distributions.

### 4. Recency Bias

Multi-epoch learning (epochs=10) reinforces the END of text more than the beginning because RWKV's state is dominated by recent tokens. Facts at the document start get "washed out."

---

## Recommendations

1. **Rely on Entity Context Injection**: The JSON sidecar approach works better than state deltas for factual recall.

2. **Smaller, Focused Deltas**: Create micro-deltas for specific facts instead of entire documents.

3. **Layer-Wise Injection**: RWKV v6 has 3 state tensors per layer. Perhaps only certain layers encode factual info.

4. **Lower Temperature**: Use temperature 0.1-0.2 for factual recall tasks.

5. **Fix Base State Issue**: When base is zeros, consider not normalizing or using a different initialization strategy.
