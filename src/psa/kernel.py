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

# Optimize for inference
os.environ["RWKV_JIT_ON"] = "1"
# Only enable CUDA if available
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
        """
        Initialize the Liquid Engine.

        Args:
            model_path: Path to .pth RWKV v6 model
            strategy: 'cuda fp16', 'cpu fp32', 'mps fp32', etc.
            tokenizer: 'auto', 'world', or 'pile' - auto detects from vocab size
        """
        print(f"[*] Loading Liquid Kernel: {model_path} ({strategy})")
        self.model = RWKV(model=model_path, strategy=strategy)
        self.model_path = model_path
        self.strategy = strategy

        # Model architecture info
        self.n_layer = self.model.args.n_layer
        self.n_embd = self.model.args.n_embd

        # Auto-detect tokenizer from vocab size
        vocab_size = self.model.args.n_embd  # This isn't right, need to check embedding
        # Check embedding weight shape to get actual vocab size
        emb_shape = self.model.w['emb.weight'].shape
        actual_vocab_size = emb_shape[0]

        # Resolve tokenizer path
        home = Path.home()
        pile_tokenizer_path = home / "models" / "20B_tokenizer.json"

        if tokenizer == "auto":
            if actual_vocab_size == 50277:
                tokenizer = str(pile_tokenizer_path)  # Pile tokenizer
                print(f"[*] Auto-detected Pile tokenizer (vocab={actual_vocab_size})")
            else:
                tokenizer = "rwkv_vocab_v20230424"  # World tokenizer
                print(f"[*] Auto-detected World tokenizer (vocab={actual_vocab_size})")
        elif tokenizer == "pile":
            tokenizer = str(pile_tokenizer_path)
        elif tokenizer == "world":
            tokenizer = "rwkv_vocab_v20230424"

        self.pipeline = PIPELINE(self.model, tokenizer)
        self._raw_tokenizer = self.pipeline.tokenizer
        self._is_pile_tokenizer = (actual_vocab_size == 50277)

        # Define Zero State (S_0)
        # RWKV uses 'None' to represent pure zero state during inference
        self.S_0 = None

        # Cache device/dtype to avoid scanning weights on every call (Fix #5)
        self._cached_device, self._cached_dtype = self._scan_device_dtype()

        print(f"[*] Model loaded: {self.n_layer} layers, {self.n_embd} dim")

    def _encode(self, text: str) -> List[int]:
        """Encode text to tokens, handling both tokenizer types."""
        if self._is_pile_tokenizer:
            return self._raw_tokenizer.encode(text).ids
        else:
            return self._raw_tokenizer.encode(text)

    def _decode(self, tokens: List[int]) -> str:
        """Decode tokens to text, handling both tokenizer types."""
        return self._raw_tokenizer.decode(tokens)

    def _scan_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        """
        Scan model weights to find device and dtype. Called once during __init__.
        """
        for key, tensor in self.model.w.items():
            # Skip embedding which may be on CPU
            if 'emb' in key:
                continue
            if isinstance(tensor, torch.Tensor):
                return tensor.device, tensor.dtype
        # Fallback to first tensor if no block weights found
        for tensor in self.model.w.values():
            if isinstance(tensor, torch.Tensor):
                return tensor.device, tensor.dtype
        # Ultimate fallback
        return torch.device('cpu'), torch.float32

    def _get_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        """
        Get cached device and dtype. O(1) operation.
        """
        return self._cached_device, self._cached_dtype

    def _get_zero_state(self) -> List[torch.Tensor]:
        """
        Create an explicit zero state tensor list.
        RWKV v6 state structure: 3 tensors per layer (time_mix_state, channel_mix_state, time_first_state)
        """
        device, dtype = self._get_device_dtype()

        state = []
        for _ in range(self.n_layer):
            # Time mix state
            state.append(torch.zeros(self.n_embd, device=device, dtype=dtype))
            # Channel mix state
            state.append(torch.zeros(self.n_embd, device=device, dtype=dtype))
            # Time first state (for RWKV v6)
            state.append(torch.zeros(self.n_embd, device=device, dtype=dtype))

        return state

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
        Run inference while maintaining/updating state.
        Uses chunked forward pass for prompt processing (Fix #1).

        Returns:
            Tuple of (generated_text, new_state)
        """
        # Pre-tokenize stop tokens for O(1) lookup (Fix #2)
        stop_token_ids = set()
        if stop_tokens:
            for st in stop_tokens:
                encoded = self._encode(st)
                if encoded:
                    stop_token_ids.add(encoded[0])

        # Encode context
        tokens = self._encode(ctx)

        # Process context through model in chunks (Fix #1)
        current_state = state
        with torch.no_grad():
            # Chunked forward pass for prompt - much faster than token-by-token
            for i in range(0, len(tokens), chunk_len):
                chunk = tokens[i:i + chunk_len]
                out, current_state = self.model.forward(chunk, current_state)

        # Generate new tokens (must be token-by-token for autoregressive sampling)
        generated_tokens = []
        for _ in range(token_count):
            # Sample next token
            probs = torch.softmax(out, dim=-1)

            if temperature > 0:
                # Apply temperature and top_p sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()

                next_token = sorted_indices[torch.multinomial(sorted_probs, 1)].item()
            else:
                next_token = torch.argmax(probs).item()

            generated_tokens.append(next_token)

            # Check stop condition using pre-tokenized IDs (Fix #2)
            if next_token in stop_token_ids:
                break
            # Fallback: also check decoded text for multi-char stop tokens
            if stop_tokens:
                decoded = self._decode([next_token])
                if any(s in decoded for s in stop_tokens):
                    break

            # Forward pass for next token
            out, current_state = self.model.forward([next_token], current_state)

        generated_text = self._decode(generated_tokens)
        return generated_text, current_state

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
        This effectively 'compresses' the text into the vector space.
        Uses chunked forward pass for efficiency (Fix #1b).

        Multi-epoch processing reinforces early content (like port numbers)
        by passing state from end of epoch N to start of epoch N+1.

        Args:
            text: Text to learn
            initial_state: Starting state (None = S_0)
            progress_callback: Optional callback(current_token, total_tokens)
            epochs: Number of times to process the text (default: 1)
            chunk_len: Number of tokens per forward pass (default: 256)

        Returns:
            Final state after processing all tokens across all epochs
        """
        tokens = self._encode(text)
        tokens_per_epoch = len(tokens)
        total_tokens = tokens_per_epoch * epochs
        state = initial_state
        processed = 0

        # Process tokens across epochs using chunked forward passes
        with torch.no_grad():
            for epoch in range(epochs):
                # Process in chunks for efficiency
                for i in range(0, len(tokens), chunk_len):
                    chunk = tokens[i:i + chunk_len]
                    out, state = self.model.forward(chunk, state)
                    processed += len(chunk)

                    if progress_callback:
                        progress_callback(processed, total_tokens)

            # Final callback
            if progress_callback:
                progress_callback(total_tokens, total_tokens)

        return state

    def compute_delta(
        self,
        state_final: List[torch.Tensor],
        state_base: Optional[List[torch.Tensor]] = None,
        clone: bool = False
    ) -> List[torch.Tensor]:
        """
        Math: ΔS = S_final - S_base
        Extracts the 'Skill Vector' from the raw state.

        Args:
            state_final: State after learning
            state_base: Base state (None = S_0, treated as zeros)
            clone: If True, clone tensors. If False, return references (Fix #3)

        Returns:
            Delta state representing learned knowledge
        """
        if state_base is None:
            # If base is S_0 (zeros), Delta is just S_final
            # Only clone if explicitly requested (Fix #3)
            if clone:
                return [s.clone() if s is not None else None for s in state_final]
            return state_final

        delta_state = []
        for s_f, s_b in zip(state_final, state_base):
            # Handle RWKV state parts (some might be None or different shapes)
            if s_f is None:
                delta_state.append(None)
                continue
            if s_b is None:
                delta_state.append(s_f.clone() if clone else s_f)
                continue

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
        Injects a skill into the base mind.

        Energy Normalization prevents "exploding state" where unbounded
        addition causes the model to output garbage. By preserving the
        base state's L2 norm, we keep the state in a stable manifold.

        Destructive Interference Detection: If the angle between base
        and delta is near 180 degrees (cosine similarity < -0.9), the
        states cancel out causing instability. A warning is printed.

        Args:
            state_base: Base state to modify (None = start from zeros)
            delta: Skill vector to apply
            weight: Scaling factor for the skill
            normalize: If True, preserve base state's energy (L2 norm)
            check_interference: If True, check for destructive interference (slower)

        Returns:
            New state with skill applied
        """
        if state_base is None:
            # Initialize zeros using delta as template
            state_base = [torch.zeros_like(d) if d is not None else None for d in delta]

        new_state = []
        destructive_count = 0

        for s_base, s_delta in zip(state_base, delta):
            if s_delta is None:
                new_state.append(s_base)
                continue
            if s_base is None:
                # If base was None but we have a delta, result is delta * weight
                new_state.append(s_delta * weight)
                continue

            # Perform addition first (always needed)
            result = s_base + (s_delta * weight)

            # Energy normalization: preserve base state's L2 norm (Fix #4)
            # Compute norms only when needed, avoid redundant .item() calls
            if normalize:
                base_norm_sq = torch.dot(s_base.flatten(), s_base.flatten())
                if base_norm_sq > 1e-12:
                    result_norm_sq = torch.dot(result.flatten(), result.flatten())
                    if result_norm_sq > 1e-12:
                        # Use sqrt ratio: base_norm / result_norm = sqrt(base_norm_sq / result_norm_sq)
                        scale = torch.sqrt(base_norm_sq / result_norm_sq)
                        result = result * scale

            # Check for destructive interference only if requested (Fix #4)
            if check_interference:
                base_norm = torch.norm(s_base).item()
                delta_norm = torch.norm(s_delta).item()
                if base_norm > 1e-6 and delta_norm > 1e-6:
                    cos_sim = torch.dot(s_base.flatten(), s_delta.flatten()).item() / (base_norm * delta_norm)
                    if cos_sim < -0.9:
                        destructive_count += 1

            new_state.append(result)

        # Warn about destructive interference
        if destructive_count > 0:
            print(f"[!] WARNING: Destructive Interference detected in {destructive_count}/{len(delta)} tensors (cos_sim < -0.9)")
            print(f"[!] This may cause unstable output. Consider adjusting gain or using a different delta.")

        return new_state

    def merge_skills(
        self,
        skill_weights: Dict[str, float]
    ) -> List[torch.Tensor]:
        """
        Math: S_composite = Σ (w_i * ΔS_i)
        Mixes multiple skills (e.g. 0.8*Medical + 0.2*Python).

        Args:
            skill_weights: Dict mapping skill file paths to weights

        Returns:
            Composite state with all skills merged
        """
        composite_state = None

        for skill_path, weight in skill_weights.items():
            delta_tensor = self.load_state(skill_path)
            composite_state = self.apply_delta(composite_state, delta_tensor, weight)

        return composite_state

    def interpolate_states(
        self,
        state_a: List[torch.Tensor],
        state_b: List[torch.Tensor],
        alpha: float = 0.5
    ) -> List[torch.Tensor]:
        """
        Spherical linear interpolation between two states.

        Math: S_interp = (1 - α) * S_a + α * S_b

        Args:
            state_a: First state
            state_b: Second state
            alpha: Interpolation factor (0 = state_a, 1 = state_b)

        Returns:
            Interpolated state
        """
        result = []
        for s_a, s_b in zip(state_a, state_b):
            if s_a is None or s_b is None:
                result.append(s_a if s_a is not None else s_b)
                continue
            result.append((1 - alpha) * s_a + alpha * s_b)
        return result

    def subtract_skill(
        self,
        state: List[torch.Tensor],
        skill_path: str,
        weight: float = 1.0
    ) -> List[torch.Tensor]:
        """
        Remove a skill from state (inverse of apply_delta).

        Math: S_new = S - (weight * ΔS_skill)

        Useful for "unlearning" or removing unwanted behaviors.
        """
        skill_delta = self.load_state(skill_path)
        return self.apply_delta(state, skill_delta, weight=-weight)

    def save_state(self, state: List[torch.Tensor], filepath: str):
        """Save state to .psa file."""
        # Move to CPU for portable storage
        cpu_state = []
        for s in state:
            if s is not None:
                cpu_state.append(s.cpu())
            else:
                cpu_state.append(None)

        torch.save({
            'state': cpu_state,
            'n_layer': self.n_layer,
            'n_embd': self.n_embd,
            'version': 'psa_v3',
        }, filepath)

    def load_state(self, filepath: str) -> List[torch.Tensor]:
        """Load state from .psa/.delta file."""
        data = torch.load(filepath, map_location='cpu', weights_only=False)

        # Handle both old format (just list) and new format (dict)
        if isinstance(data, dict):
            state = data['state']
        else:
            state = data

        # Move to model device using robust detection
        device, dtype = self._get_device_dtype()

        loaded_state = []
        for s in state:
            if s is not None:
                loaded_state.append(s.to(device=device, dtype=dtype))
            else:
                loaded_state.append(None)

        return loaded_state

    def get_state_stats(self, state: List[torch.Tensor]) -> Dict:
        """Get statistics about a state for debugging."""
        stats = {
            'num_tensors': len(state),
            'non_null': sum(1 for s in state if s is not None),
            'total_params': 0,
            'norms': [],
        }

        for s in state:
            if s is not None:
                stats['total_params'] += s.numel()
                stats['norms'].append(float(s.norm().item()))

        stats['avg_norm'] = sum(stats['norms']) / len(stats['norms']) if stats['norms'] else 0
        stats['size_mb'] = stats['total_params'] * 2 / (1024 * 1024)  # fp16

        return stats
