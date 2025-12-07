"""
PSA v3 - Cognitive Delta System
-------------------------------
The "Liquid Mixer" for inference-time knowledge injection.

Cognitive Deltas are "crystallized memories" - vector differences between
a knowledgeable state and the base state (Tabula Rasa).

Core Concept:
    ΔS = S_final - S_0  (The "Imprinting" step)
    S_active = S_0 + Σ(α_i * ΔS_i)  (The "Injection" step)

Unlike RAG (which retrieves text to read), Cognitive Deltas retrieve
brain states for the model to "feel."

Usage:
    from psa.deltas import DeltaMixer
    from psa.kernel import LiquidAgent

    kernel = LiquidAgent("path/to/model.pth")
    mixer = DeltaMixer(kernel)

    # Inject multiple deltas with gain factors
    active_state = mixer.inject({
        "python_expert": 0.8,    # Suppress slightly
        "hydra_protocol": 1.2,   # Amplify
    })
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import torch


class DeltaMixer:
    """
    The Liquid Mixer - runtime kernel for Cognitive Delta injection.

    Loads the Base Model state and dynamically injects requested
    Deltas into the active memory buffer before generation.

    Think of it like an audio mixing board:
    - Each Delta is a "track"
    - Each Gain Factor is a "fader" (0.0 = muted, 1.0 = unity, 2.0 = amplified)
    - The output is the mixed "active state"
    """

    def __init__(self, kernel):
        """
        Initialize DeltaMixer with a LiquidAgent kernel.

        Args:
            kernel: LiquidAgent instance (the Imprinter)
        """
        self.kernel = kernel
        self._injection_history: List[Dict[str, Any]] = []

    def inject(
        self,
        delta_manifest: Dict[str, float],
        base_state: Optional[List[torch.Tensor]] = None,
        deltas_dir: Optional[Path] = None
    ) -> Optional[List[torch.Tensor]]:
        """
        Inject multiple Cognitive Deltas with gain factors.

        This is the core "Liquid" operation - mathematically composing
        multiple knowledge domains at inference time.

        Args:
            delta_manifest: Dict mapping delta names to gain factors
                           e.g., {"python": 0.8, "hydra": 1.2}
            base_state: Optional base state (default: Tabula Rasa / None)
            deltas_dir: Directory containing .delta files

        Returns:
            Active state tensor with all deltas injected

        Example:
            active = mixer.inject({
                "devops_expert": 1.0,     # Full strength
                "incident_logs": 0.8,     # Slightly suppressed
                "python_scripting": 1.5,  # Amplified
            })
        """
        if deltas_dir is None:
            deltas_dir = Path.home() / ".psa" / "deltas"

        active_state = base_state
        injected = []

        for delta_name, gain in delta_manifest.items():
            # Try both .delta and .psa extensions for compatibility
            delta_path = deltas_dir / f"{delta_name}.delta"
            if not delta_path.exists():
                delta_path = deltas_dir / f"{delta_name}.psa"

            if not delta_path.exists():
                print(f"  [WARN] Delta not found: {delta_name}")
                continue

            # Load the cognitive delta using kernel's load_state (handles device transfer)
            delta = self.kernel.load_state(str(delta_path))

            # Inject with gain factor
            active_state = self.kernel.apply_delta(
                active_state,
                delta,
                weight=gain  # Gain factor maps to weight
            )

            injected.append({
                "name": delta_name,
                "gain": gain,
                "path": str(delta_path),
            })

        # Record injection history
        self._injection_history.append({
            "deltas": injected,
            "total_injected": len(injected),
        })

        return active_state

    def inject_from_paths(
        self,
        manifest: Dict[str, float],
        base_state: Optional[List[torch.Tensor]] = None
    ) -> Optional[List[torch.Tensor]]:
        """
        Inject deltas from full file paths.

        Args:
            manifest: Dict mapping file paths to gain factors
            base_state: Optional base state

        Returns:
            Active state with deltas injected
        """
        active_state = base_state
        injected = []

        for path_str, gain in manifest.items():
            path = Path(path_str).expanduser()

            if not path.exists():
                print(f"  [WARN] Delta file not found: {path}")
                continue

            # Load using kernel's load_state (handles device transfer)
            delta = self.kernel.load_state(str(path))
            active_state = self.kernel.apply_delta(active_state, delta, weight=gain)

            injected.append({
                "path": str(path),
                "name": path.stem,
                "gain": gain,
            })

        self._injection_history.append({
            "deltas": injected,
            "total_injected": len(injected),
        })

        return active_state

    def get_injection_history(self) -> List[Dict[str, Any]]:
        """Get history of all injection operations."""
        return self._injection_history

    def preview_injection(
        self,
        delta_manifest: Dict[str, float],
        deltas_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Preview an injection without performing it.

        Args:
            delta_manifest: Dict mapping delta names to gain factors
            deltas_dir: Directory containing delta files

        Returns:
            Preview information including file sizes and availability
        """
        if deltas_dir is None:
            deltas_dir = Path.home() / ".psa" / "deltas"

        preview = {
            "deltas": [],
            "total_gain": 0.0,
            "missing": [],
            "estimated_size_mb": 0.0,
        }

        for name, gain in delta_manifest.items():
            delta_path = deltas_dir / f"{name}.delta"
            if not delta_path.exists():
                delta_path = deltas_dir / f"{name}.psa"

            if delta_path.exists():
                size_mb = delta_path.stat().st_size / (1024 * 1024)
                preview["deltas"].append({
                    "name": name,
                    "gain": gain,
                    "size_mb": round(size_mb, 2),
                    "exists": True,
                })
                preview["total_gain"] += gain
                preview["estimated_size_mb"] += size_mb
            else:
                preview["missing"].append(name)
                preview["deltas"].append({
                    "name": name,
                    "gain": gain,
                    "exists": False,
                })

        preview["estimated_size_mb"] = round(preview["estimated_size_mb"], 2)
        return preview


class CognitiveDelta:
    """
    Represents a single Cognitive Delta (crystallized memory).

    A Delta is the vector difference between a knowledgeable state
    and the Tabula Rasa (base state).
    """

    def __init__(
        self,
        name: str,
        state: List[torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.state = state
        self.metadata = metadata or {}

    @classmethod
    def from_file(cls, path: Path) -> "CognitiveDelta":
        """Load a delta from a .delta file."""
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        # Handle both dict format (new) and list format (legacy)
        if isinstance(data, dict):
            state = data['state']
            metadata = {k: v for k, v in data.items() if k != 'state'}
        else:
            state = data
            metadata = {}
        return cls(name=path.stem, state=state, metadata=metadata)

    def save(self, path: Path):
        """Save delta to file."""
        torch.save(self.state, str(path))

    @property
    def size_mb(self) -> float:
        """Estimated size in megabytes."""
        total_bytes = sum(
            t.numel() * t.element_size() for t in self.state if t is not None
        )
        return total_bytes / (1024 * 1024)


def quick_inject(
    kernel,
    deltas: Dict[str, float],
    deltas_dir: Optional[Path] = None
) -> Optional[List[torch.Tensor]]:
    """
    Convenience function for quick delta injection.

    Args:
        kernel: LiquidAgent instance
        deltas: Dict mapping delta names to gain factors
        deltas_dir: Optional deltas directory

    Returns:
        Active state with deltas injected

    Example:
        state = quick_inject(kernel, {"python": 0.8, "devops": 1.2})
    """
    mixer = DeltaMixer(kernel)
    return mixer.inject(deltas, deltas_dir=deltas_dir)


# Backwards compatibility aliases
SkillMixer = DeltaMixer  # Alias for old code
quick_mix = quick_inject  # Alias for old code
