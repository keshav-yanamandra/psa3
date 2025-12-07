"""
PSA v3 Skill Mixer - "DJ Deck" for Knowledge Composition
---------------------------------------------------------
Provides weighted mixing of skill vectors for composite agent capabilities.

Core Concept:
    S_composite = Σ(w_i * ΔS_i)

    Mix "Python Expert" (weight=0.8) + "Legacy Hydra Docs" (weight=1.0)
    = Agent that can fix legacy system using Python

Usage:
    from psa.skills import SkillMixer
    from psa.kernel import LiquidAgent

    kernel = LiquidAgent("path/to/model.pth")
    mixer = SkillMixer(kernel)

    composite = mixer.mix_skills({
        "~/.psa/skills/python_expert.psa": 0.8,
        "~/.psa/skills/hydra_docs.psa": 1.0,
    })
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import torch


class SkillMixer:
    """
    Mix multiple skills with weighted composition.

    Think of it like a DJ deck - you have multiple skill tracks,
    and you can blend them at different volumes to create the
    perfect mix for your task.
    """

    def __init__(self, kernel):
        """
        Initialize SkillMixer with a LiquidAgent kernel.

        Args:
            kernel: LiquidAgent instance for state operations
        """
        self.kernel = kernel
        self._mix_history: List[Dict[str, Any]] = []

    def mix_skills(
        self,
        skill_manifest: Dict[str, float],
        base_state: Optional[List[torch.Tensor]] = None
    ) -> Optional[List[torch.Tensor]]:
        """
        Mix multiple skills with specified weights.

        Args:
            skill_manifest: Dict mapping skill paths to weights
                           e.g., {"python.psa": 0.8, "hydra.psa": 1.0}
            base_state: Optional base state to start from (default: None/zero)

        Returns:
            Composite state tensor, or None if no skills loaded

        Example:
            composite = mixer.mix_skills({
                "python_expert.psa": 0.8,
                "legacy_docs.psa": 1.0,
                "incident_logs.psa": 0.5,
            })
        """
        composite_state = base_state
        loaded_skills = []

        for skill_path_str, weight in skill_manifest.items():
            path = Path(skill_path_str).expanduser()

            if not path.exists():
                print(f"  [WARN] Skill not found: {path}")
                continue

            # Load skill delta
            delta = torch.load(str(path), map_location="cpu", weights_only=False)

            # Apply with weight
            composite_state = self.kernel.apply_delta(
                composite_state,
                delta,
                weight=weight
            )

            loaded_skills.append({
                "path": str(path),
                "name": path.stem,
                "weight": weight,
            })

        # Record mix history
        self._mix_history.append({
            "skills": loaded_skills,
            "total_skills": len(loaded_skills),
        })

        return composite_state

    def mix_from_names(
        self,
        skills: Dict[str, float],
        skills_dir: Optional[Path] = None
    ) -> Optional[List[torch.Tensor]]:
        """
        Mix skills by name (looks up in skills directory).

        Args:
            skills: Dict mapping skill names to weights
                   e.g., {"python": 0.8, "hydra_docs": 1.0}
            skills_dir: Directory containing skill files (default: ~/.psa/skills)

        Returns:
            Composite state tensor
        """
        if skills_dir is None:
            skills_dir = Path.home() / ".psa" / "skills"

        # Convert names to full paths
        manifest = {}
        for name, weight in skills.items():
            skill_path = skills_dir / f"{name}.psa"
            manifest[str(skill_path)] = weight

        return self.mix_skills(manifest)

    def get_mix_history(self) -> List[Dict[str, Any]]:
        """Get history of all mixes performed."""
        return self._mix_history

    def preview_mix(
        self,
        skill_manifest: Dict[str, float],
        skills_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Preview a mix without actually performing it.

        Shows what skills would be loaded and their weights.

        Args:
            skill_manifest: Dict mapping skill names to weights
            skills_dir: Directory containing skill files

        Returns:
            Preview information dict
        """
        if skills_dir is None:
            skills_dir = Path.home() / ".psa" / "skills"

        preview = {
            "skills": [],
            "total_weight": 0.0,
            "missing": [],
        }

        for name, weight in skill_manifest.items():
            skill_path = skills_dir / f"{name}.psa"

            if skill_path.exists():
                size_mb = skill_path.stat().st_size / (1024 * 1024)
                preview["skills"].append({
                    "name": name,
                    "weight": weight,
                    "size_mb": round(size_mb, 2),
                    "exists": True,
                })
                preview["total_weight"] += weight
            else:
                preview["missing"].append(name)
                preview["skills"].append({
                    "name": name,
                    "weight": weight,
                    "exists": False,
                })

        return preview


def quick_mix(
    kernel,
    skills: Dict[str, float],
    skills_dir: Optional[Path] = None
) -> Optional[List[torch.Tensor]]:
    """
    Convenience function for quick skill mixing.

    Args:
        kernel: LiquidAgent instance
        skills: Dict mapping skill names to weights
        skills_dir: Optional skills directory

    Returns:
        Composite state tensor

    Example:
        state = quick_mix(kernel, {"python": 0.8, "devops": 1.0})
    """
    mixer = SkillMixer(kernel)
    return mixer.mix_from_names(skills, skills_dir)
